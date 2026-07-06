use std::collections::HashSet;
use std::sync::mpsc::sync_channel;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use burn::module::{AutodiffModule, ParamId};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;

use crate::data::{Sample, S, T};
use crate::ema::ema_update;
use crate::flat_opt::{Filter, FlatAdamW, FlatEma};
use crate::gaze::MicroChad;
use crate::gpu_aug::AugBackend;
use crate::heads::ConsensusHeads;
use crate::losses::{consensus_term, evenness, mt_consistency, supervised_deconfuse, temporal, ExprConsts};
use crate::mnv4::Mnv4;
use crate::sampler::{gaze_weights, Weighted};

use super::evalhelp::{cos_lr, eval_consensus, eval_expr, eval_gaze, load_pretrained_expr, load_pretrained_gaze};
use super::frames::Frames;
use super::index::{Paired, Pools, Unlabeled};
use super::prefetch::{
    spawn_paired_prefetch, spawn_paired_prefetch_spec, spawn_prefetch, spawn_prefetch_spec,
    spawn_unlabeled_prefetch, spawn_unlabeled_prefetch_spec,
};

#[derive(Clone)]
pub struct FastCfg {
    pub steps: usize,
    pub batch: usize,
    pub warmup: usize,
    pub val_every: usize,
    pub lr: f64,
    pub wd: f32,
    pub workers: usize,
}

// 2-stage expression curriculum (BabbleTrainer fit_expr_stages), pretrained-backbone init, and
// backbone/head param-group LRs with a frozen-backbone warmup. Total budget = cfg.steps (split 50/50
// into stage A + stage B), batch = cfg.batch, so the per-user regime is preserved. Reference LRs are
// batch-64 values scaled by sqrt(batch/64), matching config.py's own batch-scaling rule.
//
// Stage A: supervised (deconfuse) + mixup + evenness; backbone frozen for the warmup, head-only at a
// higher LR; then unfreeze with backbone 1e-4 / head 5e-4 (cosine); EMA stride 25.
// Stage B: init continues from A (teacher reset); backbone 5e-5 / head 2e-4 (cosine); EMA stride 1;
// adds the consensus head, mean-teacher consistency, and temporal smoothness on the unlabeled stream.
pub fn fit_expr_staged<B: AutodiffBackend>(
    frames: Arc<Frames>, samples: Arc<Vec<Sample>>, train: &[usize], val: &[usize], cfg: &FastCfg,
    device: &B::Device, ckpt: &str,
) -> f32
where
    B::InnerBackend: AugBackend,
{
    let scale = (cfg.batch as f64 / 64.0).sqrt();
    let a_steps = cfg.steps / 2;
    let b_steps = cfg.steps - a_steps;
    let warmup_a = (a_steps / 10).max(1);
    let (lr_wu_head, a_lr_head, a_lr_bb) = (1e-3 * scale, 5e-4 * scale, 1e-4 * scale);
    let (b_lr_head, b_lr_bb) = (2e-4 * scale, 5e-5 * scale);
    let decay = 1.0 - 0.001 * (cfg.batch as f64 / 64.0); // per-step EMA decay (config EXPR_EMA_DECAY)

    let w = Arc::new(Weighted::expr(&samples, train, 4.0));
    let pools = Arc::new(Pools::expr_build(&samples, train));
    // GPU-aug tiering: BB_GPU_AUG=0 forces the CPU-aug path; =1 forces device-resident frames;
    // =2 forces the per-step tier (upload only the step's ~5 MB of unique frames, ids remapped);
    // unset = auto — resident if the whole frame store fits the VRAM budget (BB_VRAM_MB, default
    // 2000 MB for the store itself), else per-step (fused kernels either way, identical output).
    #[derive(PartialEq, Clone, Copy)]
    enum AugTier { Cpu, Resident, PerStep }
    let frames_mb = frames.raw().len() * S * S / 4 * 4 / (1 << 20);
    let vram_budget_mb: usize = std::env::var("BB_VRAM_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(2000);
    let aug_tier = match std::env::var("BB_GPU_AUG").ok().as_deref() {
        Some("0") => AugTier::Cpu,
        Some("1") => AugTier::Resident,
        Some("2") => AugTier::PerStep,
        _ if frames_mb <= vram_budget_mb => AugTier::Resident,
        _ => AugTier::PerStep,
    };
    let gpu_aug_on = aug_tier != AugTier::Cpu;
    match aug_tier {
        AugTier::Cpu => println!("[expr] gpu-aug OFF — CPU aug path"),
        AugTier::PerStep => println!("[expr] gpu-aug PER-STEP (frames {frames_mb} MB > budget {vram_budget_mb} MB or BB_GPU_AUG=2)"),
        AugTier::Resident => {}
    }

    // Paired-eye stream for evenness (both stages) + consensus (stage B). pbatch = batch/2 (reference
    // EXPR_EVEN_BATCH). Falls back to expr-only if the capture has no records with both eyes valid.
    let paired = Arc::new(Paired::build(&samples, train));
    let do_paired = !paired.is_empty();
    let paired_val = Paired::build(&samples, val); // held-out paired eyes for the consensus val metric
    let pbatch = (cfg.batch / 2).max(1);

    // Explicit consensus checkpoint path — never a blind replace that could collide with the expr ckpt
    // if it lacks "expr". Delete any stale one up front so a run that trains no consensus (no paired
    // data) can't leave an old cons that export_bridge would wrongly fuse into a 15-dim brow.
    let cons_ckpt = if ckpt.contains("expr") { ckpt.replace("expr", "cons") } else { format!("{ckpt}.cons") };
    let _ = std::fs::remove_file(&cons_ckpt);

    // Semi-supervised anti-jitter stream (mean-teacher + temporal), stage B only. ub = batch.
    let unlab = Arc::new(Unlabeled::build(&samples, train));
    let do_unlab = !unlab.is_empty();
    let ub = cfg.batch;
    // (rec,side) of every GOOD decoded eye — lets the temporal stream reject blank/non-good neighbors.
    let present: Arc<HashSet<(usize, usize)>> = Arc::new(samples.iter().map(|s| (s.rec, s.side)).collect());
    println!("[expr] paired={} ({} recs) unlabeled={} ({} frames) gpu_aug={}", do_paired, paired.recs.len(), do_unlab, unlab.items.len(), gpu_aug_on);

    // Assembler thread: receives the three prefetch streams, concatenates the mega-batch + packed
    // labels, and uploads BOTH tensors to the device off the critical thread (double-buffered).
    struct StepBatch<BI: Backend> {
        big_i: Tensor<BI, 4>,
        lbl: Tensor<BI, 1>,
        y_end: usize,
        off_pl: usize,
        off_pr: usize,
        off_cs: usize,
        off_cw: usize,
        off_nw: usize,
        pll_off: usize,
        prl_off: usize,
        ok_off: usize,
        has_paired: bool,
        has_unlab: bool,
    }
    let arx = if gpu_aug_on {
        // GPU-aug arm: spec workers + on-device mega-batch build. Resident tier uploads the whole
        // frame store once; per-step tier packs+uploads only each step's unique frames (~5 MB).
        let gframes = (aug_tier == AugTier::Resident).then(|| {
            let g = <B::InnerBackend as AugBackend>::aug_upload(frames.raw(), device);
            println!("[expr] frames resident on device: {} eyes (~{} MB u8-packed)", frames.raw().len(), frames.raw().len() * S * S / (1 << 20));
            g
        });
        let aframes = frames.clone();
        let rx = spawn_prefetch_spec(frames.clone(), samples.clone(), w, cfg.batch, cfg.workers, 0xE47, pools);
        let prx = do_paired.then(|| spawn_paired_prefetch_spec(frames.clone(), paired.clone(), pbatch, cfg.workers, 0x9A17));
        let urx = do_unlab.then(|| spawn_unlabeled_prefetch_spec(frames.clone(), unlab.clone(), present.clone(), ub, cfg.workers, 0x51A9));
        let (atx, arx) = sync_channel::<StepBatch<B::InnerBackend>>(2);
        let adev = device.clone();
        let batch = cfg.batch;
        thread::spawn(move || {
            for step in 0.. {
                let stage_b = step >= a_steps;
                let (mut specs, ys) = match rx.recv() {
                    Ok(b) => b,
                    Err(_) => break,
                };
                let mut rows = batch;
                let paired_b = prx.as_ref().map(|p| p.recv().unwrap());
                let (mut off_pl, mut off_pr) = (0usize, 0usize);
                if let Some((sl, sr, _, _)) = &paired_b {
                    off_pl = rows;
                    specs.extend_from_slice(sl);
                    rows += pbatch;
                    off_pr = rows;
                    specs.extend_from_slice(sr);
                    rows += pbatch;
                }
                let unlab_b = if stage_b { urx.as_ref().map(|u| u.recv().unwrap()) } else { None };
                let (mut off_cs, mut off_cw, mut off_nw) = (0usize, 0usize, 0usize);
                if let Some((cs, cw, nw, _)) = &unlab_b {
                    off_cs = rows;
                    specs.extend_from_slice(cs);
                    rows += ub;
                    off_cw = rows;
                    specs.extend_from_slice(cw);
                    rows += ub;
                    off_nw = rows;
                    specs.extend_from_slice(nw);
                    rows += ub;
                }
                let mut lbl = ys;
                let y_end = lbl.len();
                let (mut pll_off, mut prl_off, mut ok_off) = (0usize, 0usize, 0usize);
                if let Some((_, _, plld, prld)) = &paired_b {
                    pll_off = lbl.len();
                    lbl.extend_from_slice(plld);
                    prl_off = lbl.len();
                    lbl.extend_from_slice(prld);
                }
                if let Some((_, _, _, ok)) = &unlab_b {
                    ok_off = lbl.len();
                    lbl.extend_from_slice(ok);
                }
                let lbl_len = lbl.len();
                let has_paired = paired_b.is_some();
                let has_unlab = unlab_b.is_some();
                debug_assert_eq!(specs.len(), rows);
                let lbl = Tensor::<B::InnerBackend, 1>::from_data(TensorData::new(lbl, [lbl_len]), &adev);
                let big_i = match &gframes {
                    Some(g) => <B::InnerBackend as AugBackend>::aug_batch(g, &specs, &adev),
                    None => <B::InnerBackend as AugBackend>::aug_batch_step(aframes.raw(), &specs, &adev),
                };
                let sb = StepBatch {
                    big_i, lbl, y_end, off_pl, off_pr, off_cs, off_cw, off_nw,
                    pll_off, prl_off, ok_off, has_paired, has_unlab,
                };
                if atx.send(sb).is_err() {
                    break;
                }
            }
        });
        arx
    } else {
        let rx = spawn_prefetch(frames.clone(), samples.clone(), w, cfg.batch, cfg.workers, true, 0xE47, pools);
        let prx = do_paired.then(|| spawn_paired_prefetch(frames.clone(), paired.clone(), pbatch, cfg.workers, 0x9A17));
        let urx = do_unlab.then(|| spawn_unlabeled_prefetch(frames.clone(), unlab.clone(), present.clone(), ub, cfg.workers, 0x51A9));
        let (atx, arx) = sync_channel::<StepBatch<B::InnerBackend>>(2);
        let adev = device.clone();
        let batch = cfg.batch;
        thread::spawn(move || {
            for step in 0.. {
                let stage_b = step >= a_steps;
                let (mut sdata, ys) = match rx.recv() {
                    Ok(b) => b,
                    Err(_) => break,
                };
                let mut rows = batch;
                let paired_b = prx.as_ref().map(|p| p.recv().unwrap());
                let (mut off_pl, mut off_pr) = (0usize, 0usize);
                if let Some((pld, prd, _, _)) = &paired_b {
                    off_pl = rows;
                    sdata.extend_from_slice(pld);
                    rows += pbatch;
                    off_pr = rows;
                    sdata.extend_from_slice(prd);
                    rows += pbatch;
                }
                let unlab_b = if stage_b { urx.as_ref().map(|u| u.recv().unwrap()) } else { None };
                let (mut off_cs, mut off_cw, mut off_nw) = (0usize, 0usize, 0usize);
                if let Some((cs, cw, nw, _)) = &unlab_b {
                    off_cs = rows;
                    sdata.extend_from_slice(cs);
                    rows += ub;
                    off_cw = rows;
                    sdata.extend_from_slice(cw);
                    rows += ub;
                    off_nw = rows;
                    sdata.extend_from_slice(nw);
                    rows += ub;
                }
                // Pack ALL labels (supervised y, paired pll/prl, temporal pair_ok) into one buffer,
                // one h2d, sliced on-device (OPT #12). Offsets are in ELEMENTS.
                let mut lbl = ys;
                let y_end = lbl.len();
                let (mut pll_off, mut prl_off, mut ok_off) = (0usize, 0usize, 0usize);
                if let Some((_, _, plld, prld)) = &paired_b {
                    pll_off = lbl.len();
                    lbl.extend_from_slice(plld);
                    prl_off = lbl.len();
                    lbl.extend_from_slice(prld);
                }
                if let Some((_, _, _, ok)) = &unlab_b {
                    ok_off = lbl.len();
                    lbl.extend_from_slice(ok);
                }
                let lbl_len = lbl.len();
                let has_paired = paired_b.is_some();
                let has_unlab = unlab_b.is_some();
                let lbl = Tensor::<B::InnerBackend, 1>::from_data(TensorData::new(lbl, [lbl_len]), &adev);
                // Upload in per-stream chunks and cat on-device: shorter client-lock holds, so the
                // training thread's dispatches interleave instead of stalling behind one 42MB write.
                let mut segs: Vec<Tensor<B::InnerBackend, 4>> = Vec::new();
                let mut off = 0usize;
                let plan = [batch, pbatch, pbatch, ub, ub, ub];
                let mut planned = 0usize;
                for (k, &n) in plan.iter().enumerate() {
                    let present = match k {
                        0 => true,
                        1 | 2 => has_paired,
                        _ => has_unlab,
                    };
                    if !present {
                        continue;
                    }
                    let elems = n * T * S * S;
                    let part: Vec<f32> = sdata[off..off + elems].to_vec();
                    segs.push(Tensor::from_data(TensorData::new(part, [n, T, S, S]), &adev));
                    off += elems;
                    planned += n;
                }
                debug_assert_eq!(planned, rows);
                let big_i = if segs.len() == 1 {
                    segs.pop().unwrap()
                } else {
                    Tensor::cat(segs, 0)
                };
                let sb = StepBatch {
                    big_i, lbl, y_end, off_pl, off_pr, off_cs, off_cw, off_nw,
                    pll_off, prl_off, ok_off, has_paired, has_unlab,
                };
                if atx.send(sb).is_err() {
                    break;
                }
            }
        });
        arx
    };

    let mut model = Mnv4::<B>::new(T, 4, device);
    if load_pretrained_expr(&mut model) {
        model.small_init_head(device); // reference re-inits the head small after loading the backbone
    }
    let head_ids = model.head_ids();
    let bn_ids: HashSet<ParamId> = model.running_state_ids().into_iter().collect();
    let mut opt_head = AdamWConfig::new().with_weight_decay(cfg.wd).init();
    let mut opt_bb = AdamWConfig::new().with_weight_decay(cfg.wd).init();
    // BB_FLAT_OPT=1: fused multi-tensor AdamW+EMA (flat arenas; numerically identical, wgpu-only).
    let flat_on = std::env::var("BB_FLAT_OPT").ok().as_deref() == Some("1");
    let head_set: HashSet<ParamId> = head_ids.iter().copied().collect();
    let mut f_head = FlatAdamW::<B::InnerBackend>::new(cfg.wd, Filter::Include(head_set.clone()));
    let mut f_bb = FlatAdamW::<B::InnerBackend>::new(cfg.wd, Filter::Exclude(head_set));
    let mut f_ema = FlatEma::<B::InnerBackend>::new();
    // Consensus heads (feat_dim = classifier input = 1280); trained in stage B on detached teacher feats.
    let mut cons = ConsensusHeads::<B>::new(1280, device);
    let mut opt_cons = AdamWConfig::new().with_weight_decay(cfg.wd).init();

    let mut teacher = model.valid();
    let mut best = f32::INFINITY;
    let mut win = Instant::now();
    let cos = |p: f64| 0.5 * (1.0 + (std::f64::consts::PI * p).cos());
    let lc = ExprConsts::<B>::new(device); // per-step loss constants, built once (not from_floats/step)

    // BB_PROF=1: per-phase host-time breakdown (dispatch cost) + an end-of-step sync so GPU tail time
    // shows up as `sync` instead of hiding inside the next step's first blocking call.
    // BB_PROF=2: micro mode — device sync after EVERY phase, so each bucket = dispatch + GPU execution
    // of that phase (serializes the pipeline; compare against mode 1 + unprofiled wall for truth).
    let prof_lvl: u8 = std::env::var("BB_PROF").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let prof = prof_lvl >= 1;
    let micro = prof_lvl >= 2;
    let psync = |on: bool| { if on { let _ = B::sync(device); } };
    let mut pt = [0f32; 8]; // recv h2d fwd aux bwd opt ema sync

    for step in 0..cfg.steps {
        let stage_b = step >= a_steps;
        let (lr_head, lr_bb, frozen, ema_every) = if !stage_b {
            let s = step;
            if s < warmup_a {
                (lr_wu_head, 0.0, true, 25usize) // frozen head-only warmup
            } else {
                let p = (s - warmup_a) as f64 / (a_steps - warmup_a).max(1) as f64;
                (a_lr_head * cos(p), a_lr_bb * cos(p), false, 25)
            }
        } else {
            let s = step - a_steps;
            let p = s as f64 / b_steps.max(1) as f64;
            (b_lr_head * cos(p), b_lr_bb * cos(p), false, 1)
        };
        if stage_b && step == a_steps {
            teacher = model.valid(); // fresh teacher at stage B (matches reference EMATeacher reset)
            f_ema.reset(); // flat arena re-captured from the fresh teacher on next update
            if do_paired {
                // The deployed 15-dim model fuses brow to the consensus head, which ONLY trains in
                // stage B. Reset best so the exported checkpoint is selected from stage B, pairing the
                // saved teacher with a TRAINED consensus — a stage-A best would export a near-init
                // consensus and a flat deployed brow. See audit/BABBLEBURN_VS_GOALPOST.md.
                best = f32::INFINITY;
            }
        }

        // Mega-batch assembly + h2d happen on the assembler thread (double-buffered); the critical
        // thread just receives prebuilt device tensors and lifts them into autodiff (free ops).
        let t0 = Instant::now();
        let sb = arx.recv().unwrap();
        pt[0] += t0.elapsed().as_secs_f32();

        let t0 = Instant::now();
        let (off_pl, off_pr) = (sb.off_pl, sb.off_pr);
        let (off_cs, off_cw, off_nw) = (sb.off_cs, sb.off_cw, sb.off_nw);
        let (pll_off, prl_off, ok_off) = (sb.pll_off, sb.prl_off, sb.ok_off);
        let lbl_t = Tensor::<B, 1>::from_inner(sb.lbl);
        let y = lbl_t.clone().slice([0..sb.y_end]).reshape([cfg.batch, 4]);
        let big_i = sb.big_i;
        psync(micro);
        pt[1] += t0.elapsed().as_secs_f32();

        let t0 = Instant::now();
        let out = model.forward(Tensor::from_inner(big_i.clone()));
        let mut loss = supervised_deconfuse(out.clone().slice([0..cfg.batch]), y, &lc.deconf);
        psync(micro);
        pt[2] += t0.elapsed().as_secs_f32();
        let t0 = Instant::now();

        // ONE folded teacher inference pass over every row that needs teacher features (pl, pr, cw),
        // kept on-device: from_inner lifts the results into autodiff as constant leaves (detached),
        // so grads still flow only into the consensus heads / student — no host readback.
        let need_pair_feats = stage_b && sb.has_paired;
        let tfeats = if need_pair_feats || sb.has_unlab {
            let mut segs: Vec<Tensor<B::InnerBackend, 4>> = Vec::with_capacity(3);
            if need_pair_feats {
                segs.push(big_i.clone().slice([off_pl..off_pl + pbatch]));
                segs.push(big_i.clone().slice([off_pr..off_pr + pbatch]));
            }
            if sb.has_unlab {
                segs.push(big_i.clone().slice([off_cw..off_cw + ub]));
            }
            Some(teacher.forward_features(Tensor::cat(segs, 0)))
        } else {
            None
        };
        let cw_base = if need_pair_feats { 2 * pbatch } else { 0 };

        // Evenness (both stages) + consensus (stage B) on the paired eyes.
        let mut cons_step = false;
        if sb.has_paired {
            let out_pl = out.clone().slice([off_pl..off_pl + pbatch]);
            let out_pr = out.clone().slice([off_pr..off_pr + pbatch]);
            let pll = lbl_t.clone().slice([pll_off..pll_off + pbatch * 4]).reshape([pbatch, 4]);
            let prl = lbl_t.clone().slice([prl_off..prl_off + pbatch * 4]).reshape([pbatch, 4]);
            loss = loss + evenness(out_pl, out_pr, pll.clone(), prl.clone(), &lc.even_w);
            if stage_b {
                let tf = tfeats.as_ref().unwrap();
                let fl = Tensor::from_inner(tf.clone().slice([0..pbatch]));
                let fr = Tensor::from_inner(tf.clone().slice([pbatch..2 * pbatch]));
                let tgt = (pll + prl).mul_scalar(0.5);
                let g0 = tgt.clone().slice([0..pbatch, 1..3]); // widen, squint
                let g1 = tgt.slice([0..pbatch, 3..4]); // brow
                loss = loss + consensus_term(cons.forward(0, fl.clone(), fr.clone()), g0);
                loss = loss + consensus_term(cons.forward(1, fl, fr), g1);
                cons_step = true;
            }
        }

        // Mean-teacher (strong<->weak consistency, quadratic ramp) + temporal smoothness (adjacent
        // frames within margin), stage B, on the unlabeled in-domain stream.
        if sb.has_unlab {
            let cs_out = out.clone().slice([off_cs..off_cs + ub]);
            let cw_out = out.clone().slice([off_cw..off_cw + ub]);
            let nw_out = out.clone().slice([off_nw..off_nw + ub]);
            let cw_f = tfeats.as_ref().unwrap().clone().slice([cw_base..cw_base + ub]);
            let t_out = Tensor::from_inner(teacher.classify(cw_f));
            let rampup = (b_steps / 4).max(1) as f64;
            let lam = (((step - a_steps) as f64 / rampup).min(1.0).powi(2)) as f32; // lambda_cons = 1.0
            loss = loss + mt_consistency(cs_out, t_out).mul_scalar(lam);
            let ok_t = lbl_t.clone().slice([ok_off..ok_off + ub]);
            loss = loss + temporal(cw_out, nw_out, ok_t, &lc.margin).mul_scalar(0.1); // lambda_temp = 0.1
        }
        psync(micro);
        pt[3] += t0.elapsed().as_secs_f32();

        let t0 = Instant::now();
        let mut grads = loss.backward();
        psync(micro);
        pt[4] += t0.elapsed().as_secs_f32();
        let t0 = Instant::now();
        let head_gp = GradientsParams::from_params(&mut grads, &model, &head_ids);
        let bb_gp = GradientsParams::from_module(&mut grads, &model); // remaining = backbone
        if flat_on {
            f_bb.ensure_init(&model, &bb_gp); // frozen steps too: the flat EMA needs the bb arena
            model = f_head.step(lr_head, model, head_gp);
            if !frozen {
                model = f_bb.step(lr_bb, model, bb_gp);
            }
        } else {
            model = opt_head.step(lr_head, model, head_gp);
            if !frozen {
                model = opt_bb.step(lr_bb, model, bb_gp);
            }
        }
        if cons_step {
            let cons_gp = GradientsParams::from_module(&mut grads, &cons);
            cons = opt_cons.step(lr_head, cons, cons_gp);
        }
        psync(micro);
        pt[5] += t0.elapsed().as_secs_f32();

        let t0 = Instant::now();
        if step % ema_every == 0 {
            let keep = decay.powi(ema_every as i32) as f32;
            teacher = if flat_on {
                f_ema.update(teacher, &model.valid(), &[&f_head, &f_bb], keep, &bn_ids)
            } else {
                ema_update::<B::InnerBackend, _>(teacher, &model.valid(), keep, &bn_ids)
            };
        }
        psync(micro);
        pt[6] += t0.elapsed().as_secs_f32();
        if prof {
            let t0 = Instant::now();
            let _ = out.slice([0..1, 0..1]).into_data(); // force queue drain: GPU tail time
            pt[7] += t0.elapsed().as_secs_f32();
        }
        if step % 100 == 99 {
            let dt = win.elapsed().as_secs_f32();
            println!("[expr {step:5}] {:.1} ms/step ({}) windowed", dt * 10.0, if stage_b { "B" } else { "A" });
            if prof {
                println!("[prof {step:5}] recv={:.1} h2d={:.1} fwd={:.1} aux={:.1} bwd={:.1} opt={:.1} ema={:.1} sync={:.1} ms/step",
                    pt[0] * 10.0, pt[1] * 10.0, pt[2] * 10.0, pt[3] * 10.0, pt[4] * 10.0, pt[5] * 10.0, pt[6] * 10.0, pt[7] * 10.0);
                pt = [0.0; 8];
            }
            win = Instant::now();
        }
        if step % cfg.val_every == 0 || step == cfg.steps - 1 {
            // Fold the latest student into a TEMPORARY teacher for eval/deploy, without committing it —
            // so val_every no longer perturbs the persistent EMA trajectory (BUGS #7: was a committed
            // fold on top of the scheduled one). The persistent `teacher` keeps its own stride.
            let keep = decay.powi(ema_every as i32) as f32;
            let eval_teacher = ema_update::<B::InnerBackend, _>(teacher.clone(), &model.valid(), keep, &bn_ids);
            let (mse, r) = eval_expr(&eval_teacher, &frames, &samples, val, device);
            // In stage B the deployed model is the fused 15-dim output, so select on main-head MSE PLUS
            // the consensus (widen/squint/brow) val MSE. This refuses the first stage-B step (consensus
            // near-init -> flat, high cons MSE) in favor of one where the consensus has actually trained,
            // and saves the teacher + same-step consensus together (BUGS #1). Both are MSE in [0,1].
            let cons_mse = if stage_b && do_paired {
                eval_consensus(&eval_teacher, &cons.valid(), &frames, &paired_val, device)
            } else {
                0.0
            };
            let score = mse + cons_mse;
            let tag = if score < best { "*" } else { "" };
            let st = if stage_b { "B" } else { "A" };
            println!("[expr {st} {step:5}/{}] mse={mse:.5} cons={cons_mse:.5} r({:.3} {:.3} {:.3} {:.3}) {tag}", cfg.steps, r[0], r[1], r[2], r[3]);
            if score < best {
                best = score;
                let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
                eval_teacher.save_file(ckpt, &rec).unwrap();
                if stage_b && do_paired {
                    cons.clone().save_file(&cons_ckpt, &rec).unwrap();
                }
            }
        }
    }
    best
}

pub fn fit_gaze_fast<B: AutodiffBackend>(
    frames: Arc<Frames>, samples: Arc<Vec<Sample>>, train: &[usize], val: &[usize], cfg: &FastCfg,
    device: &B::Device, ckpt: &str,
) -> f32 {
    let w = Arc::new(gaze_weights(&samples, train));
    let pools = Arc::new(Pools::empty());
    let rx = spawn_prefetch(frames.clone(), samples.clone(), w, cfg.batch, cfg.workers, false, 0x6A2E, pools);
    let mut model = MicroChad::<B>::new(T, 2, device);
    // Opt-in gaze warm-start (BB_GAZE_WARM=1) mirrors the deployed Python recipe (12k-step MicroChad
    // prior + per-user fine-tune). Default = cold start (Burn's regime trains gaze from scratch).
    if std::env::var("BB_GAZE_WARM").is_ok() {
        load_pretrained_gaze(&mut model);
    }
    let mut opt = AdamWConfig::new().with_weight_decay(cfg.wd).init();
    let mut best = f32::INFINITY; // best val MSE (selection metric, matches BabbleTrainer)
    let mut best_mae = f32::INFINITY; // MAE(deg) at the best-MSE checkpoint (for the summary line)
    let mut win = Instant::now();
    // Gaze deploys the STUDENT (BabbleTrainer only uses a gaze EMA teacher under semi-supervised
    // mean-teacher, wired separately).
    let prof_lvl: u8 = std::env::var("BB_PROF").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let prof = prof_lvl >= 1;
    let micro = prof_lvl >= 2;
    let psync = |on: bool| { if on { let _ = B::sync(device); } };
    let mut pt = [0f32; 5]; // recv h2d fwd bwd+opt sync
    for step in 0..cfg.steps {
        let t0 = Instant::now();
        let (xs, ys) = rx.recv().unwrap();
        pt[0] += t0.elapsed().as_secs_f32();
        let t0 = Instant::now();
        let x = Tensor::<B, 4>::from_data(TensorData::new(xs, [cfg.batch, T, S, S]), device);
        let y = Tensor::<B, 2>::from_data(TensorData::new(ys, [cfg.batch, 2]), device);
        psync(micro);
        pt[1] += t0.elapsed().as_secs_f32();
        let t0 = Instant::now();
        let loss = (model.forward(x) - y).powf_scalar(2.0).mean();
        psync(micro);
        pt[2] += t0.elapsed().as_secs_f32();
        let t0 = Instant::now();
        let gp = GradientsParams::from_grads(loss.backward(), &model);
        model = opt.step(cos_lr(step, cfg.steps, cfg.warmup, cfg.lr), model, gp);
        psync(micro);
        pt[3] += t0.elapsed().as_secs_f32();
        if prof {
            let t0 = Instant::now();
            let _ = loss.into_data(); // force queue drain: GPU tail time
            pt[4] += t0.elapsed().as_secs_f32();
        }
        if step % 100 == 99 {
            let dt = win.elapsed().as_secs_f32();
            println!("[gaze {step:5}] {:.1} steps/s ({:.1} ms/step) windowed", 100.0 / dt, dt * 10.0);
            if prof {
                println!("[gprof {step:5}] recv={:.1} h2d={:.1} fwd={:.1} bwd+opt={:.1} sync={:.1} ms/step",
                    pt[0] * 10.0, pt[1] * 10.0, pt[2] * 10.0, pt[3] * 10.0, pt[4] * 10.0);
                pt = [0.0; 5];
            }
            win = Instant::now();
        }
        if step % cfg.val_every == 0 || step == cfg.steps - 1 {
            let vm = model.valid();
            let (mse, mae, r) = eval_gaze(&vm, &frames, &samples, val, device);
            let tag = if mse < best { "*" } else { "" };
            println!("[gaze {step:5}/{}] mse={mse:.5} mae={mae:.2}deg r({:.3},{:.3}) {tag}", cfg.steps, r[0], r[1]);
            if mse < best {
                best = mse;
                best_mae = mae;
                vm.clone().save_file(ckpt, &NamedMpkFileRecorder::<FullPrecisionSettings>::new()).unwrap();
            }
        }
    }
    best_mae
}
