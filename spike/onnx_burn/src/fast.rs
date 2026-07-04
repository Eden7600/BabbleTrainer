// Fast data path: rayon-parallel decode up front + background-prefetch batchers so the GPU stays fed
// (batch size unchanged — larger batches hurt expression quality, an optimization-dynamics effect).
use std::collections::HashSet;
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use burn::module::AutodiffModule;
use burn::module::ParamId;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use rayon::prelude::*;

use crate::aug;
use crate::data::{process_eye, Bin, Sample, S, T};
use crate::ema::ema_update;
use crate::flat_opt::{Filter, FlatAdamW, FlatEma};
use crate::gaze::MicroChad;
use crate::heads::ConsensusHeads;
use crate::losses::{consensus_term, evenness, mt_consistency, supervised_deconfuse, temporal, ExprConsts};
use crate::mnv4::Mnv4;
use crate::sampler::{gaze_weights, Rng, Weighted};

pub struct Frames {
    data: Vec<Option<Vec<u8>>>, // [rec*2 + side] -> processed 128x128 u8, None if blank
}

impl Frames {
    pub fn decode(bin: &Bin) -> Self {
        let n = bin.frames.len() * 2;
        let data = (0..n).into_par_iter().map(|k| process_eye(bin, k / 2, k % 2)).collect();
        Frames { data }
    }
    fn get(&self, rec: usize, side: usize) -> Option<&[u8]> {
        self.data[rec * 2 + side].as_deref()
    }
    fn stack_into(&self, rec: usize, side: usize, out: &mut Vec<f32>) {
        for k in 0..T {
            let f = if rec >= k { self.get(rec - k, side) } else { None };
            match f {
                Some(px) => out.extend(px.iter().map(|&v| v as f32 / 255.0)),
                None => out.extend(std::iter::repeat(0.0).take(S * S)),
            }
        }
    }
    fn stack_f32(&self, rec: usize, side: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(T * S * S);
        self.stack_into(rec, side, &mut out);
        out
    }
}

// Mixup source pools (expr only): neutral = max expr label < 0.2, expr = max expr label > 0.8, all =
// the full train subset for general cross-frame blends.
pub struct Pools {
    neutral: Vec<usize>,
    expr: Vec<usize>,
    all: Vec<usize>,
}
impl Pools {
    fn expr_build(samples: &[Sample], train: &[usize]) -> Self {
        let m = |i: usize| samples[i].label[2..6].iter().cloned().fold(0.0f32, f32::max);
        Self {
            neutral: train.iter().cloned().filter(|&i| m(i) < 0.2).collect(),
            expr: train.iter().cloned().filter(|&i| m(i) > 0.8).collect(),
            all: train.to_vec(),
        }
    }
    fn empty() -> Self {
        Self { neutral: vec![], expr: vec![], all: vec![] }
    }
}

fn raw_gaze(bin: &Bin, rec: usize, side: usize) -> (f32, f32) {
    let f = &bin.frames[rec].fields;
    if side == 0 { (f[5], f[6]) } else { (f[7], f[8]) }
}

// Paired-eye index: records where BOTH sides are in `train`, with the 4-dim expr labels of each eye.
// Drives the evenness (L/R symmetry) and consensus-head terms. active-boost weighted like the expr pool.
pub struct Paired {
    recs: Vec<usize>,
    ll: Vec<[f32; 4]>,
    rl: Vec<[f32; 4]>,
    cum: Vec<f32>,
    total: f32,
}
impl Paired {
    fn build(samples: &[Sample], train: &[usize]) -> Self {
        use std::collections::HashMap;
        let (mut left, mut right) = (HashMap::new(), HashMap::new());
        for &i in train {
            let s = &samples[i];
            let e = [s.label[2], s.label[3], s.label[4], s.label[5]];
            if s.side == 0 { left.insert(s.rec, e); } else { right.insert(s.rec, e); }
        }
        let (mut recs, mut ll, mut rl, mut cum) = (vec![], vec![], vec![], vec![]);
        let mut acc = 0.0f32;
        let mut keys: Vec<usize> = left.keys().cloned().filter(|k| right.contains_key(k)).collect();
        keys.sort_unstable();
        for rec in keys {
            let (l, r) = (left[&rec], right[&rec]);
            let m = l.iter().chain(r.iter()).cloned().fold(0.0f32, f32::max);
            acc += 1.0 + 4.0 * m;
            recs.push(rec);
            ll.push(l);
            rl.push(r);
            cum.push(acc);
        }
        Self { recs, ll, rl, cum, total: acc }
    }
    fn sample(&self, rng: &mut Rng) -> usize {
        let u = rng.f01() * self.total;
        self.cum.partition_point(|&c| c < u).min(self.cum.len().saturating_sub(1))
    }
    fn is_empty(&self) -> bool {
        self.recs.is_empty()
    }
}

// Unlabeled anti-jitter stream = the capture's own FLAG_EXPR_UNLABELED (free-expression gaze-section)
// frames — the same in-domain footage the frontend uses. Yields (cur_strong, cur_weak, nbr_weak,
// pair_ok) for the mean-teacher + temporal-smoothness losses (stage B).
pub struct Unlabeled {
    items: Vec<(usize, usize)>, // (rec, side)
}
impl Unlabeled {
    fn build(samples: &[Sample], train: &[usize]) -> Self {
        let items = train
            .iter()
            .filter_map(|&i| {
                let s = &samples[i];
                (s.expr_unlabeled > 0.5).then_some((s.rec, s.side))
            })
            .collect();
        Self { items }
    }
    fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

type UnlabBatch = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>); // cur_strong, cur_weak, nbr_weak, pair_ok

fn spawn_unlabeled_prefetch(
    frames: Arc<Frames>, unlab: Arc<Unlabeled>, present: Arc<HashSet<(usize, usize)>>, ub: usize,
    workers: usize, seed: u64,
) -> Receiver<UnlabBatch> {
    let (tx, rx) = sync_channel::<UnlabBatch>(workers * 2);
    for wi in 0..workers.max(1) {
        let (frames, unlab, present, tx) = (frames.clone(), unlab.clone(), present.clone(), tx.clone());
        thread::spawn(move || {
            let mut rng = Rng::new(seed.wrapping_add((wi as u64).wrapping_mul(0x85EBCA77)));
            let mut scratch = Vec::new();
            loop {
                let (mut cs, mut cw, mut nw, mut ok) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for _ in 0..ub {
                    let (rec, side) = unlab.items[rng.usize(unlab.items.len())];
                    let mut a = frames.stack_f32(rec, side);
                    aug::aug_stack_expr(&mut a, &mut rng, &mut scratch); // strong view
                    let g = aug::sample_geom_weak(&mut rng); // shared geom for cur/nbr weak views
                    let mut b = frames.stack_f32(rec, side);
                    aug::aug_stack_weak(&mut b, &g, &mut rng);
                    // Only pair with the previous frame if it is a real GOOD, decoded same-side frame
                    // (present in the sample set). Otherwise the temporal hinge would train against a
                    // blank / non-good neighbor; mask it out (pair_ok=0, self as placeholder).
                    let has_nbr = rec >= 1 && present.contains(&(rec - 1, side));
                    let pair_ok = if has_nbr { 1.0 } else { 0.0 };
                    let mut c = frames.stack_f32(if has_nbr { rec - 1 } else { rec }, side);
                    aug::aug_stack_weak(&mut c, &g, &mut rng);
                    cs.extend_from_slice(&a);
                    cw.extend_from_slice(&b);
                    nw.extend_from_slice(&c);
                    ok.push(pair_ok);
                }
                if tx.send((cs, cw, nw, ok)).is_err() {
                    break;
                }
            }
        });
    }
    rx
}

type PairedBatch = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>); // pl, pr, pll(4), prl(4)

fn spawn_paired_prefetch(
    frames: Arc<Frames>, paired: Arc<Paired>, pbatch: usize, workers: usize, seed: u64,
) -> Receiver<PairedBatch> {
    let (tx, rx) = sync_channel::<PairedBatch>(workers * 2);
    for wi in 0..workers.max(1) {
        let (frames, paired, tx) = (frames.clone(), paired.clone(), tx.clone());
        thread::spawn(move || {
            let mut rng = Rng::new(seed.wrapping_add((wi as u64).wrapping_mul(0x9E3779B1)));
            let mut scratch = Vec::new();
            loop {
                let (mut pl, mut pr, mut pll, mut prl) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for _ in 0..pbatch {
                    let k = paired.sample(&mut rng);
                    let rec = paired.recs[k];
                    let mut a = frames.stack_f32(rec, 0);
                    aug::aug_stack_expr(&mut a, &mut rng, &mut scratch);
                    let mut b = frames.stack_f32(rec, 1);
                    aug::aug_stack_expr(&mut b, &mut rng, &mut scratch);
                    pl.extend_from_slice(&a);
                    pr.extend_from_slice(&b);
                    pll.extend_from_slice(&paired.ll[k]);
                    prl.extend_from_slice(&paired.rl[k]);
                }
                if tx.send((pl, pr, pll, prl)).is_err() {
                    break;
                }
            }
        });
    }
    rx
}

pub fn build_samples(bin: &Bin, frames: &Frames) -> Vec<Sample> {
    let mut out = Vec::new();
    for rec in 0..bin.frames.len() {
        let fr = &bin.frames[rec];
        if !fr.good() {
            continue;
        }
        let (l, r) = fr.labels();
        let (gv, eu) = (fr.gaze_valid(), fr.expr_unlabeled());
        for (side, lab) in [(0usize, l), (1usize, r)] {
            if frames.get(rec, side).is_some() {
                let sticky = rec > 0 && raw_gaze(bin, rec - 1, side) == raw_gaze(bin, rec, side);
                let gaze_usable = gv > 0.5 && !sticky;
                out.push(Sample { rec, side, label: lab, gaze_valid: gv, expr_unlabeled: eu, gaze_usable });
            }
        }
    }
    out
}

type Batch = (Vec<f32>, Vec<f32>);

fn build_batch(frames: &Frames, samples: &[Sample], idxs: &[usize], expr: bool) -> Batch {
    let b = idxs.len();
    let m = if expr { 4 } else { 2 };
    let mut xs = Vec::with_capacity(b * T * S * S);
    let mut ys = Vec::with_capacity(b * m);
    for &i in idxs {
        let s = &samples[i];
        frames.stack_into(s.rec, s.side, &mut xs);
        if expr {
            ys.extend_from_slice(&s.label[2..6]);
        } else {
            ys.extend_from_slice(&s.label[0..2]);
        }
    }
    (xs, ys)
}

// Training batch with augmentation. Expr: 50% neutral<->expression mixup, 15% cross-frame blend, else
// strong photometric+geometric aug (the linearity + generalization drivers). Gaze: BabbleTrainer gaze
// aug. Eval uses the plain build_batch (no aug).
#[allow(clippy::too_many_arguments)]
fn build_batch_train(
    frames: &Frames, samples: &[Sample], idxs: &[usize], expr: bool, rng: &mut Rng, pools: &Pools,
    scratch: &mut Vec<f32>,
) -> Batch {
    let b = idxs.len();
    let m = if expr { 4 } else { 2 };
    let mut xs = Vec::with_capacity(b * T * S * S);
    let mut ys = Vec::with_capacity(b * m);
    for &i in idxs {
        let s = &samples[i];
        if !expr {
            let mut stack = frames.stack_f32(s.rec, s.side);
            aug::aug_stack_gaze(&mut stack, rng, scratch);
            xs.extend_from_slice(&stack);
            ys.extend_from_slice(&s.label[0..2]);
            continue;
        }
        let r = rng.f01();
        let neutral_mix = r < 0.5 && !pools.neutral.is_empty() && !pools.expr.is_empty();
        let cross_mix = !neutral_mix && r < 0.65 && pools.all.len() > 1;
        if neutral_mix || cross_mix {
            let (ia, ib) = if neutral_mix {
                (pools.neutral[rng.usize(pools.neutral.len())], pools.expr[rng.usize(pools.expr.len())])
            } else {
                (pools.all[rng.usize(pools.all.len())], pools.all[rng.usize(pools.all.len())])
            };
            let lam = rng.f01();
            let (sa, sb) = (&samples[ia], &samples[ib]);
            let fa = frames.stack_f32(sa.rec, sa.side);
            let fb = frames.stack_f32(sb.rec, sb.side);
            xs.extend_from_slice(&aug::mixup_stacks(&fa, &fb, lam, rng, scratch));
            for c in 0..4 {
                ys.push((1.0 - lam) * sa.label[2 + c] + lam * sb.label[2 + c]);
            }
        } else {
            let mut stack = frames.stack_f32(s.rec, s.side);
            aug::aug_stack_expr(&mut stack, rng, scratch);
            xs.extend_from_slice(&stack);
            ys.extend_from_slice(&s.label[2..6]);
        }
    }
    (xs, ys)
}

fn spawn_prefetch(
    frames: Arc<Frames>, samples: Arc<Vec<Sample>>, w: Arc<Weighted>, batch: usize, workers: usize,
    expr: bool, seed: u64, pools: Arc<Pools>,
) -> Receiver<Batch> {
    let (tx, rx) = sync_channel::<Batch>(workers * 2);
    for wi in 0..workers.max(1) {
        let (frames, samples, w, tx, pools) =
            (frames.clone(), samples.clone(), w.clone(), tx.clone(), pools.clone());
        thread::spawn(move || {
            let mut rng = Rng::new(seed.wrapping_add((wi as u64).wrapping_mul(2654435761)));
            let mut scratch = Vec::new();
            loop {
                let idxs: Vec<usize> = (0..batch).map(|_| w.sample(&mut rng)).collect();
                let b = build_batch_train(&frames, &samples, &idxs, expr, &mut rng, &pools, &mut scratch);
                if tx.send(b).is_err() {
                    break;
                }
            }
        });
    }
    rx
}

fn pearson(p: &[f32], t: &[f32]) -> f32 {
    let n = p.len() as f32;
    let (mp, mt) = (p.iter().sum::<f32>() / n, t.iter().sum::<f32>() / n);
    let (mut cov, mut vp, mut vt) = (0.0, 0.0, 0.0);
    for i in 0..p.len() {
        let (dp, dt) = (p[i] - mp, t[i] - mt);
        cov += dp * dt;
        vp += dp * dp;
        vt += dt * dt;
    }
    if vp.sqrt() < 1e-6 || vt.sqrt() < 1e-6 { 0.0 } else { cov / (vp.sqrt() * vt.sqrt()) }
}

fn cos_lr(step: usize, steps: usize, warmup: usize, lr: f64) -> f64 {
    if step < warmup {
        lr * (step + 1) as f64 / warmup as f64
    } else {
        let p = (step - warmup) as f64 / (steps - warmup).max(1) as f64;
        lr * 0.5 * (1.0 + (std::f64::consts::PI * p).cos())
    }
}

pub struct FastCfg {
    pub steps: usize,
    pub batch: usize,
    pub warmup: usize,
    pub val_every: usize,
    pub lr: f64,
    pub wd: f32,
    pub workers: usize,
}

// Pretrained-backbone init (BabbleTrainer build_expr_net(pretrained=True)): load the timm
// MNv4-conv-small-050 weights (4-ch stem + 4-out head already adapted) so the per-user calibration
// starts from ImageNet features instead of random. mnv4_parity.rs proves this load matches timm
// block-by-block. Returns true if the weights were found and loaded.
fn load_pretrained_expr<B: Backend>(model: &mut Mnv4<B>) -> bool {
    for path in ["../mnv4_weights.safetensors", "mnv4_weights.safetensors"] {
        if std::path::Path::new(path).exists() {
            let mut store = SafetensorsStore::from_file(path)
                .with_from_adapter(PyTorchToBurnAdapter)
                .with_key_remapping(r"(bn1|bn|norm_head)\.weight$", "${1}.gamma")
                .with_key_remapping(r"(bn1|bn|norm_head)\.bias$", "${1}.beta")
                .allow_partial(true)
                .validate(false);
            match model.load_from(&mut store) {
                Ok(res) => {
                    println!("[expr] pretrained init from {path}: {res}");
                    return true;
                }
                Err(e) => {
                    println!("[expr] pretrained load failed ({e}); random init");
                    return false;
                }
            }
        }
    }
    println!("[expr] no pretrained weights found; random init");
    false
}

fn eval_expr<BI: Backend>(model: &Mnv4<BI>, frames: &Frames, samples: &[Sample], val: &[usize], device: &BI::Device) -> (f32, [f32; 4]) {
    let (mut preds, mut tgts) = (Vec::new(), Vec::new());
    for chunk in val.chunks(256) {
        let (xs, ys) = build_batch(frames, samples, chunk, true);
        let x = Tensor::<BI, 4>::from_data(TensorData::new(xs, [chunk.len(), T, S, S]), device);
        preds.extend(model.forward(x).clamp(0.0, 1.0).into_data().to_vec::<f32>().unwrap());
        tgts.extend(ys);
    }
    let mse = preds.iter().zip(&tgts).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / preds.len() as f32;
    let mut r = [0.0; 4];
    for c in 0..4 {
        let pc: Vec<f32> = preds.iter().skip(c).step_by(4).cloned().collect();
        let tc: Vec<f32> = tgts.iter().skip(c).step_by(4).cloned().collect();
        r[c] = pearson(&pc, &tc);
    }
    (mse, r)
}

// Returns (val MSE in normalized [0,1] space, MAE in degrees, per-channel Pearson r). Best-checkpoint
// selection is on MSE to match BabbleTrainer (evaluate_gaze selects on `((p-t)**2).mean()`); MAE(deg)
// is reported for readability but not used for selection.
fn eval_gaze<BI: Backend>(model: &MicroChad<BI>, frames: &Frames, samples: &[Sample], val: &[usize], device: &BI::Device) -> (f32, f32, [f32; 2]) {
    let sel: Vec<usize> = val.iter().cloned().filter(|&i| samples[i].gaze_usable).collect();
    let sel = if sel.is_empty() { val.to_vec() } else { sel };
    let (mut preds, mut tgts) = (Vec::new(), Vec::new());
    for chunk in sel.chunks(256) {
        let (xs, ys) = build_batch(frames, samples, chunk, false);
        let x = Tensor::<BI, 4>::from_data(TensorData::new(xs, [chunk.len(), T, S, S]), device);
        preds.extend(model.forward(x).into_data().to_vec::<f32>().unwrap());
        tgts.extend(ys);
    }
    let n = preds.len() as f32;
    let mse = preds.iter().zip(&tgts).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / n;
    let mae = preds.iter().zip(&tgts).map(|(a, b)| (a - b).abs()).sum::<f32>() / n * 90.0;
    let mut r = [0.0; 2];
    for c in 0..2 {
        let pc: Vec<f32> = preds.iter().skip(c).step_by(2).cloned().collect();
        let tc: Vec<f32> = tgts.iter().skip(c).step_by(2).cloned().collect();
        r[c] = pearson(&pc, &tc);
    }
    (mse, mae, r)
}

// Consensus-head val metric: MSE of the DEPLOYED fused outputs (widen/squint/brow) on paired val
// records, so best-checkpoint selection can prefer a step where the consensus is actually trained.
// Runs teacher.forward_features (no host readback of grads — inference only) + the consensus heads.
fn eval_consensus<BI: Backend>(
    teacher: &Mnv4<BI>, cons: &ConsensusHeads<BI>, frames: &Frames, pv: &Paired, device: &BI::Device,
) -> f32 {
    if pv.is_empty() {
        return 0.0;
    }
    let (mut se, mut cnt) = (0.0f32, 0usize);
    let idxs: Vec<usize> = (0..pv.recs.len()).collect();
    for chunk in idxs.chunks(256) {
        let b = chunk.len();
        let (mut xl, mut xr, mut tgt) = (Vec::new(), Vec::new(), Vec::new());
        for &k in chunk {
            let rec = pv.recs[k];
            frames.stack_into(rec, 0, &mut xl);
            frames.stack_into(rec, 1, &mut xr);
            let (l, r) = (pv.ll[k], pv.rl[k]); // expr 4-vec = [lid,widen,squint,brow]; consensus = mean L/R
            tgt.push(0.5 * (l[1] + r[1])); // widen
            tgt.push(0.5 * (l[2] + r[2])); // squint
            tgt.push(0.5 * (l[3] + r[3])); // brow
        }
        let xl = Tensor::<BI, 4>::from_data(TensorData::new(xl, [b, T, S, S]), device);
        let xr = Tensor::<BI, 4>::from_data(TensorData::new(xr, [b, T, S, S]), device);
        let fl = teacher.forward_features(xl);
        let fr = teacher.forward_features(xr);
        let ws = cons.forward(0, fl.clone(), fr.clone()); // [b,2] widen,squint
        let br = cons.forward(1, fl, fr); // [b,1] brow
        let pred = Tensor::cat(vec![ws, br], 1).into_data().to_vec::<f32>().unwrap();
        for (p, t) in pred.iter().zip(&tgt) {
            se += (p - t).powi(2);
            cnt += 1;
        }
    }
    se / cnt.max(1) as f32
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
) -> f32 {
    let scale = (cfg.batch as f64 / 64.0).sqrt();
    let a_steps = cfg.steps / 2;
    let b_steps = cfg.steps - a_steps;
    let warmup_a = (a_steps / 10).max(1);
    let (lr_wu_head, a_lr_head, a_lr_bb) = (1e-3 * scale, 5e-4 * scale, 1e-4 * scale);
    let (b_lr_head, b_lr_bb) = (2e-4 * scale, 5e-5 * scale);
    let decay = 1.0 - 0.001 * (cfg.batch as f64 / 64.0); // per-step EMA decay (config EXPR_EMA_DECAY)

    let w = Arc::new(Weighted::expr(&samples, train, 4.0));
    let pools = Arc::new(Pools::expr_build(&samples, train));
    let rx = spawn_prefetch(frames.clone(), samples.clone(), w, cfg.batch, cfg.workers, true, 0xE47, pools);

    // Paired-eye stream for evenness (both stages) + consensus (stage B). pbatch = batch/2 (reference
    // EXPR_EVEN_BATCH). Falls back to expr-only if the capture has no records with both eyes valid.
    let paired = Arc::new(Paired::build(&samples, train));
    let do_paired = !paired.is_empty();
    let paired_val = Paired::build(&samples, val); // held-out paired eyes for the consensus val metric
    let pbatch = (cfg.batch / 2).max(1);
    let prx = do_paired.then(|| {
        spawn_paired_prefetch(frames.clone(), paired.clone(), pbatch, cfg.workers, 0x9A17)
    });
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
    let urx = do_unlab.then(|| spawn_unlabeled_prefetch(frames.clone(), unlab.clone(), present.clone(), ub, cfg.workers, 0x51A9));
    println!("[expr] paired={} ({} recs) unlabeled={} ({} frames)", do_paired, paired.recs.len(), do_unlab, unlab.items.len());

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
        // per-stage LR schedule + freeze + EMA stride
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

        // One student forward over the concatenated mega-batch (matches reference xcat): the supervised
        // batch, the paired eyes (evenness), and the stage-B unlabeled strong/weak/neighbor views
        // (mean-teacher + temporal). Auxiliary streams cost forward rows but never touch the regime.
        let t0 = Instant::now();
        let (mut sdata, ys) = rx.recv().unwrap();
        let mut rows = cfg.batch;

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
        // Pack ALL labels (supervised y, paired pll/prl, temporal pair_ok) into one host buffer so they
        // upload in a single h2d and are sliced on-device (OPT #12). Offsets are in ELEMENTS.
        let mut lbl = ys;
        let y_end = lbl.len(); // cfg.batch * 4
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
        pt[0] += t0.elapsed().as_secs_f32();

        let t0 = Instant::now();
        let lbl_t = Tensor::<B, 1>::from_data(TensorData::new(lbl, [lbl_len]), device);
        let y = lbl_t.clone().slice([0..y_end]).reshape([cfg.batch, 4]);
        // Upload the mega-batch ONCE as an inner (non-autodiff) tensor; the student consumes it via
        // from_inner (inputs need no grad) and the teacher slices it directly — no duplicate h2d.
        let big_i = Tensor::<B::InnerBackend, 4>::from_data(TensorData::new(sdata, [rows, T, S, S]), device);
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
        let need_pair_feats = stage_b && paired_b.is_some();
        let tfeats = if need_pair_feats || unlab_b.is_some() {
            let mut segs: Vec<Tensor<B::InnerBackend, 4>> = Vec::with_capacity(3);
            if need_pair_feats {
                segs.push(big_i.clone().slice([off_pl..off_pl + pbatch]));
                segs.push(big_i.clone().slice([off_pr..off_pr + pbatch]));
            }
            if unlab_b.is_some() {
                segs.push(big_i.clone().slice([off_cw..off_cw + ub]));
            }
            Some(teacher.forward_features(Tensor::cat(segs, 0)))
        } else {
            None
        };
        let cw_base = if need_pair_feats { 2 * pbatch } else { 0 };

        // Evenness (both stages) + consensus (stage B) on the paired eyes.
        let mut cons_step = false;
        if paired_b.is_some() {
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
        if unlab_b.is_some() {
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
