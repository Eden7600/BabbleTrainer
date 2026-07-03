// Fast data path: rayon-parallel decode up front + background-prefetch batchers so the GPU stays fed
// (batch size unchanged — larger batches hurt expression quality, an optimization-dynamics effect).
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use rayon::prelude::*;

use crate::aug;
use crate::data::{process_eye, Bin, Sample, S, T};
use crate::ema::ema_update;
use crate::gaze::MicroChad;
use crate::heads::ConsensusHeads;
use crate::losses::{consensus_term, evenness, mt_consistency, supervised_deconfuse, temporal};
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
    frames: Arc<Frames>, unlab: Arc<Unlabeled>, ub: usize, workers: usize, seed: u64,
) -> Receiver<UnlabBatch> {
    let (tx, rx) = sync_channel::<UnlabBatch>(workers * 2);
    for wi in 0..workers.max(1) {
        let (frames, unlab, tx) = (frames.clone(), unlab.clone(), tx.clone());
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
                    let pair_ok = if rec >= 1 { 1.0 } else { 0.0 };
                    let mut c = frames.stack_f32(if rec >= 1 { rec - 1 } else { rec }, side);
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

fn eval_gaze<BI: Backend>(model: &MicroChad<BI>, frames: &Frames, samples: &[Sample], val: &[usize], device: &BI::Device) -> (f32, [f32; 2]) {
    let sel: Vec<usize> = val.iter().cloned().filter(|&i| samples[i].gaze_usable).collect();
    let sel = if sel.is_empty() { val.to_vec() } else { sel };
    let (mut preds, mut tgts) = (Vec::new(), Vec::new());
    for chunk in sel.chunks(256) {
        let (xs, ys) = build_batch(frames, samples, chunk, false);
        let x = Tensor::<BI, 4>::from_data(TensorData::new(xs, [chunk.len(), T, S, S]), device);
        preds.extend(model.forward(x).into_data().to_vec::<f32>().unwrap());
        tgts.extend(ys);
    }
    let mae = preds.iter().zip(&tgts).map(|(a, b)| (a - b).abs()).sum::<f32>() / preds.len() as f32 * 90.0;
    let mut r = [0.0; 2];
    for c in 0..2 {
        let pc: Vec<f32> = preds.iter().skip(c).step_by(2).cloned().collect();
        let tc: Vec<f32> = tgts.iter().skip(c).step_by(2).cloned().collect();
        r[c] = pearson(&pc, &tc);
    }
    (mae, r)
}

pub fn fit_expr_fast<B: AutodiffBackend>(
    frames: Arc<Frames>, samples: Arc<Vec<Sample>>, train: &[usize], val: &[usize], cfg: &FastCfg,
    device: &B::Device, ckpt: &str,
) -> f32 {
    let w = Arc::new(Weighted::expr(&samples, train, 4.0));
    let pools = Arc::new(Pools::expr_build(&samples, train));
    let rx = spawn_prefetch(frames.clone(), samples.clone(), w, cfg.batch, cfg.workers, true, 0xE47, pools);
    let mut model = Mnv4::<B>::new(T, 4, device);
    load_pretrained_expr(&mut model);
    let mut opt = AdamWConfig::new().with_weight_decay(cfg.wd).init();
    let mut teacher = model.valid(); // EMA teacher (what we deploy/export), matches BabbleTrainer
    let ema_every = 16usize;
    let keep = 0.997f64.powi(ema_every as i32) as f32; // EMA teacher; ~333-step horizon (converges well
                                                       // inside the single-stage budget while smoothing)
    let mut best = f32::INFINITY;
    let mut win = Instant::now();
    // BB_PROF=1: measure recv-wait = time the main thread blocks on the prefetch channel. Large => the
    // aug workers can't keep up (aug/CPU-feed bound). ~0 => workers keep up, so any GPU under-use is the
    // single-threaded op dispatch, not the aug (read GPU power externally to confirm).
    let prof = std::env::var("BB_PROF").is_ok();
    let mut recv_acc = 0f32;
    for step in 0..cfg.steps {
        let r0 = Instant::now();
        let (xs, ys) = rx.recv().unwrap();
        if prof {
            recv_acc += r0.elapsed().as_secs_f32();
        }
        let x = Tensor::<B, 4>::from_data(TensorData::new(xs, [cfg.batch, T, S, S]), device);
        let y = Tensor::<B, 2>::from_data(TensorData::new(ys, [cfg.batch, 4]), device);
        let loss = supervised_deconfuse(model.forward(x), y, device);
        let gp = GradientsParams::from_grads(loss.backward(), &model);
        model = opt.step(cos_lr(step, cfg.steps, cfg.warmup, cfg.lr), model, gp);
        if step % ema_every == 0 {
            teacher = ema_update::<B::InnerBackend, _>(teacher, &model.valid(), keep);
        }
        if step % 100 == 99 {
            let dt = win.elapsed().as_secs_f32();
            if prof {
                println!("[expr {step:5}] {:.1} ms/step | recv-wait(aug)={:.1} ms/step",
                         dt * 10.0, recv_acc * 10.0);
                recv_acc = 0.0;
            } else {
                println!("[expr {step:5}] {:.1} steps/s ({:.1} ms/step) windowed", 100.0 / dt, dt * 10.0);
            }
            win = Instant::now();
        }
        if step % cfg.val_every == 0 || step == cfg.steps - 1 {
            teacher = ema_update::<B::InnerBackend, _>(teacher, &model.valid(), keep); // fold latest steps
            let (mse, r) = eval_expr(&teacher, &frames, &samples, val, device);
            let tag = if mse < best { "*" } else { "" };
            println!("[expr {step:5}/{}] mse={mse:.5} r({:.3} {:.3} {:.3} {:.3}) {tag}", cfg.steps, r[0], r[1], r[2], r[3]);
            if mse < best {
                best = mse;
                teacher.clone().save_file(ckpt, &NamedMpkFileRecorder::<FullPrecisionSettings>::new()).unwrap();
            }
        }
    }
    best
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
    let pbatch = (cfg.batch / 2).max(1);
    let prx = do_paired.then(|| {
        spawn_paired_prefetch(frames.clone(), paired.clone(), pbatch, cfg.workers, 0x9A17)
    });
    let cons_ckpt = ckpt.replace("expr", "cons");

    // Semi-supervised anti-jitter stream (mean-teacher + temporal), stage B only. ub = batch.
    let unlab = Arc::new(Unlabeled::build(&samples, train));
    let do_unlab = !unlab.is_empty();
    let ub = cfg.batch;
    let urx = do_unlab.then(|| spawn_unlabeled_prefetch(frames.clone(), unlab.clone(), ub, cfg.workers, 0x51A9));
    println!("[expr] paired={} ({} recs) unlabeled={} ({} frames)", do_paired, paired.recs.len(), do_unlab, unlab.items.len());

    let mut model = Mnv4::<B>::new(T, 4, device);
    if load_pretrained_expr(&mut model) {
        model.small_init_head(device); // reference re-inits the head small after loading the backbone
    }
    let head_ids = model.head_ids();
    let mut opt_head = AdamWConfig::new().with_weight_decay(cfg.wd).init();
    let mut opt_bb = AdamWConfig::new().with_weight_decay(cfg.wd).init();
    // Consensus heads (feat_dim = classifier input = 1280); trained in stage B on detached teacher feats.
    let mut cons = ConsensusHeads::<B>::new(1280, device);
    let mut opt_cons = AdamWConfig::new().with_weight_decay(cfg.wd).init();

    let mut teacher = model.valid();
    let mut best = f32::INFINITY;
    let mut win = Instant::now();
    let cos = |p: f64| 0.5 * (1.0 + (std::f64::consts::PI * p).cos());

    // BB_PROF=1: per-phase host-time breakdown (dispatch cost) + an end-of-step sync so GPU tail time
    // shows up as `sync` instead of hiding inside the next step's first blocking call.
    let prof = std::env::var("BB_PROF").is_ok();
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
        pt[0] += t0.elapsed().as_secs_f32();

        let t0 = Instant::now();
        let y = Tensor::<B, 2>::from_data(TensorData::new(ys, [cfg.batch, 4]), device);
        let big = Tensor::<B, 4>::from_data(TensorData::new(sdata, [rows, T, S, S]), device);
        pt[1] += t0.elapsed().as_secs_f32();

        let t0 = Instant::now();
        let out = model.forward(big);
        let mut loss = supervised_deconfuse(out.clone().slice([0..cfg.batch]), y, device);
        pt[2] += t0.elapsed().as_secs_f32();
        let t0 = Instant::now();

        // Evenness (both stages) + consensus (stage B) on the paired eyes.
        let mut cons_step = false;
        if let Some((pld, prd, plld, prld)) = &paired_b {
            let out_pl = out.clone().slice([off_pl..off_pl + pbatch]);
            let out_pr = out.clone().slice([off_pr..off_pr + pbatch]);
            let pll = Tensor::<B, 2>::from_data(TensorData::new(plld.clone(), [pbatch, 4]), device);
            let prl = Tensor::<B, 2>::from_data(TensorData::new(prld.clone(), [pbatch, 4]), device);
            loss = loss + evenness(out_pl, out_pr, pll.clone(), prl.clone(), device);
            if stage_b {
                // detached teacher features (matches deployment, where consensus runs on the exported
                // teacher); autodiff grads flow only into the consensus heads.
                let pl_i = Tensor::<B::InnerBackend, 4>::from_data(TensorData::new(pld.clone(), [pbatch, T, S, S]), device);
                let pr_i = Tensor::<B::InnerBackend, 4>::from_data(TensorData::new(prd.clone(), [pbatch, T, S, S]), device);
                let fl = Tensor::<B, 2>::from_data(teacher.forward_features(pl_i).into_data(), device);
                let fr = Tensor::<B, 2>::from_data(teacher.forward_features(pr_i).into_data(), device);
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
        if let Some((_, cw, _, ok)) = &unlab_b {
            let cs_out = out.clone().slice([off_cs..off_cs + ub]);
            let cw_out = out.clone().slice([off_cw..off_cw + ub]);
            let nw_out = out.clone().slice([off_nw..off_nw + ub]);
            let cw_i = Tensor::<B::InnerBackend, 4>::from_data(TensorData::new(cw.clone(), [ub, T, S, S]), device);
            let t_out = Tensor::<B, 2>::from_data(teacher.classify(teacher.forward_features(cw_i)).into_data(), device);
            let rampup = (b_steps / 4).max(1) as f64;
            let lam = (((step - a_steps) as f64 / rampup).min(1.0).powi(2)) as f32; // lambda_cons = 1.0
            loss = loss + mt_consistency(cs_out, t_out).mul_scalar(lam);
            let ok_t = Tensor::<B, 1>::from_data(TensorData::new(ok.clone(), [ub]), device);
            loss = loss + temporal(cw_out, nw_out, ok_t, device).mul_scalar(0.1); // lambda_temp = 0.1
        }
        pt[3] += t0.elapsed().as_secs_f32();

        let t0 = Instant::now();
        let mut grads = loss.backward();
        pt[4] += t0.elapsed().as_secs_f32();
        let t0 = Instant::now();
        let head_gp = GradientsParams::from_params(&mut grads, &model, &head_ids);
        let bb_gp = GradientsParams::from_module(&mut grads, &model); // remaining = backbone
        model = opt_head.step(lr_head, model, head_gp);
        if !frozen {
            model = opt_bb.step(lr_bb, model, bb_gp);
        }
        if cons_step {
            let cons_gp = GradientsParams::from_module(&mut grads, &cons);
            cons = opt_cons.step(lr_head, cons, cons_gp);
        }
        pt[5] += t0.elapsed().as_secs_f32();

        let t0 = Instant::now();
        if step % ema_every == 0 {
            let keep = decay.powi(ema_every as i32) as f32;
            teacher = ema_update::<B::InnerBackend, _>(teacher, &model.valid(), keep);
        }
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
            let keep = decay.powi(ema_every as i32) as f32;
            teacher = ema_update::<B::InnerBackend, _>(teacher, &model.valid(), keep);
            let (mse, r) = eval_expr(&teacher, &frames, &samples, val, device);
            let tag = if mse < best { "*" } else { "" };
            let st = if stage_b { "B" } else { "A" };
            println!("[expr {st} {step:5}/{}] mse={mse:.5} r({:.3} {:.3} {:.3} {:.3}) {tag}", cfg.steps, r[0], r[1], r[2], r[3]);
            if mse < best {
                best = mse;
                let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
                teacher.clone().save_file(ckpt, &rec).unwrap();
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
    let mut best = f32::INFINITY;
    let mut win = Instant::now();
    // Gaze deploys the STUDENT (BabbleTrainer only uses a gaze EMA teacher under semi-supervised
    // mean-teacher, wired separately).
    let prof = std::env::var("BB_PROF").is_ok();
    let mut pt = [0f32; 5]; // recv h2d fwd bwd+opt sync
    for step in 0..cfg.steps {
        let t0 = Instant::now();
        let (xs, ys) = rx.recv().unwrap();
        pt[0] += t0.elapsed().as_secs_f32();
        let t0 = Instant::now();
        let x = Tensor::<B, 4>::from_data(TensorData::new(xs, [cfg.batch, T, S, S]), device);
        let y = Tensor::<B, 2>::from_data(TensorData::new(ys, [cfg.batch, 2]), device);
        pt[1] += t0.elapsed().as_secs_f32();
        let t0 = Instant::now();
        let loss = (model.forward(x) - y).powf_scalar(2.0).mean();
        pt[2] += t0.elapsed().as_secs_f32();
        let t0 = Instant::now();
        let gp = GradientsParams::from_grads(loss.backward(), &model);
        model = opt.step(cos_lr(step, cfg.steps, cfg.warmup, cfg.lr), model, gp);
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
            let (mae, r) = eval_gaze(&vm, &frames, &samples, val, device);
            let tag = if mae < best { "*" } else { "" };
            println!("[gaze {step:5}/{}] mae={mae:.2}deg r({:.3},{:.3}) {tag}", cfg.steps, r[0], r[1]);
            if mae < best {
                best = mae;
                vm.clone().save_file(ckpt, &NamedMpkFileRecorder::<FullPrecisionSettings>::new()).unwrap();
            }
        }
    }
    best
}
