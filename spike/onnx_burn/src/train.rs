// Expr + gaze training loops: weighted sampling, LR schedule, validation, best-checkpoint.
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;

use crate::data::{temporal_stack, Bin, EyeCache, Sample, S, T};
use crate::eval::{eval_expr, eval_gaze};
use crate::gaze::MicroChad;
use crate::losses::{supervised, supervised_deconfuse};
use crate::mnv4::Mnv4;
use crate::sampler::{gaze_weights, split, Rng, Weighted};

struct XorShift(u64);
impl XorShift {
    fn next_usize(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

pub fn make_batch<B: Backend>(
    bin: &Bin,
    samples: &[Sample],
    idxs: &[usize],
    cache: &EyeCache,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 2>) {
    let b = idxs.len();
    let mut xs = Vec::with_capacity(b * T * S * S);
    let mut ys = Vec::with_capacity(b * 4);
    for &i in idxs {
        let s = &samples[i];
        let stack = temporal_stack(bin, s.rec, s.side, cache);
        xs.extend(stack.iter().map(|&v| v as f32 / 255.0));
        ys.extend_from_slice(&s.label[2..6]); // EXPR_SLICE = channels 2..6
    }
    let x = Tensor::from_data(TensorData::new(xs, [b, T, S, S]), device);
    let y = Tensor::from_data(TensorData::new(ys, [b, 4]), device);
    (x, y)
}

pub fn fit_supervised<B: AutodiffBackend>(
    bin: &Bin,
    samples: &[Sample],
    steps: usize,
    batch: usize,
    lr: f64,
    device: &B::Device,
) -> Mnv4<B> {
    let mut model = Mnv4::<B>::new(T, 4, device);
    let mut opt = AdamWConfig::new().with_weight_decay(1e-4).init();
    let cache = EyeCache::default();
    let mut rng = XorShift(0x2545F4914F6CDD1D);
    let n = samples.len();
    let mut first = 0.0f32;
    for step in 0..steps {
        let idxs: Vec<usize> = (0..batch).map(|_| rng.next_usize(n)).collect();
        let (x, y) = make_batch::<B>(bin, samples, &idxs, &cache, device);
        let loss = supervised(model.forward(x), y);
        let val = loss.clone().into_scalar().elem::<f32>();
        if step == 0 {
            first = val;
        }
        let grads = loss.backward();
        let gp = GradientsParams::from_grads(grads, &model);
        model = opt.step(lr, model, gp);
        if step % 5 == 0 || step == steps - 1 {
            println!("step {step:4} loss {val:.5}");
        }
    }
    println!("first->last loss: {first:.5} -> (see above)");
    model
}

fn cos_lr(step: usize, steps: usize, warmup: usize, lr: f64) -> f64 {
    if step < warmup {
        lr * (step + 1) as f64 / warmup as f64
    } else {
        let p = (step - warmup) as f64 / (steps - warmup).max(1) as f64;
        lr * 0.5 * (1.0 + (std::f64::consts::PI * p).cos())
    }
}

fn save_expr<BI: Backend>(m: &Mnv4<BI>, path: &str) {
    let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    m.clone().save_file(path, &rec).expect("save expr ckpt");
}
fn save_gaze<BI: Backend>(m: &MicroChad<BI>, path: &str) {
    let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    m.clone().save_file(path, &rec).expect("save gaze ckpt");
}

pub struct ExprCfg {
    pub steps: usize,
    pub batch: usize,
    pub warmup: usize,
    pub val_every: usize,
    pub lr: f64,
    pub wd: f32,
}

// Supervised expr training with deconfuse-weighted reg + range, weighted (active-boosted) sampling,
// held-out validation, best-MSE checkpoint. Deploys the student (EMA teacher is a later refinement).
pub fn fit_expr<B: AutodiffBackend>(
    bin: &Bin, samples: &[Sample], cfg: &ExprCfg, device: &B::Device, ckpt: &str,
) -> f32 {
    let (train_idx, val_idx) = split(samples.len(), 0.1, 1234);
    let w = Weighted::expr(samples, &train_idx, 4.0);
    let mut model = Mnv4::<B>::new(T, 4, device);
    let mut opt = AdamWConfig::new().with_weight_decay(cfg.wd).init();
    let cache = EyeCache::default();
    let mut rng = Rng::new(0xE47);
    let mut best = f32::INFINITY;
    for step in 0..cfg.steps {
        let lr = cos_lr(step, cfg.steps, cfg.warmup, cfg.lr);
        let idxs: Vec<usize> = (0..cfg.batch).map(|_| w.sample(&mut rng)).collect();
        let (x, y) = make_batch::<B>(bin, samples, &idxs, &cache, device);
        let loss = supervised_deconfuse(model.forward(x), y, device);
        let grads = loss.backward();
        let gp = GradientsParams::from_grads(grads, &model);
        model = opt.step(lr, model, gp);
        if step % cfg.val_every == 0 || step == cfg.steps - 1 {
            let vm = model.valid();
            let (mse, r) = eval_expr(&vm, bin, samples, &val_idx, &cache, device);
            let tag = if mse < best { "*" } else { "" };
            println!(
                "[expr {step:5}/{}] mse={mse:.5} r(lid={:.3} widen={:.3} squint={:.3} brow={:.3}) {tag}",
                cfg.steps, r[0], r[1], r[2], r[3]
            );
            if mse < best {
                best = mse;
                save_expr(&vm, ckpt);
            }
        }
    }
    best
}

pub struct GazeCfg {
    pub steps: usize,
    pub batch: usize,
    pub warmup: usize,
    pub val_every: usize,
    pub lr: f64,
    pub wd: f32,
}

pub fn fit_gaze<B: AutodiffBackend>(
    bin: &Bin, samples: &[Sample], cfg: &GazeCfg, device: &B::Device, ckpt: &str,
) -> f32 {
    let (train_idx, val_idx) = split(samples.len(), 0.1, 1234);
    let w = gaze_weights(samples, &train_idx);
    let mut model = MicroChad::<B>::new(T, 2, device);
    let mut opt = AdamWConfig::new().with_weight_decay(cfg.wd).init();
    let cache = EyeCache::default();
    let mut rng = Rng::new(0x6A2E);
    let mut best = f32::INFINITY;
    for step in 0..cfg.steps {
        let lr = cos_lr(step, cfg.steps, cfg.warmup, cfg.lr);
        let idxs: Vec<usize> = (0..cfg.batch).map(|_| w.sample(&mut rng)).collect();
        let b = idxs.len();
        let mut xs = Vec::with_capacity(b * T * S * S);
        let mut ys = Vec::with_capacity(b * 2);
        for &i in &idxs {
            let s = &samples[i];
            let st = temporal_stack(bin, s.rec, s.side, &cache);
            xs.extend(st.iter().map(|&v| v as f32 / 255.0));
            ys.extend_from_slice(&s.label[0..2]);
        }
        let x = Tensor::<B, 4>::from_data(TensorData::new(xs, [b, T, S, S]), device);
        let y = Tensor::<B, 2>::from_data(TensorData::new(ys, [b, 2]), device);
        let loss = (model.forward(x) - y).powf_scalar(2.0).mean();
        let grads = loss.backward();
        let gp = GradientsParams::from_grads(grads, &model);
        model = opt.step(lr, model, gp);
        if step % cfg.val_every == 0 || step == cfg.steps - 1 {
            let vm = model.valid();
            let (mse, mae, r) = eval_gaze(&vm, bin, samples, &val_idx, &cache, device);
            let tag = if mse_better(mae, best) { "*" } else { "" };
            println!("[gaze {step:5}/{}] mse={mse:.5} mae={mae:.2}deg r({:.3},{:.3}) {tag}", cfg.steps, r[0], r[1]);
            if mse_better(mae, best) {
                best = mae;
                save_gaze(&vm, ckpt);
            }
        }
    }
    best
}

fn mse_better(v: f32, best: f32) -> bool {
    v < best
}
