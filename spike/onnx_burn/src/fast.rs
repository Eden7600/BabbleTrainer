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
use rayon::prelude::*;

use crate::data::{process_eye, Bin, Sample, S, T};
use crate::gaze::MicroChad;
use crate::losses::supervised_deconfuse;
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
}

fn raw_gaze(bin: &Bin, rec: usize, side: usize) -> (f32, f32) {
    let f = &bin.frames[rec].fields;
    if side == 0 { (f[5], f[6]) } else { (f[7], f[8]) }
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

fn spawn_prefetch(
    frames: Arc<Frames>, samples: Arc<Vec<Sample>>, w: Arc<Weighted>, batch: usize, workers: usize,
    expr: bool, seed: u64,
) -> Receiver<Batch> {
    let (tx, rx) = sync_channel::<Batch>(workers * 2);
    for wi in 0..workers.max(1) {
        let (frames, samples, w, tx) = (frames.clone(), samples.clone(), w.clone(), tx.clone());
        thread::spawn(move || {
            let mut rng = Rng::new(seed.wrapping_add((wi as u64).wrapping_mul(2654435761)));
            loop {
                let idxs: Vec<usize> = (0..batch).map(|_| w.sample(&mut rng)).collect();
                if tx.send(build_batch(&frames, &samples, &idxs, expr)).is_err() {
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
    let rx = spawn_prefetch(frames.clone(), samples.clone(), w, cfg.batch, cfg.workers, true, 0xE47);
    let mut model = Mnv4::<B>::new(T, 4, device);
    let mut opt = AdamWConfig::new().with_weight_decay(cfg.wd).init();
    let mut best = f32::INFINITY;
    let mut win = Instant::now();
    for step in 0..cfg.steps {
        let (xs, ys) = rx.recv().unwrap();
        let x = Tensor::<B, 4>::from_data(TensorData::new(xs, [cfg.batch, T, S, S]), device);
        let y = Tensor::<B, 2>::from_data(TensorData::new(ys, [cfg.batch, 4]), device);
        let loss = supervised_deconfuse(model.forward(x), y, device);
        let gp = GradientsParams::from_grads(loss.backward(), &model);
        model = opt.step(cos_lr(step, cfg.steps, cfg.warmup, cfg.lr), model, gp);
        if step % 100 == 99 {
            let dt = win.elapsed().as_secs_f32();
            println!("[expr {step:5}] {:.1} steps/s ({:.1} ms/step) windowed", 100.0 / dt, dt * 10.0);
            win = Instant::now();
        }
        if step % cfg.val_every == 0 || step == cfg.steps - 1 {
            let vm = model.valid();
            let (mse, r) = eval_expr(&vm, &frames, &samples, val, device);
            let tag = if mse < best { "*" } else { "" };
            println!("[expr {step:5}/{}] mse={mse:.5} r({:.3} {:.3} {:.3} {:.3}) {tag}", cfg.steps, r[0], r[1], r[2], r[3]);
            if mse < best {
                best = mse;
                vm.clone().save_file(ckpt, &NamedMpkFileRecorder::<FullPrecisionSettings>::new()).unwrap();
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
    let rx = spawn_prefetch(frames.clone(), samples.clone(), w, cfg.batch, cfg.workers, false, 0x6A2E);
    let mut model = MicroChad::<B>::new(T, 2, device);
    let mut opt = AdamWConfig::new().with_weight_decay(cfg.wd).init();
    let mut best = f32::INFINITY;
    let mut win = Instant::now();
    for step in 0..cfg.steps {
        let (xs, ys) = rx.recv().unwrap();
        let x = Tensor::<B, 4>::from_data(TensorData::new(xs, [cfg.batch, T, S, S]), device);
        let y = Tensor::<B, 2>::from_data(TensorData::new(ys, [cfg.batch, 2]), device);
        let loss = (model.forward(x) - y).powf_scalar(2.0).mean();
        let gp = GradientsParams::from_grads(loss.backward(), &model);
        model = opt.step(cos_lr(step, cfg.steps, cfg.warmup, cfg.lr), model, gp);
        if step % 100 == 99 {
            let dt = win.elapsed().as_secs_f32();
            println!("[gaze {step:5}] {:.1} steps/s ({:.1} ms/step) windowed", 100.0 / dt, dt * 10.0);
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
