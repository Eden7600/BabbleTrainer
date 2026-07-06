use burn::prelude::*;
use burn::tensor::backend::Backend;
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

use crate::data::{Sample, S, T};
use crate::gaze::MicroChad;
use crate::heads::ConsensusHeads;
use crate::mnv4::Mnv4;

use super::frames::Frames;
use super::index::Paired;

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

pub(crate) fn pearson(p: &[f32], t: &[f32]) -> f32 {
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

pub(crate) fn cos_lr(step: usize, steps: usize, warmup: usize, lr: f64) -> f64 {
    if step < warmup {
        lr * (step + 1) as f64 / warmup as f64
    } else {
        let p = (step - warmup) as f64 / (steps - warmup).max(1) as f64;
        lr * 0.5 * (1.0 + (std::f64::consts::PI * p).cos())
    }
}

// Pretrained-backbone init (BabbleTrainer build_expr_net(pretrained=True)): load the timm
// MNv4-conv-small-050 weights (4-ch stem + 4-out head already adapted) so the per-user calibration
// starts from ImageNet features instead of random. mnv4_parity.rs proves this load matches timm
// block-by-block. Returns true if the weights were found and loaded.
pub(crate) fn load_pretrained_expr<B: Backend>(model: &mut Mnv4<B>) -> bool {
    // BB_EXPR_WARM warm-starts from the trained expr model (expr_weights.safetensors, parity with the
    // deployed Python model_best.pt student, incl. the classifier head) instead of just the ImageNet
    // MNv4 backbone. Falls back to the backbone if the trained-expr file is absent.
    let mut paths: Vec<&str> = Vec::new();
    if std::env::var("BB_EXPR_WARM").is_ok() {
        paths.push("../expr_weights.safetensors");
        paths.push("expr_weights.safetensors");
    }
    paths.push("../mnv4_weights.safetensors");
    paths.push("mnv4_weights.safetensors");
    for path in paths {
        if std::path::Path::new(path).exists() {
            let mut store = SafetensorsStore::from_file(path)
                .with_from_adapter(PyTorchToBurnAdapter)
                .with_key_remapping(r"(bn1|bn|norm_head)\.weight$", "${1}.gamma")
                .with_key_remapping(r"(bn1|bn|norm_head)\.bias$", "${1}.beta")
                .allow_partial(true)
                .validate(false);
            match model.load_from(&mut store) {
                Ok(res) => {
                    println!("[expr] init from {path}: {res}");
                    return true;
                }
                Err(e) => println!("[expr] load {path} failed ({e}); trying next"),
            }
        }
    }
    println!("[expr] no pretrained weights found; random init");
    false
}

// Optional gaze warm-start (parity with the deployed Python recipe: load a 12k-step MicroChad prior
// then fine-tune per user). Off by default — Burn trains gaze from scratch. MicroChad field names
// (conv1..conv6, fc_gaze) match the torch checkpoint keys 1:1; no BatchNorm so no gamma/beta remap.
pub(crate) fn load_pretrained_gaze<B: Backend>(model: &mut MicroChad<B>) -> bool {
    for path in ["../gaze_weights.safetensors", "gaze_weights.safetensors"] {
        if std::path::Path::new(path).exists() {
            let mut store = SafetensorsStore::from_file(path)
                .with_from_adapter(PyTorchToBurnAdapter)
                .allow_partial(true)
                .validate(false);
            match model.load_from(&mut store) {
                Ok(res) => {
                    println!("[gaze] warm-start from {path}: {res}");
                    return true;
                }
                Err(e) => {
                    println!("[gaze] warm-start load failed ({e}); cold start");
                    return false;
                }
            }
        }
    }
    println!("[gaze] no gaze prior found (gaze_weights.safetensors); cold start");
    false
}

pub(crate) fn eval_expr<BI: Backend>(model: &Mnv4<BI>, frames: &Frames, samples: &[Sample], val: &[usize], device: &BI::Device) -> (f32, [f32; 4]) {
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
pub(crate) fn eval_gaze<BI: Backend>(model: &MicroChad<BI>, frames: &Frames, samples: &[Sample], val: &[usize], device: &BI::Device) -> (f32, f32, [f32; 2]) {
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
pub(crate) fn eval_consensus<BI: Backend>(
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
