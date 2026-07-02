// Held-out metrics: expr per-channel MSE + Pearson r; gaze MSE + MAE(deg). Inner (non-autodiff) backend.
use burn::prelude::*;

use crate::data::{Bin, EyeCache, Sample};
use crate::gaze::MicroChad;
use crate::mnv4::Mnv4;
use crate::train::make_batch;

const GAZE_RANGE_DEG: f32 = 45.0;

fn pearson(p: &[f32], t: &[f32]) -> f32 {
    let n = p.len() as f32;
    let (mp, mt) = (p.iter().sum::<f32>() / n, t.iter().sum::<f32>() / n);
    let mut cov = 0.0;
    let mut vp = 0.0;
    let mut vt = 0.0;
    for i in 0..p.len() {
        let (dp, dt) = (p[i] - mp, t[i] - mt);
        cov += dp * dt;
        vp += dp * dp;
        vt += dt * dt;
    }
    if vp.sqrt() < 1e-6 || vt.sqrt() < 1e-6 {
        0.0
    } else {
        cov / (vp.sqrt() * vt.sqrt())
    }
}

// returns (expr_mse, [r; 4]) over the 4 expr channels (label cols 2..6).
pub fn eval_expr<BI: Backend>(
    model: &Mnv4<BI>, bin: &Bin, samples: &[Sample], val: &[usize], cache: &EyeCache, device: &BI::Device,
) -> (f32, [f32; 4]) {
    let mut preds: Vec<f32> = Vec::new();
    let mut tgts: Vec<f32> = Vec::new();
    for chunk in val.chunks(256) {
        let (x, y) = make_batch::<BI>(bin, samples, chunk, cache, device);
        let out = model.forward(x).clamp(0.0, 1.0);
        preds.extend(out.into_data().to_vec::<f32>().unwrap());
        tgts.extend(y.into_data().to_vec::<f32>().unwrap());
    }
    let n = preds.len();
    let mse = preds.iter().zip(&tgts).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / n as f32;
    let mut r = [0.0f32; 4];
    for c in 0..4 {
        let pc: Vec<f32> = preds.iter().skip(c).step_by(4).cloned().collect();
        let tc: Vec<f32> = tgts.iter().skip(c).step_by(4).cloned().collect();
        r[c] = pearson(&pc, &tc);
    }
    (mse, r)
}

// gaze eval over gaze-valid val samples: (mse, mae_deg, [r;2]). Uses label cols 0..2.
pub fn eval_gaze<BI: Backend>(
    model: &MicroChad<BI>, bin: &Bin, samples: &[Sample], val: &[usize], cache: &EyeCache, device: &BI::Device,
) -> (f32, f32, [f32; 2]) {
    use crate::data::{temporal_stack, S, T};
    let sel: Vec<usize> = val.iter().cloned().filter(|&i| samples[i].gaze_valid > 0.5).collect();
    let sel = if sel.is_empty() { val.to_vec() } else { sel };
    let mut preds: Vec<f32> = Vec::new();
    let mut tgts: Vec<f32> = Vec::new();
    for chunk in sel.chunks(256) {
        let b = chunk.len();
        let mut xs = Vec::with_capacity(b * T * S * S);
        for &i in chunk {
            let s = &samples[i];
            let st = temporal_stack(bin, s.rec, s.side, cache);
            xs.extend(st.iter().map(|&v| v as f32 / 255.0));
            tgts.extend_from_slice(&s.label[0..2]);
        }
        let x = Tensor::<BI, 4>::from_data(TensorData::new(xs, [b, T, S, S]), device);
        preds.extend(model.forward(x).into_data().to_vec::<f32>().unwrap());
    }
    let n = preds.len();
    let mse = preds.iter().zip(&tgts).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / n as f32;
    let mae_deg = preds.iter().zip(&tgts).map(|(a, b)| (a - b).abs()).sum::<f32>() / n as f32
        * 2.0
        * GAZE_RANGE_DEG;
    let mut r = [0.0f32; 2];
    for c in 0..2 {
        let pc: Vec<f32> = preds.iter().skip(c).step_by(2).cloned().collect();
        let tc: Vec<f32> = tgts.iter().skip(c).step_by(2).cloned().collect();
        r[c] = pearson(&pc, &tc);
    }
    (mse, mae_deg, r)
}
