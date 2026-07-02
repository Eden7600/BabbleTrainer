// Full expr + gaze training on wgpu with validation + best-checkpoint. Args: [expr_steps] [gaze_steps].
use burn::backend::{Autodiff, Wgpu};
use onnx_burn::data::{build_samples, parse_bin, EyeCache};
use onnx_burn::train::{fit_expr, fit_gaze, ExprCfg, GazeCfg};

type AB = Autodiff<Wgpu>;

fn main() {
    let device = Default::default();
    let bin_path = std::env::args().nth(1).unwrap_or_else(|| "../../../ModelData/user_cal.bin".into());
    let es: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(4000);
    let gs: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(3200);

    println!("loading + indexing {bin_path} ...");
    let bin = parse_bin(&bin_path).expect("parse bin");
    let cache = EyeCache::default();
    let samples = build_samples(&bin, &cache);
    println!("{} eye-samples", samples.len());

    let e = fit_expr::<AB>(
        &bin, &samples,
        &ExprCfg { steps: es, batch: 32, warmup: es / 10, val_every: (es / 4).max(1), lr: 5e-4, wd: 1e-4 },
        &device, "../ckpt_expr.mpk",
    );
    let g = fit_gaze::<AB>(
        &bin, &samples,
        &GazeCfg { steps: gs, batch: 64, warmup: gs / 10, val_every: (gs / 4).max(1), lr: 1e-3, wd: 1e-2 },
        &device, "../ckpt_gaze.mpk",
    );
    println!("\nBEST expr mse={e:.5}  |  BEST gaze mae={g:.2} deg");
    println!("checkpoints: ../ckpt_expr.mpk  ../ckpt_gaze.mpk");
}
