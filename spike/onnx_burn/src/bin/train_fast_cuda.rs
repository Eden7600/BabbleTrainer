// train_fast on the burn-cuda backend — full-pipeline comparison vs wgpu (same regime, same code).
use std::sync::Arc;
use std::time::Instant;

use burn::backend::{Autodiff, Cuda};
use onnx_burn::data::parse_bin;
use onnx_burn::fast::{build_samples, fit_expr_staged, fit_gaze_fast, FastCfg, Frames};
use onnx_burn::sampler::split;

type AB = Autodiff<Cuda>;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let bin_path = args.get(1).cloned().unwrap_or_else(|| "../../../ModelData/user_cal.bin".into());
    let es: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4000);
    let gs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3200);
    let workers: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6);

    let device = Default::default();
    println!("parsing {bin_path} ...");
    let bin = parse_bin(&bin_path).expect("parse bin");
    let t = Instant::now();
    let frames = Arc::new(Frames::decode(&bin));
    println!("parallel-decoded {} eyes in {:.1}s", bin.frames.len() * 2, t.elapsed().as_secs_f32());
    let samples = Arc::new(build_samples(&bin, &frames));
    println!("{} eye-samples", samples.len());
    let (train, val) = split(samples.len(), 0.1, 1234);

    let (frames_g, samples_g, train_g, val_g) = (frames.clone(), samples.clone(), train.clone(), val.clone());
    let gaze_h = std::thread::spawn(move || {
        let device = Default::default();
        fit_gaze_fast::<AB>(
            frames_g, samples_g, &train_g, &val_g,
            &FastCfg { steps: gs, batch: 64, warmup: gs / 10, val_every: (gs / 4).max(1), lr: 1e-3, wd: 1e-2, workers: 2 },
            &device, "../ckpt_gaze_cuda.mpk",
        )
    });
    let e = fit_expr_staged::<AB>(
        frames, samples, &train, &val,
        &FastCfg { steps: es, batch: 32, warmup: es / 10, val_every: (es / 4).max(1), lr: 5e-4, wd: 1e-4, workers },
        &device, "../ckpt_expr_cuda.mpk",
    );
    let g = gaze_h.join().unwrap();
    println!("\nBEST expr mse={e:.5} | BEST gaze mae={g:.2}deg");
}
