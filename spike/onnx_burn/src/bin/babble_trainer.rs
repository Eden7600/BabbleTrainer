// Baballonia TrainerService entrypoint (train_frontend.py parity): BabbleTrainer "<bin>" "<out.onnx>".
// Warm-start priors resolve next to the exe, LOW-VRAM tier always, progress on the unified
// expr+gaze step axis in the TrainerService stdout protocol, native ONNX written at the end.
// Scratch (checkpoints) goes to a temp dir so the install directory is never written.
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use burn::backend::wgpu::{graphics::AutoGraphicsApi, init_setup, WgpuDevice};
use burn::backend::{Autodiff, NdArray, Wgpu};
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use onnx_burn::data::parse_bin;
use onnx_burn::fast::{build_samples, fit_expr_staged, fit_gaze_fast, pool_opts, FastCfg, Frames};
use onnx_burn::gaze::MicroChad;
use onnx_burn::heads::ConsensusHeads;
use onnx_burn::mnv4::Mnv4;
use onnx_burn::onnx_export::export_merged;
use onnx_burn::sampler::split_by_record;

type AB = Autodiff<Wgpu>;

fn env_default(k: &str, v: &str) {
    if std::env::var(k).is_err() {
        unsafe { std::env::set_var(k, v) };
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut pos = Vec::new();
    let (mut es, mut gs) = (4000usize, 3200usize);
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => { es = args[i + 1].parse().expect("--steps"); i += 1; }
            "--gaze-steps" => { gs = args[i + 1].parse().expect("--gaze-steps"); i += 1; }
            a => pos.push(a.to_string()),
        }
        i += 1;
    }
    let [bin_path, out_onnx] = &pos[..] else {
        eprintln!("usage: BabbleTrainer <user_cal.bin> <out.onnx> [--steps N] [--gaze-steps N]");
        std::process::exit(2);
    };

    // Absolutize both paths, then chdir next to the exe so the shipped priors
    // (expr/gaze/mnv4_weights.safetensors) resolve as plain filenames whatever the caller's cwd.
    let cwd = std::env::current_dir().expect("cwd");
    let abs = |p: &str| if Path::new(p).is_absolute() { PathBuf::from(p) } else { cwd.join(p) };
    let bin_path = abs(bin_path);
    let out_onnx = abs(out_onnx);
    let exe_dir = std::env::current_exe().expect("exe path").parent().expect("exe dir").to_path_buf();
    std::env::set_current_dir(&exe_dir).expect("chdir exe dir");

    let scratch = std::env::temp_dir().join(format!("babbleburn-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).expect("scratch dir");
    let ckpt = |name: &str| scratch.join(name).to_string_lossy().into_owned();

    // Deployment defaults: warm-start both nets (the Python trainer always warm-trains) and the
    // LOW tier — sequential expr->gaze on one device, per-step gpu-aug, 32 MB pool pages.
    env_default("BB_EXPR_WARM", "1");
    env_default("BB_GAZE_WARM", "1");
    env_default("BB_VRAM_MB", "0");
    env_default("BB_LOG", "frontend");
    unsafe { std::env::set_var("BB_PROG_TOTAL", (es + gs).to_string()) };
    unsafe { std::env::set_var("BB_PROG_OFFSET", "0") };

    let pool_mb: u64 = std::env::var("BB_POOL_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    if pool_mb > 0 {
        init_setup::<AutoGraphicsApi>(&WgpuDevice::DefaultDevice, pool_opts(pool_mb));
    }
    let device = Default::default();

    let t0 = Instant::now();
    let bin = parse_bin(bin_path.to_str().expect("bin path")).expect("parse bin");
    let frames = Arc::new(Frames::decode(&bin));
    let samples = Arc::new(build_samples(&bin, &frames));
    println!("{} eye-samples", samples.len());
    let (train, val) = split_by_record(&samples, 0.1, 1234);

    fit_expr_staged::<AB>(
        frames.clone(), samples.clone(), &train, &val,
        &FastCfg { steps: es, batch: 32, warmup: es / 10, val_every: (es / 4).max(1), lr: 5e-4, wd: 1e-4, workers: 6 },
        &device, &ckpt("ckpt_expr.mpk"),
    );
    unsafe { std::env::set_var("BB_PROG_OFFSET", es.to_string()) };
    let gaze_best = fit_gaze_fast::<AB>(
        frames, samples, &train, &val,
        &FastCfg { steps: gs, batch: 64, warmup: gs / 10, val_every: (gs / 4).max(1), lr: 1e-3, wd: 1e-2, workers: 2 },
        &device, &ckpt("ckpt_gaze.mpk"),
    );
    println!("\nEpoch 1/1 completed in {:.2}s. Average loss: {:.6}\n", t0.elapsed().as_secs_f64(), gaze_best);

    type B = NdArray<f32>;
    let dev = Default::default();
    let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let expr = Mnv4::<B>::new(4, 4, &dev).load_file(ckpt("ckpt_expr.mpk"), &rec, &dev).expect("load expr ckpt");
    let gaze = MicroChad::<B>::new(4, 2, &dev).load_file(ckpt("ckpt_gaze.mpk"), &rec, &dev).expect("load gaze ckpt");
    let cons = scratch.join("ckpt_cons.mpk").exists().then(|| {
        ConsensusHeads::<B>::new(1280, &dev).load_file(ckpt("ckpt_cons.mpk"), &rec, &dev).expect("load cons ckpt")
    });
    export_merged(&expr, &gaze, cons.as_ref(), out_onnx.to_str().expect("onnx path")).expect("write onnx");
    let _ = std::fs::remove_dir_all(&scratch);

    println!("Model exported to ONNX: {}", out_onnx.display());
    println!("\nTraining completed successfully!\n");
}
