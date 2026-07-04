// Prefetch-fed training. `probe <bin>` = CPU-only data-feed measurement (no GPU); otherwise full train.
use std::sync::Arc;
use std::time::Instant;

use burn::backend::{Autodiff, Wgpu};
use onnx_burn::data::{parse_bin, process_eye, S, T};
use onnx_burn::fast::{build_samples, fit_expr_staged, fit_gaze_fast, FastCfg, Frames};
use onnx_burn::sampler::{split_by_record, Rng};

type AB = Autodiff<Wgpu>;

fn probe(bin_path: &str) {
    let bin = parse_bin(bin_path).expect("parse bin");
    let n = bin.frames.len();

    let t = Instant::now();
    let cnt = 500.min(n);
    for k in 0..cnt {
        std::hint::black_box(process_eye(&bin, k, 0));
    }
    let per = t.elapsed().as_secs_f64() * 1000.0 / cnt as f64;
    let total1 = per * (n * 2) as f64 / 1000.0;
    println!("decode: {per:.2} ms/eye (1 thread) -> ~{total1:.0}s for {} eyes single-thread, ~{:.0}s across 8 cores",
             n * 2, total1 / 8.0);

    let recs = 1024.min(n);
    let mut local: Vec<Option<Vec<u8>>> = Vec::with_capacity(recs * 2);
    for r in 0..recs {
        local.push(process_eye(&bin, r, 0));
        local.push(process_eye(&bin, r, 1));
    }
    let batch = 32;
    let iters = 300;
    let mut rng = Rng::new(1);
    let t = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..iters {
        let mut xs: Vec<f32> = Vec::with_capacity(batch * T * S * S);
        for _ in 0..batch {
            let rec = rng.usize(recs);
            let side = rng.usize(2);
            for k in 0..T {
                match if rec >= k { local[(rec - k) * 2 + side].as_deref() } else { None } {
                    Some(px) => xs.extend(px.iter().map(|&v| v as f32 / 255.0)),
                    None => xs.extend(std::iter::repeat(0.0).take(S * S)),
                }
            }
        }
        sink += xs[rng.usize(xs.len())];
    }
    let per_batch = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("batch-assembly (decoded->f32 stack, batch {batch}): {per_batch:.2} ms/batch (1 thread) [sink {sink:.1}]");
    println!("=> with N prefetch workers, per-step CPU feed ~ {:.2} ms; GPU-bound if step time >> this", per_batch);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("probe") {
        probe(args.get(2).map(|s| s.as_str()).unwrap_or("../../../ModelData/user_cal.bin"));
        return;
    }
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
    let (train, val) = split_by_record(&samples, 0.1, 1234);

    // Gaze runs CONCURRENTLY with expr: expr is host-dispatch-bound, gaze is GPU-kernel-bound, so the
    // two streams interleave on one device and gaze's wall hides under expr's.
    // BB_ONLY=expr|gaze isolates one stream (profiling: device syncs must not cross-pollute).
    let only = std::env::var("BB_ONLY").unwrap_or_default();
    let gaze_h = (only != "expr").then(|| {
        let (frames_g, samples_g, train_g, val_g) = (frames.clone(), samples.clone(), train.clone(), val.clone());
        std::thread::spawn(move || {
            let device = Default::default();
            fit_gaze_fast::<AB>(
                frames_g, samples_g, &train_g, &val_g,
                &FastCfg { steps: gs, batch: 64, warmup: gs / 10, val_every: (gs / 4).max(1), lr: 1e-3, wd: 1e-2, workers: 2 },
                &device, "../ckpt_gaze.mpk",
            )
        })
    });
    let e = if only != "gaze" {
        fit_expr_staged::<AB>(
            frames, samples, &train, &val,
            &FastCfg { steps: es, batch: 32, warmup: es / 10, val_every: (es / 4).max(1), lr: 5e-4, wd: 1e-4, workers },
            &device, "../ckpt_expr.mpk",
        )
    } else {
        f32::INFINITY
    };
    let g = gaze_h.map(|h| h.join().unwrap()).unwrap_or(f32::INFINITY);
    println!("\nBEST expr mse={e:.5} | BEST gaze mae={g:.2}deg");
}
