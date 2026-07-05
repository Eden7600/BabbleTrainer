// Prefetch-fed training. `probe <bin>` = CPU-only data-feed measurement (no GPU); otherwise full train.
use std::sync::Arc;
use std::time::Instant;

use burn::backend::wgpu::{graphics::AutoGraphicsApi, init_setup, MemoryConfiguration, RuntimeOptions, WgpuDevice};
use burn::backend::{Autodiff, Wgpu};
use cubecl_runtime::memory_management::{MemoryPoolOptions, PoolType};
use onnx_burn::data::{parse_bin, process_eye, S, T};
use onnx_burn::fast::{build_samples, fit_expr_staged, fit_gaze_fast, FastCfg, Frames};
use onnx_burn::sampler::{split_by_record, Rng};

type AB = Autodiff<Wgpu>;

const MB: u64 = 1024 * 1024;

// Capped-pool memory config: the cubecl default (SubSlices) sizes its pages from
// max_storage_buffer_binding_size (GBs) and NEVER deallocates them, so each device server holds
// far more VRAM than its live tensors (measured: gaze holds ~2 GB for ~500 MB of need). Here:
// small sliced pools sized to the actual working set, finite dealloc periods so idle pages are
// returned, and an exclusive catch-all so big one-offs (e.g. the resident frame store) allocate
// exactly-sized pages instead of pinning a giant sliced page.
fn pool_opts(page_mb: u64) -> RuntimeOptions {
    let page = page_mb * MB;
    RuntimeOptions {
        memory_config: MemoryConfiguration::Custom {
            pool_options: vec![
                MemoryPoolOptions { pool_type: PoolType::ExclusivePages { max_alloc_size: 0 }, dealloc_period: None },
                MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages { page_size: page / 4, max_slice_size: page / 16 },
                    dealloc_period: Some(4000),
                },
                MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages { page_size: page, max_slice_size: page },
                    dealloc_period: Some(2000),
                },
                // Exclusive buckets for big allocations (activations, frame store): pages are
                // sized ~to the request (NOT the bucket; a u64::MAX catch-all would try to
                // allocate cur_avg_size = MAX/2 bytes on first use and poison the fusion stream),
                // and idle pages are returned. Top bucket = 4 GB, the wgpu binding limit region.
                MemoryPoolOptions { pool_type: PoolType::ExclusivePages { max_alloc_size: 128 * MB }, dealloc_period: Some(1000) },
                MemoryPoolOptions { pool_type: PoolType::ExclusivePages { max_alloc_size: 256 * MB }, dealloc_period: Some(1000) },
                MemoryPoolOptions { pool_type: PoolType::ExclusivePages { max_alloc_size: 512 * MB }, dealloc_period: Some(500) },
                MemoryPoolOptions { pool_type: PoolType::ExclusivePages { max_alloc_size: 1024 * MB }, dealloc_period: Some(500) },
                MemoryPoolOptions { pool_type: PoolType::ExclusivePages { max_alloc_size: 4096 * MB }, dealloc_period: Some(300) },
            ],
        },
        ..Default::default()
    }
}

// Free VRAM right now, in MB (NVIDIA only; other vendors return None and we use BB_VRAM_MB/default).
fn free_vram_mb() -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).lines().next()?.trim().parse().ok()
}

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

    // ---- VRAM policy ----------------------------------------------------------------------
    // Budget = BB_VRAM_MB if set, else measured free VRAM minus headroom, else a conservative
    // 2500 MB. Ladder thresholds are MEASURED on a 3060 (full 4000/3200 run, pools capped):
    //   FAST  concurrent + resident frame store   peak 5438 MiB   8m58s
    //   MID   concurrent + per-step frame upload  peak 4725 MiB   8m52s
    //   LOW   sequential + per-step               peak 3380 MiB   8m48s  (32 MB pool pages)
    // Wall time is FLAT across tiers here (expr is host-bound; gaze concurrency buys nothing),
    // so the ladder only opts up where spare VRAM makes the resident store / dual device free.
    // Training must coexist with VRChat on the user's GPU — never assume the card is ours.
    // BB_POOL_MB (default 32, 0=off) caps the cubecl pools; BB_SEQ=0|1 / BB_GPU_AUG override.
    let headroom = 700u64;
    let budget_mb: u64 = std::env::var("BB_VRAM_MB").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
        match free_vram_mb() {
            Some(f) => f.saturating_sub(headroom),
            None => 2500,
        }
    });
    let seq = match std::env::var("BB_SEQ").ok().as_deref() {
        Some("1") => true,
        Some("0") => false,
        _ => budget_mb < 4900, // below the measured MID peak: one device, one pool set
    };
    if std::env::var("BB_GPU_AUG").is_err() && std::env::var("BB_VRAM_MB").is_err() {
        // fast.rs auto-tier compares the frame store against BB_VRAM_MB: hand it the budget left
        // after the measured non-store training footprint (~4.7 GB concurrent w/ pools).
        let store_budget = budget_mb.saturating_sub(4700);
        unsafe { std::env::set_var("BB_VRAM_MB", store_budget.to_string()) };
    }
    let pool_mb: u64 = std::env::var("BB_POOL_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
    if pool_mb > 0 {
        init_setup::<AutoGraphicsApi>(&WgpuDevice::DefaultDevice, pool_opts(pool_mb));
        if !seq && std::env::var("BB_SHARED_DEV").ok().as_deref() != Some("1") {
            init_setup::<AutoGraphicsApi>(&WgpuDevice::DiscreteGpu(0), pool_opts(pool_mb));
        }
    }
    println!(
        "[vram] budget {budget_mb} MB -> {} | pools {}",
        if seq { "SEQUENTIAL expr->gaze, shared device" } else { "concurrent expr+gaze" },
        if pool_mb > 0 { format!("capped @ {pool_mb} MB pages") } else { "default".into() },
    );

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
    let gaze_cfg = FastCfg { steps: gs, batch: 64, warmup: gs / 10, val_every: (gs / 4).max(1), lr: 1e-3, wd: 1e-2, workers: 2 };
    let gaze_cfg_t = gaze_cfg.clone();
    let gaze_h = (only != "expr" && !seq).then(|| {
        let (frames_g, samples_g, train_g, val_g) = (frames.clone(), samples.clone(), train.clone(), val.clone());
        std::thread::spawn(move || {
            // Own logical wgpu device (distinct cubecl server + queue on the same GPU): expr and
            // gaze encode/submit in parallel instead of serializing on one device server.
            // BB_SHARED_DEV=1 reverts to the shared default device.
            let device = if std::env::var("BB_SHARED_DEV").ok().as_deref() == Some("1") {
                Default::default()
            } else {
                burn::backend::wgpu::WgpuDevice::DiscreteGpu(0)
            };
            fit_gaze_fast::<AB>(frames_g, samples_g, &train_g, &val_g, &gaze_cfg_t, &device, "../ckpt_gaze.mpk")
        })
    });
    let e = if only != "gaze" {
        fit_expr_staged::<AB>(
            frames.clone(), samples.clone(), &train, &val,
            &FastCfg { steps: es, batch: 32, warmup: es / 10, val_every: (es / 4).max(1), lr: 5e-4, wd: 1e-4, workers },
            &device, "../ckpt_expr.mpk",
        )
    } else {
        f32::INFINITY
    };
    // Sequential (low-VRAM) mode: gaze runs AFTER expr on the SAME device — one cubecl server, one
    // pool set, and expr's transient pages have been returned by the time gaze allocates.
    let g = if seq && only != "expr" {
        fit_gaze_fast::<AB>(frames, samples, &train, &val, &gaze_cfg, &device, "../ckpt_gaze.mpk")
    } else {
        gaze_h.map(|h| h.join().unwrap()).unwrap_or(f32::INFINITY)
    };
    println!("\nBEST expr mse={e:.5} | BEST gaze mae={g:.2}deg");
}
