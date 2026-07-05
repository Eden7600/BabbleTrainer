// Isolated gpu_aug::augment bench: R rows, realistic spec mix, synced per iter.
use std::time::Instant;

use burn::backend::{wgpu::WgpuDevice, Wgpu};
use burn::prelude::*;
use onnx_burn::data::{S, T};
use onnx_burn::gpu_aug::{AugBackend, AugSpec};
use onnx_burn::sampler::Rng;

type B = Wgpu;

fn main() {
    let device = WgpuDevice::default();
    let mut rng = Rng::new(7);
    let n = 44740usize;
    let data: Vec<Option<Vec<u8>>> = (0..n)
        .map(|k| if k % 50 == 3 { None } else { Some(vec![(k % 251) as u8; S * S]) })
        .collect();
    let t0 = Instant::now();
    let frames = <B as AugBackend>::aug_upload(&data, &device);
    println!("upload: {:.1}s", t0.elapsed().as_secs_f64());

    let ids = |k: usize| -> [u32; T] { [(k + 1) as u32, k as u32, (k.max(1) - 1) as u32, (k.max(2) - 2) as u32] };
    for r in [64usize, 160] {
        let specs: Vec<AugSpec> = (0..r)
            .map(|i| match i % 3 {
                0 => AugSpec::strong(ids(i * 40 + 8), &mut rng),
                1 => AugSpec::mixup(ids(i * 40 + 8), ids(i * 31 + 9), 0.4, &mut rng),
                _ => {
                    let g = onnx_burn::aug::sample_geom_weak(&mut rng);
                    AugSpec::weak(ids(i * 40 + 8), &g, &mut rng)
                }
            })
            .collect();
        for _ in 0..5 {
            let _ = <B as AugBackend>::aug_batch(&frames, &specs, &device).sum().into_scalar();
        }
        let iters = 30;
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = <B as AugBackend>::aug_batch(&frames, &specs, &device).sum().into_scalar();
        }
        println!("augment R={r}: {:.2} ms/iter (host+gpu, synced)", t0.elapsed().as_secs_f64() * 1000.0 / iters as f64);
    }
}
