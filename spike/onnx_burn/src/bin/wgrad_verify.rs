// Verify dense_small_wgrad vs ndarray reference + bench vs fallback, across conv shapes.
// Run twice: CUBECL_WGRAD=dense and =fallback; correctness always checked vs CPU.
use std::time::Instant;

use burn::backend::{ndarray::NdArrayDevice, wgpu::WgpuDevice, Autodiff, NdArray, Wgpu};
use burn::nn::conv::Conv2dConfig;
use burn::nn::PaddingConfig2d;
use burn::prelude::*;
use burn::tensor::Distribution;

fn case(cin: usize, cout: usize, s: usize, stride: usize, k: usize) {
    let dev: WgpuDevice = Default::default();
    let cdev = NdArrayDevice::default();
    let cfg = Conv2dConfig::new([cin, cout], [k, k])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(k / 2, k / 2, k / 2, k / 2));
    let cg = cfg.init::<Autodiff<Wgpu>>(&dev);
    let x = Tensor::<Autodiff<Wgpu>, 4>::random([32, cin, s, s], Distribution::Uniform(-1.0, 1.0), &dev);

    // CPU reference with identical weights/input
    let cc = cfg.init::<Autodiff<NdArray>>(&cdev);
    let wdata = cg.weight.val().inner().into_data();
    let xdata = x.clone().inner().into_data();
    let cc = {
        let mut r = cc;
        r.weight = burn::module::Param::initialized(r.weight.id, Tensor::from_data(wdata.clone(), &cdev).require_grad());
        r.bias = None;
        r
    };
    let cg = {
        let mut r = cg;
        r.bias = None;
        r
    };
    let xc = Tensor::<Autodiff<NdArray>, 4>::from_data(xdata, &cdev);

    let g_gpu = cg.forward(x.clone()).sum().backward();
    let wg_gpu = cg.weight.grad(&g_gpu).unwrap().into_data().to_vec::<f32>().unwrap();
    let g_cpu = cc.forward(xc).sum().backward();
    let wg_cpu = cc.weight.grad(&g_cpu).unwrap().into_data().to_vec::<f32>().unwrap();
    let mut worst = 0f32;
    let mut denom = 0f32;
    for (a, b) in wg_gpu.iter().zip(&wg_cpu) {
        worst = worst.max((a - b).abs());
        denom = denom.max(b.abs());
    }
    let rel = worst / denom.max(1e-6);

    // bench
    let run = || {
        let g = cg.forward(x.clone()).sum().backward();
        let wg: Tensor<Wgpu, 4> = cg.weight.grad(&g).unwrap();
        let _ = wg.sum().into_scalar();
    };
    for _ in 0..8 { run(); }
    let t = Instant::now();
    let iters = 15;
    for _ in 0..iters { run(); }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("  {cin:4}->{cout:4} @{s:3} s{stride} k{k}: rel_err={rel:.2e}  fwd+wgrad={ms:7.2} ms  {}", if rel < 1e-3 { "OK" } else { "FAIL" });
}

fn main() {
    println!("mode CUBECL_WGRAD={:?}", std::env::var("CUBECL_WGRAD").ok());
    // MicroChad shapes (batch 32 to keep CPU ref fast; relative behavior identical)
    case(4, 14, 128, 1, 3);
    case(14, 21, 64, 1, 3);
    case(21, 32, 32, 1, 3);
    case(32, 47, 16, 1, 3);
    case(47, 70, 8, 1, 3);
    case(70, 106, 4, 1, 3);
    // MNv4-ish dense shapes: stem s2 + ER expands
    case(4, 32, 128, 2, 3);
    case(32, 32, 64, 2, 3);
    case(32, 64, 32, 2, 3);
}
