// Decompose MicroChad conv backward: wgrad-only (input constant) vs wgrad+dgrad (input grad).
// Also bench an unfold(im2col)+matmul formulation of the same conv for comparison.
use std::time::Instant;

use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::prelude::*;
use burn::tensor::module::unfold4d;
use burn::tensor::ops::UnfoldOptions;
use burn::tensor::Distribution;

type AB = Autodiff<Wgpu>;

fn conv(cin: usize, cout: usize, dev: &WgpuDevice) -> Conv2d<AB> {
    Conv2dConfig::new([cin, cout], [3, 3])
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .init(dev)
}

fn time_it<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    for _ in 0..8 { f(); }
    let t = Instant::now();
    for _ in 0..iters { f(); }
    t.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn main() {
    let dev: WgpuDevice = Default::default();
    let batch = 64;
    let iters = 15;
    let layers: [(usize, usize, usize); 6] =
        [(4, 14, 128), (14, 21, 64), (21, 32, 32), (32, 47, 16), (47, 70, 8), (70, 106, 4)];
    println!("per-layer (batch {batch}): wgrad-only | wgrad+dgrad | unfold-matmul wgrad-only");
    for (cin, cout, s) in layers {
        let c = conv(cin, cout, &dev);
        let x_const = Tensor::<AB, 4>::random([batch, cin, s, s], Distribution::Uniform(0.0, 1.0), &dev);
        let x_grad = x_const.clone().require_grad();

        let w_only = time_it(|| {
            let g = c.forward(x_const.clone()).sum().backward();
            let wg: Tensor<Wgpu, 4> = c.weight.grad(&g).unwrap();
            let _ = wg.sum().into_scalar();
        }, iters);
        let w_d = time_it(|| {
            let g = c.forward(x_grad.clone()).sum().backward();
            let wg: Tensor<Wgpu, 4> = c.weight.grad(&g).unwrap();
            let _ = wg.sum().into_scalar();
        }, iters);

        // unfold formulation: cols [B, cin*9, L], w2 [cout, cin*9] -> out [B, cout, L]
        let w2 = c.weight.val().reshape([cout, cin * 9]).detach().require_grad();
        let uf = time_it(|| {
            let cols = unfold4d(x_const.clone(), [3, 3], UnfoldOptions::new([1, 1], [1, 1], [1, 1]));
            let out = w2.clone().unsqueeze::<3>().matmul(cols); // [B, cout, L]
            let g = out.sum().backward();
            let wg: Tensor<Wgpu, 2> = w2.grad(&g).unwrap();
            let _ = wg.sum().into_scalar();
        }, iters);
        println!("  {cin:3}->{cout:3} @{s:3}:  {w_only:7.2} | {w_d:7.2} | {uf:7.2} ms");
    }
}
