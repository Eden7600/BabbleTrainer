// Per-op forward benchmark on wgpu to localize the slow kernel(s). Each op timed identically
// (per-iter sync via into_scalar), so relative ms exposes the culprit (suspect: depthwise conv).
use std::time::Instant;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::nn::conv::Conv2dConfig;
use burn::nn::{BatchNormConfig, PaddingConfig2d};
use burn::prelude::*;
use burn::tensor::Distribution;

type B = Wgpu;

fn conv(cin: usize, cout: usize, k: usize, stride: usize, groups: usize, dev: &WgpuDevice) -> burn::nn::conv::Conv2d<B> {
    Conv2dConfig::new([cin, cout], [k, k])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(k / 2, k / 2, k / 2, k / 2))
        .with_groups(groups)
        .with_bias(false)
        .init(dev)
}

fn time_conv(name: &str, c: &burn::nn::conv::Conv2d<B>, x: &Tensor<B, 4>, iters: usize) {
    for _ in 0..20 {
        let _ = c.forward(x.clone()).sum().into_scalar();
    }
    let t = Instant::now();
    for _ in 0..iters {
        let _ = c.forward(x.clone()).sum().into_scalar();
    }
    println!("{name:28} {:6.2} ms/fwd", t.elapsed().as_secs_f64() * 1000.0 / iters as f64);
}

fn main() {
    let dev: WgpuDevice = Default::default();
    let batch = 64;
    let rnd = |shape: [usize; 4]| Tensor::<B, 4>::random(shape, Distribution::Uniform(0.0, 1.0), &dev);
    let iters = 40;
    println!("Burn wgpu per-op forward (batch {batch}):");

    let stem = conv(4, 32, 3, 2, 1, &dev);
    time_conv("stem conv3x3 s2 4->32 @128", &stem, &rnd([batch, 4, 128, 128]), iters);

    let pw = conv(32, 96, 1, 1, 1, &dev);
    time_conv("pointwise 1x1 32->96 @32", &pw, &rnd([batch, 32, 32, 32]), iters);

    let dw3 = conv(96, 96, 3, 1, 96, &dev);
    time_conv("depthwise 3x3 g96 @32", &dw3, &rnd([batch, 96, 32, 32]), iters);

    let dw5 = conv(96, 96, 5, 1, 96, &dev);
    time_conv("depthwise 5x5 g96 @16", &dw5, &rnd([batch, 96, 16, 16]), iters);

    let big = conv(64, 256, 1, 1, 1, &dev);
    time_conv("pointwise 1x1 64->256 @4", &big, &rnd([batch, 64, 4, 4]), iters);

    let bn: burn::nn::BatchNorm<B> = BatchNormConfig::new(96).init(&dev);
    {
        let x = rnd([batch, 96, 32, 32]);
        for _ in 0..20 {
            let _ = bn.forward(x.clone()).sum().into_scalar();
        }
        let t = Instant::now();
        for _ in 0..iters {
            let _ = bn.forward(x.clone()).sum().into_scalar();
        }
        println!("{:28} {:6.2} ms/fwd", "batchnorm 96 @32", t.elapsed().as_secs_f64() * 1000.0 / iters as f64);
    }
}
