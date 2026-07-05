// MicroChad conv-shape bench: exact training shapes, plain (odd channels) vs padded to multiples
// of 8/16 (vec4-friendly). If padding wins big, the gaze forward gets an exact-math padded path.
use std::time::Instant;

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::prelude::*;
use burn::tensor::Distribution;

type B = Wgpu;

fn conv(cin: usize, cout: usize, dev: &WgpuDevice) -> Conv2d<B> {
    Conv2dConfig::new([cin, cout], [3, 3])
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .init(dev)
}

fn bench(name: &str, c: &Conv2d<B>, x: &Tensor<B, 4>, iters: usize) -> f64 {
    for _ in 0..15 {
        let _ = c.forward(x.clone()).sum().into_scalar();
    }
    let t = Instant::now();
    for _ in 0..iters {
        let _ = c.forward(x.clone()).sum().into_scalar();
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("  {name:34} {ms:7.2} ms/fwd");
    ms
}

fn main() {
    let dev: WgpuDevice = Default::default();
    let batch = 64;
    let iters = 30;
    let rnd = |c: usize, s: usize| Tensor::<B, 4>::random([batch, c, s, s], Distribution::Uniform(0.0, 1.0), &dev);
    // (cin, cout, spatial) per MicroChad layer; padded = next multiple of 8 (in), 8 (out)
    let layers: [(usize, usize, usize); 6] =
        [(4, 14, 128), (14, 21, 64), (21, 32, 32), (32, 47, 16), (47, 70, 8), (70, 106, 4)];
    let pad8 = |c: usize| c.div_ceil(8) * 8;
    let pad16 = |c: usize| c.div_ceil(16) * 16;

    println!("MicroChad conv shapes, batch {batch} (fwd only):");
    let mut tot_plain = 0.0;
    let mut tot_p8 = 0.0;
    let mut tot_p16 = 0.0;
    for (cin, cout, s) in layers {
        let c0 = conv(cin, cout, &dev);
        tot_plain += bench(&format!("plain {cin}->{cout} @{s}"), &c0, &rnd(cin, s), iters);
        let (i8, o8) = (pad8(cin), pad8(cout));
        let c1 = conv(i8, o8, &dev);
        tot_p8 += bench(&format!("pad8  {i8}->{o8} @{s}"), &c1, &rnd(i8, s), iters);
        let (i16, o16) = (pad16(cin), pad16(cout));
        let c2 = conv(i16, o16, &dev);
        tot_p16 += bench(&format!("pad16 {i16}->{o16} @{s}"), &c2, &rnd(i16, s), iters);
        println!();
    }
    println!("TOTAL fwd: plain={tot_plain:.2} ms  pad8={tot_p8:.2} ms  pad16={tot_p16:.2} ms");
}
