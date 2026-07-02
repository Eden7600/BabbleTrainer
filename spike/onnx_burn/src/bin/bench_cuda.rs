// Diagnostic: same MNv4 train-step benchmark on Burn's CUDA backend (isolates wgpu-kernel vs CubeCL).
use std::time::Instant;

use burn::backend::cuda::CudaDevice;
use burn::backend::{Autodiff, Cuda};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::Distribution;
use onnx_burn::mnv4::Mnv4;

type AB = Autodiff<Cuda>;

fn step(model: Mnv4<AB>, opt: &mut impl Optimizer<Mnv4<AB>, AB>, x: &Tensor<AB, 4>, y: &Tensor<AB, 2>) -> Mnv4<AB> {
    let loss = (model.forward(x.clone()) - y.clone()).powf_scalar(2.0).mean();
    let gp = GradientsParams::from_grads(loss.backward(), &model);
    opt.step(1e-4, model, gp)
}

fn bench(batch: usize, iters: usize, device: &CudaDevice) {
    let mut model = Mnv4::<AB>::new(4, 4, device);
    let mut opt = AdamWConfig::new().with_weight_decay(1e-4).init();
    let x = Tensor::<AB, 4>::random([batch, 4, 128, 128], Distribution::Uniform(0.0, 1.0), device);
    let y = Tensor::<AB, 2>::random([batch, 4], Distribution::Uniform(0.0, 1.0), device);
    for _ in 0..25 {
        model = step(model, &mut opt, &x, &y);
    }
    let _ = model.forward(x.clone()).sum().into_scalar();
    let t = Instant::now();
    for _ in 0..iters {
        model = step(model, &mut opt, &x, &y);
    }
    let _ = model.forward(x.clone()).sum().into_scalar();
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("cuda TRAIN batch {batch:3}: {ms:6.1} ms/step | {:.2} ms/sample", ms / batch as f64);
}

fn main() {
    let device: CudaDevice = Default::default();
    println!("Burn CUDA backend, MNv4 expr net, fwd+bwd+AdamW:");
    for b in [32usize, 64, 96, 128] {
        bench(b, 50, &device);
    }
}
