// Minimal supervised expr training (Stage-A core). Curriculum/MT/gaze layered on later.
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use crate::data::{temporal_stack, Bin, EyeCache, Sample, S, T};

struct XorShift(u64);
impl XorShift {
    fn next_usize(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}
use crate::losses::supervised;
use crate::mnv4::Mnv4;

pub fn make_batch<B: Backend>(
    bin: &Bin,
    samples: &[Sample],
    idxs: &[usize],
    cache: &EyeCache,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 2>) {
    let b = idxs.len();
    let mut xs = Vec::with_capacity(b * T * S * S);
    let mut ys = Vec::with_capacity(b * 4);
    for &i in idxs {
        let s = &samples[i];
        let stack = temporal_stack(bin, s.rec, s.side, cache);
        xs.extend(stack.iter().map(|&v| v as f32 / 255.0));
        ys.extend_from_slice(&s.label[2..6]); // EXPR_SLICE = channels 2..6
    }
    let x = Tensor::from_data(TensorData::new(xs, [b, T, S, S]), device);
    let y = Tensor::from_data(TensorData::new(ys, [b, 4]), device);
    (x, y)
}

pub fn fit_supervised<B: AutodiffBackend>(
    bin: &Bin,
    samples: &[Sample],
    steps: usize,
    batch: usize,
    lr: f64,
    device: &B::Device,
) -> Mnv4<B> {
    let mut model = Mnv4::<B>::new(T, 4, device);
    let mut opt = AdamWConfig::new().with_weight_decay(1e-4).init();
    let cache = EyeCache::default();
    let mut rng = XorShift(0x2545F4914F6CDD1D);
    let n = samples.len();
    let mut first = 0.0f32;
    for step in 0..steps {
        let idxs: Vec<usize> = (0..batch).map(|_| rng.next_usize(n)).collect();
        let (x, y) = make_batch::<B>(bin, samples, &idxs, &cache, device);
        let loss = supervised(model.forward(x), y);
        let val = loss.clone().into_scalar().elem::<f32>();
        if step == 0 {
            first = val;
        }
        let grads = loss.backward();
        let gp = GradientsParams::from_grads(grads, &model);
        model = opt.step(lr, model, gp);
        if step % 5 == 0 || step == steps - 1 {
            println!("step {step:4} loss {val:.5}");
        }
    }
    println!("first->last loss: {first:.5} -> (see above)");
    model
}
