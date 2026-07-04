// Isolated bench: per-tensor AdamW+EMA (burn) vs flat multi-tensor AdamW+EMA (flat_opt) on the
// real Mnv4 expr net, wgpu. Grads are captured once from a real backward and re-registered every
// iter (handle clones), so the loop measures ONLY optimizer + EMA dispatch/exec cost.
use std::collections::HashSet;
use std::time::Instant;

use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use burn::module::{AutodiffModule, Module, ModuleVisitor, Param, ParamId};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::Distribution;

use onnx_burn::data::{S, T};
use onnx_burn::ema::ema_update;
use onnx_burn::flat_opt::{Filter, FlatAdamW, FlatEma};
use onnx_burn::mnv4::Mnv4;

type AB = Autodiff<Wgpu>;
type IB = Wgpu;

struct GradBank {
    grads: Vec<(ParamId, usize, Tensor<IB, 1>, Vec<usize>)>,
}

impl GradBank {
    fn capture<M: AutodiffModule<AB>>(model: &M, gp: &mut GradientsParams) -> Self {
        struct V<'a> {
            gp: &'a mut GradientsParams,
            out: Vec<(ParamId, usize, Tensor<IB, 1>, Vec<usize>)>,
        }
        impl ModuleVisitor<AB> for V<'_> {
            fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<AB, D>>) {
                if let Some(g) = self.gp.remove::<IB, D>(p.id) {
                    let dims = g.dims().to_vec();
                    let len: usize = dims.iter().product();
                    self.out.push((p.id, D, g.reshape([len]), dims));
                }
            }
        }
        let mut v = V { gp, out: Vec::new() };
        model.visit(&mut v);
        Self { grads: v.out }
    }

    fn make(&self) -> GradientsParams {
        let mut gp = GradientsParams::new();
        for (id, rank, g, dims) in &self.grads {
            match rank {
                1 => gp.register::<IB, 1>(*id, g.clone()),
                2 => gp.register::<IB, 2>(*id, g.clone().reshape([dims[0], dims[1]])),
                3 => gp.register::<IB, 3>(*id, g.clone().reshape([dims[0], dims[1], dims[2]])),
                4 => gp.register::<IB, 4>(*id, g.clone().reshape([dims[0], dims[1], dims[2], dims[3]])),
                _ => unreachable!(),
            }
        }
        gp
    }
}

fn main() {
    let device = WgpuDevice::default();
    let iters: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(200);

    let mut model = Mnv4::<AB>::new(T, 4, &device);
    let _ = model.num_params();
    let head_ids = model.head_ids();
    let head_set: HashSet<ParamId> = head_ids.iter().copied().collect();
    let bn_ids: HashSet<ParamId> = model.running_state_ids().into_iter().collect();

    let x = Tensor::<AB, 4>::random([8, T, S, S], Distribution::Default, &device);
    let mut grads = model.forward(x).powf_scalar(2.0).mean().backward();
    let mut head_gp0 = GradientsParams::from_params(&mut grads, &model, &head_ids);
    let mut bb_gp0 = GradientsParams::from_module(&mut grads, &model);
    let head_bank = GradBank::capture(&model, &mut head_gp0);
    let bb_bank = GradBank::capture(&model, &mut bb_gp0);
    println!("params: head={} bb={}", head_bank.grads.len(), bb_bank.grads.len());

    // --- per-tensor (burn) ---
    let mut o_head = AdamWConfig::new().with_weight_decay(1e-4).init();
    let mut o_bb = AdamWConfig::new().with_weight_decay(1e-4).init();
    let mut teacher = model.valid();
    for _ in 0..20 {
        model = o_head.step(1e-4, model, head_bank.make());
        model = o_bb.step(1e-4, model, bb_bank.make());
        teacher = ema_update::<IB, _>(teacher, &model.valid(), 0.999, &bn_ids);
    }
    let _ = IB::sync(&device);
    let t0 = Instant::now();
    for _ in 0..iters {
        model = o_head.step(1e-4, model, head_bank.make());
        model = o_bb.step(1e-4, model, bb_bank.make());
        teacher = ema_update::<IB, _>(teacher, &model.valid(), 0.999, &bn_ids);
    }
    let _ = IB::sync(&device);
    let per_tensor = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("per-tensor AdamW+EMA: {per_tensor:.2} ms/step");

    // --- flat ---
    let mut f_head = FlatAdamW::<IB>::new(1e-4, Filter::Include(head_set.clone()));
    let mut f_bb = FlatAdamW::<IB>::new(1e-4, Filter::Exclude(head_set));
    let mut f_ema = FlatEma::<IB>::new();
    for _ in 0..20 {
        model = f_head.step(1e-4, model, head_bank.make());
        model = f_bb.step(1e-4, model, bb_bank.make());
        teacher = f_ema.update(teacher, &model.valid(), &[&f_head, &f_bb], 0.999, &bn_ids);
    }
    let _ = IB::sync(&device);
    let t0 = Instant::now();
    for _ in 0..iters {
        model = f_head.step(1e-4, model, head_bank.make());
        model = f_bb.step(1e-4, model, bb_bank.make());
        teacher = f_ema.update(teacher, &model.valid(), &[&f_head, &f_bb], 0.999, &bn_ids);
    }
    let _ = IB::sync(&device);
    let flat = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("flat AdamW+EMA:       {flat:.2} ms/step ({:.2}x)", per_tensor / flat);
}
