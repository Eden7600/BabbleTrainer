// Exponential-moving-average "teacher": a smoothed copy of the model updated toward the student every
// few steps. BatchNorm running stats flow through map_float exactly like learnable params, so a single
// pass averages the full state (matching BabbleTrainer's EMATeacher). The teacher is on the inner
// (non-autodiff) backend, so blending builds no autograd graph; it is what gets evaluated + exported,
// and later drives the Mean-Teacher consistency loss. All on-device (no host sync).
use burn::module::{Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::optim::GradientsParams;
use burn::tensor::{backend::Backend, Tensor};

struct Collect {
    store: GradientsParams,
}
impl<B: Backend> ModuleVisitor<B> for Collect {
    fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
        self.store.register::<B, D>(p.id, p.val());
    }
}

struct Blend {
    store: GradientsParams,
    keep: f32, // teacher weight; student weight = 1-keep
}
impl<B: Backend> ModuleMapper<B> for Blend {
    fn map_float<const D: usize>(&mut self, p: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let id: ParamId = p.id;
        let t = p.val();
        match self.store.get::<B, D>(id) {
            Some(s) => Param::initialized(id, t.mul_scalar(self.keep) + s.mul_scalar(1.0 - self.keep)),
            None => Param::initialized(id, t),
        }
    }
}

// teacher <- keep*teacher + (1-keep)*student, over every param + BN buffer. Both modules on backend B
// (the inner/non-autodiff backend). ParamIds are stable across optimizer steps + valid(), so the
// lookup matches by identity, not traversal order.
pub fn ema_update<B: Backend, M: Module<B>>(teacher: M, student: &M, keep: f32) -> M {
    let mut c = Collect { store: GradientsParams::new() };
    student.visit(&mut c);
    teacher.map(&mut Blend { store: c.store, keep })
}
