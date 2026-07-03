// Exponential-moving-average "teacher": a smoothed copy of the model. Learnable params are decayed
// toward the student (teacher <- keep*teacher + (1-keep)*student); BatchNorm running stats are COPIED
// from the student, NOT decayed — matching BabbleTrainer's EMATeacher, which decays parameters but
// `torch._foreach_copy_`s buffers (model.py). Copying matters: the teacher runs BN in inference mode
// (ad disabled), so it never updates its own running stats — decaying them would make the deployed
// teacher normalize with stats that lag the student by a second EMA. `copy` holds the running-stat
// ParamIds (from Mnv4::running_state_ids). The teacher is on the inner (non-autodiff) backend, so this
// builds no autograd graph; it is what gets evaluated + exported and drives the Mean-Teacher loss.
use std::collections::HashSet;

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

struct Blend<'a> {
    store: GradientsParams,
    keep: f32, // teacher weight; student weight = 1-keep (learnable params only)
    copy: &'a HashSet<ParamId>, // BN running-stat ids: take the student value verbatim
    misses: usize, // teacher params with no matching student ParamId (a structure/ordering bug)
}
impl<B: Backend> ModuleMapper<B> for Blend<'_> {
    fn map_float<const D: usize>(&mut self, p: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let id: ParamId = p.id;
        let t = p.val();
        match self.store.get::<B, D>(id) {
            Some(s) if self.copy.contains(&id) => Param::initialized(id, s), // buffer copy
            Some(s) => Param::initialized(id, t.mul_scalar(self.keep) + s.mul_scalar(1.0 - self.keep)),
            None => {
                // No student match => teacher weight frozen. Silent today only because head param
                // ordering happens to line up; count so a future reorder fails loudly, not silently.
                self.misses += 1;
                Param::initialized(id, t)
            }
        }
    }
}

// ParamIds are stable across optimizer steps + valid(), so the lookup matches by identity, not
// traversal order. `copy` = the BN running-stat ids to hard-copy from the student. Panics if any
// teacher param fails to match a student ParamId (would ship a frozen/random-init weight otherwise).
pub fn ema_update<B: Backend, M: Module<B>>(
    teacher: M, student: &M, keep: f32, copy: &HashSet<ParamId>,
) -> M {
    let mut c = Collect { store: GradientsParams::new() };
    student.visit(&mut c);
    let mut blend = Blend { store: c.store, keep, copy, misses: 0 };
    let out = teacher.map(&mut blend);
    assert!(blend.misses == 0, "EMA: {} teacher params had no matching student ParamId (teacher/student diverged)", blend.misses);
    out
}
