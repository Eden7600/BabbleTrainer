// Fused multi-tensor AdamW + EMA. All learnable params of an optimizer group live in one flat 1-D
// arena (params + both moments), so a step is one grad-cat + ~10 big elementwise ops + a slice-out
// per param, instead of ~10 small ops per param x ~230 params. The update rule matches burn 0.21
// AdamW exactly (decoupled decay p*(1-lr*wd), bias-corrected moments, eps outside sqrt), verified
// bit-close in tests below. NOT for DirectML (per-tensor required there); the spike trains on wgpu.
use std::collections::{HashMap, HashSet};

use burn::module::{AutodiffModule, Module, ModuleMapper, ModuleVisitor, Param, ParamId};
use burn::optim::GradientsParams;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::Tensor;

#[derive(Clone)]
struct Entry {
    id: ParamId,
    rank: usize,
    off: usize,
    len: usize,
}

/// Group membership: which of the module's grad-bearing params this optimizer owns.
pub enum Filter {
    All,
    Include(HashSet<ParamId>),
    Exclude(HashSet<ParamId>),
}

impl Filter {
    fn ok(&self, id: ParamId) -> bool {
        match self {
            Filter::All => true,
            Filter::Include(s) => s.contains(&id),
            Filter::Exclude(s) => !s.contains(&id),
        }
    }
}

pub struct FlatAdamW<B: Backend> {
    filter: Filter,
    entries: Vec<Entry>,
    p: Option<Tensor<B, 1>>,
    m: Option<Tensor<B, 1>>,
    v: Option<Tensor<B, 1>>,
    t: i32,
    b1: f32,
    b2: f32,
    eps: f32,
    wd: f32,
}

impl<B: Backend> FlatAdamW<B> {
    pub fn new(wd: f32, filter: Filter) -> Self {
        Self { filter, entries: Vec::new(), p: None, m: None, v: None, t: 0, b1: 0.9, b2: 0.999, eps: 1e-5, wd }
    }

    /// Build the layout + param arena from the module's grad-bearing params (once). Safe to call
    /// every step; a no-op after the first. Call even on frozen steps so the EMA can see the group.
    pub fn ensure_init<AB, M>(&mut self, module: &M, grads: &GradientsParams)
    where
        AB: AutodiffBackend<InnerBackend = B>,
        M: AutodiffModule<AB>,
    {
        if !self.entries.is_empty() {
            return;
        }
        let mut c = Collect1D::<AB> { filter: &self.filter, grads, entries: Vec::new(), parts: Vec::new(), off: 0 };
        module.visit(&mut c);
        assert!(!c.entries.is_empty(), "FlatAdamW: no params matched the filter/grads");
        self.entries = c.entries;
        let p = Tensor::cat(c.parts, 0);
        self.m = Some(Tensor::zeros_like(&p));
        self.v = Some(Tensor::zeros_like(&p));
        self.p = Some(p);
    }

    pub fn step<AB, M>(&mut self, lr: f64, module: M, mut grads: GradientsParams) -> M
    where
        AB: AutodiffBackend<InnerBackend = B>,
        M: AutodiffModule<AB>,
    {
        self.ensure_init(&module, &grads);
        let parts: Vec<Tensor<B, 1>> = self
            .entries
            .iter()
            .map(|e| grad_1d::<B>(&mut grads, e).expect("FlatAdamW: param in layout has no grad this step"))
            .collect();
        let g = Tensor::cat(parts, 0);

        self.t += 1;
        let m = self.m.take().unwrap().mul_scalar(self.b1) + g.clone().mul_scalar(1.0 - self.b1);
        let v = self.v.take().unwrap().mul_scalar(self.b2) + g.square().mul_scalar(1.0 - self.b2);
        let m_hat = m.clone().div_scalar(1.0 - self.b1.powi(self.t));
        let v_hat = v.clone().div_scalar(1.0 - self.b2.powi(self.t));
        let delta = m_hat.div(v_hat.sqrt().add_scalar(self.eps));
        let decay = lr * self.wd as f64;
        let p = self.p.take().unwrap().mul_scalar(1.0 - decay) - delta.mul_scalar(lr);
        self.m = Some(m);
        self.v = Some(v);
        self.p = Some(p.clone());

        let mut s = ScatterAd::<AB> { slices: self.slices(&p) };
        let module = module.map(&mut s);
        assert!(s.slices.is_empty(), "FlatAdamW: {} layout params missing from module", s.slices.len());
        module
    }

    fn slices(&self, p: &Tensor<B, 1>) -> HashMap<ParamId, Tensor<B, 1>> {
        self.entries
            .iter()
            .map(|e| (e.id, p.clone().slice([e.off..e.off + e.len])))
            .collect()
    }

    fn flat(&self) -> Option<Tensor<B, 1>> {
        self.p.clone()
    }
}

fn grad_1d<B: Backend>(grads: &mut GradientsParams, e: &Entry) -> Option<Tensor<B, 1>> {
    match e.rank {
        1 => grads.remove::<B, 1>(e.id),
        2 => grads.remove::<B, 2>(e.id).map(|t| t.reshape([e.len])),
        3 => grads.remove::<B, 3>(e.id).map(|t| t.reshape([e.len])),
        4 => grads.remove::<B, 4>(e.id).map(|t| t.reshape([e.len])),
        r => panic!("FlatAdamW: unsupported param rank {r}"),
    }
}

fn has_grad<B: Backend>(grads: &GradientsParams, id: ParamId, rank: usize) -> bool {
    match rank {
        1 => grads.get::<B, 1>(id).is_some(),
        2 => grads.get::<B, 2>(id).is_some(),
        3 => grads.get::<B, 3>(id).is_some(),
        4 => grads.get::<B, 4>(id).is_some(),
        _ => false,
    }
}

struct Collect1D<'a, AB: AutodiffBackend> {
    filter: &'a Filter,
    grads: &'a GradientsParams,
    entries: Vec<Entry>,
    parts: Vec<Tensor<AB::InnerBackend, 1>>,
    off: usize,
}

impl<AB: AutodiffBackend> ModuleVisitor<AB> for Collect1D<'_, AB> {
    fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<AB, D>>) {
        if !self.filter.ok(p.id) || !has_grad::<AB::InnerBackend>(self.grads, p.id, D) {
            return;
        }
        let len: usize = p.val().dims().iter().product();
        self.entries.push(Entry { id: p.id, rank: D, off: self.off, len });
        self.parts.push(p.val().inner().reshape([len]));
        self.off += len;
    }
}

struct ScatterAd<AB: AutodiffBackend> {
    slices: HashMap<ParamId, Tensor<AB::InnerBackend, 1>>,
}

impl<AB: AutodiffBackend> ModuleMapper<AB> for ScatterAd<AB> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<AB, D>>) -> Param<Tensor<AB, D>> {
        let (id, tensor, mapper) = param.consume();
        let tensor = match self.slices.remove(&id) {
            Some(flat) => {
                let rg = tensor.is_require_grad();
                let mut t = Tensor::from_inner(flat.reshape(tensor.dims()));
                if rg {
                    t = t.require_grad();
                }
                t
            }
            None => tensor,
        };
        Param::from_mapped_value(id, tensor, mapper)
    }
}

/// Flat EMA teacher update over the FlatAdamW groups' param arenas: 3 big elementwise ops + one
/// slice-out per param, instead of 3 small ops per param. BN running stats (`copy` ids) are moved
/// from the student verbatim (handle move, no GPU math) — same semantics as ema::ema_update.
pub struct FlatEma<B: Backend> {
    flat_t: Option<Tensor<B, 1>>,
}

impl<B: Backend> FlatEma<B> {
    pub fn new() -> Self {
        Self { flat_t: None }
    }

    /// Call when the teacher is re-cloned from the student (stage boundary): the next update
    /// re-captures the teacher arena from the teacher module itself.
    pub fn reset(&mut self) {
        self.flat_t = None;
    }

    pub fn update<M: Module<B>>(
        &mut self,
        teacher: M,
        student: &M,
        groups: &[&FlatAdamW<B>],
        keep: f32,
        copy: &HashSet<ParamId>,
    ) -> M {
        // Combined layout = the groups' entries, re-offset into one concatenated arena.
        let mut entries: Vec<Entry> = Vec::new();
        let mut base = 0usize;
        for g in groups {
            assert!(!g.entries.is_empty(), "FlatEma: group not initialized (call ensure_init first)");
            for e in &g.entries {
                entries.push(Entry { off: base + e.off, ..e.clone() });
            }
            base += g.entries.iter().map(|e| e.len).sum::<usize>();
        }
        let flat_s = Tensor::cat(groups.iter().map(|g| g.flat().unwrap()).collect(), 0);
        let flat_t = match self.flat_t.take() {
            Some(t) => t,
            None => gather_by_layout(&teacher, &entries, base),
        };
        let flat_t = flat_t.mul_scalar(keep) + flat_s.mul_scalar(1.0 - keep);
        self.flat_t = Some(flat_t.clone());

        // Student BN running-stat values, taken verbatim (they change every forward).
        let mut cv = CopyVals::<B> { ids: copy, out: HashMap::new() };
        student.visit(&mut cv);

        let slices: HashMap<ParamId, Tensor<B, 1>> = entries
            .iter()
            .map(|e| (e.id, flat_t.clone().slice([e.off..e.off + e.len])))
            .collect();
        let mut s = ScatterInner::<B> { slices, copies: cv.out, misses: 0 };
        let out = teacher.map(&mut s);
        assert!(s.misses == 0, "FlatEma: {} teacher params neither blended nor copied", s.misses);
        out
    }
}

fn gather_by_layout<B: Backend, M: Module<B>>(module: &M, entries: &[Entry], total: usize) -> Tensor<B, 1> {
    let mut cv = CollectFlat::<B> { ids: entries.iter().map(|e| e.id).collect(), out: HashMap::new() };
    module.visit(&mut cv);
    let parts: Vec<Tensor<B, 1>> = entries
        .iter()
        .map(|e| cv.out.remove(&e.id).expect("FlatEma: teacher missing a layout param"))
        .collect();
    let flat = Tensor::cat(parts, 0);
    debug_assert_eq!(flat.dims()[0], total);
    flat
}

struct CollectFlat<B: Backend> {
    ids: HashSet<ParamId>,
    out: HashMap<ParamId, Tensor<B, 1>>,
}

impl<B: Backend> ModuleVisitor<B> for CollectFlat<B> {
    fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
        if self.ids.contains(&p.id) {
            let len: usize = p.val().dims().iter().product();
            self.out.insert(p.id, p.val().reshape([len]));
        }
    }
}

struct CopyVals<'a, B: Backend> {
    ids: &'a HashSet<ParamId>,
    out: HashMap<ParamId, Tensor<B, 1>>,
}

impl<B: Backend> ModuleVisitor<B> for CopyVals<'_, B> {
    fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
        if self.ids.contains(&p.id) {
            let len: usize = p.val().dims().iter().product();
            self.out.insert(p.id, p.val().reshape([len]));
        }
    }
}

struct ScatterInner<B: Backend> {
    slices: HashMap<ParamId, Tensor<B, 1>>,
    copies: HashMap<ParamId, Tensor<B, 1>>,
    misses: usize,
}

impl<B: Backend> ModuleMapper<B> for ScatterInner<B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let tensor = if let Some(c) = self.copies.remove(&id) {
            c.reshape(tensor.dims())
        } else if let Some(flat) = self.slices.remove(&id) {
            flat.reshape(tensor.dims())
        } else {
            self.misses += 1;
            tensor
        };
        Param::from_mapped_value(id, tensor, mapper)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{S, T};
    use crate::ema::ema_update;
    use crate::mnv4::Mnv4;
    use burn::backend::{Autodiff, NdArray};
    use burn::optim::{AdamWConfig, Optimizer};
    use burn::tensor::{Distribution, Tensor};

    type AB = Autodiff<NdArray<f32>>;
    type IB = NdArray<f32>;

    // Max |a-b| across every float param of two identical-structure modules.
    fn max_param_diff<M: Module<IB>>(a: &M, b: &M) -> f32 {
        struct Grab {
            out: HashMap<ParamId, Tensor<IB, 1>>,
        }
        impl ModuleVisitor<IB> for Grab {
            fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<IB, D>>) {
                let len: usize = p.val().dims().iter().product();
                self.out.insert(p.id, p.val().reshape([len]));
            }
        }
        let mut ga = Grab { out: HashMap::new() };
        let mut gb = Grab { out: HashMap::new() };
        a.visit(&mut ga);
        b.visit(&mut gb);
        assert_eq!(ga.out.len(), gb.out.len());
        let mut worst = 0f32;
        for (id, ta) in ga.out {
            let tb = gb.out.remove(&id).expect("param id mismatch");
            let d = (ta - tb).abs().max().into_scalar();
            worst = worst.max(d);
        }
        worst
    }

    // 6 steps on the real Mnv4 (head/backbone split, backbone frozen for 2 steps, LR sweep) must
    // match burn's per-tensor AdamW to float tolerance; then a strided EMA must match ema_update.
    #[test]
    fn flat_adamw_and_ema_match_reference() {
        let device = Default::default();
        let model0 = Mnv4::<AB>::new(T, 4, &device);
        let _ = model0.num_params(); // materialize lazy params BEFORE cloning (else clones roll different init)
        let head_ids = model0.head_ids();
        let bn_ids: HashSet<ParamId> = model0.running_state_ids().into_iter().collect();
        let head_set: HashSet<ParamId> = head_ids.iter().copied().collect();

        let mut m_ref = model0.clone();
        let mut m_flat = model0.clone();
        let mut o_head_ref = AdamWConfig::new().with_weight_decay(1e-4).init();
        let mut o_bb_ref = AdamWConfig::new().with_weight_decay(1e-4).init();
        let mut o_head = FlatAdamW::<IB>::new(1e-4, Filter::Include(head_set.clone()));
        let mut o_bb = FlatAdamW::<IB>::new(1e-4, Filter::Exclude(head_set));

        let mut t_ref = m_ref.valid();
        let mut t_flat = m_flat.valid();
        let mut ema = FlatEma::<IB>::new();

        for step in 0..6usize {
            let frozen = step < 2;
            let (lr_h, lr_b) = (1e-3 * (step + 1) as f64, 5e-4 * (step + 1) as f64);
            let x = Tensor::<AB, 4>::random([2, T, S, S], Distribution::Default, &device);

            <AB as burn::tensor::backend::Backend>::seed(&device, 7 + step as u64);
            let mut g_ref = m_ref.forward(x.clone()).powf_scalar(2.0).mean().backward();
            let hg = GradientsParams::from_params(&mut g_ref, &m_ref, &head_ids);
            let bg = GradientsParams::from_module(&mut g_ref, &m_ref);
            m_ref = o_head_ref.step(lr_h, m_ref, hg);
            if !frozen {
                m_ref = o_bb_ref.step(lr_b, m_ref, bg);
            }

            <AB as burn::tensor::backend::Backend>::seed(&device, 7 + step as u64);
            let mut g_flat = m_flat.forward(x).powf_scalar(2.0).mean().backward();
            let hg = GradientsParams::from_params(&mut g_flat, &m_flat, &head_ids);
            let bg = GradientsParams::from_module(&mut g_flat, &m_flat);
            o_bb.ensure_init(&m_flat, &bg); // frozen steps too: EMA needs the group arena
            m_flat = o_head.step(lr_h, m_flat, hg);
            if !frozen {
                m_flat = o_bb.step(lr_b, m_flat, bg);
            }

            if step % 2 == 0 {
                let keep = 0.999f32.powi(2);
                t_ref = ema_update::<IB, _>(t_ref, &m_ref.valid(), keep, &bn_ids);
                t_flat = ema.update(t_flat, &m_flat.valid(), &[&o_head, &o_bb], keep, &bn_ids);
            }
        }

        let d_student = max_param_diff(&m_ref.valid(), &m_flat.valid());
        let d_teacher = max_param_diff(&t_ref, &t_flat);
        assert!(d_student < 2e-5, "student diverged: max diff {d_student}");
        assert!(d_teacher < 2e-5, "teacher diverged: max diff {d_teacher}");
    }
}

