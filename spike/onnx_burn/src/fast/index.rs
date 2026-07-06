use crate::data::Sample;
use crate::sampler::Rng;

// Mixup source pools (expr only): neutral = max expr label < 0.2, expr = max expr label > 0.8, all =
// the full train subset for general cross-frame blends.
pub(crate) struct Pools {
    pub(crate) neutral: Vec<usize>,
    pub(crate) expr: Vec<usize>,
    pub(crate) all: Vec<usize>,
}

impl Pools {
    pub(crate) fn expr_build(samples: &[Sample], train: &[usize]) -> Self {
        let m = |i: usize| samples[i].label[2..6].iter().cloned().fold(0.0f32, f32::max);
        Self {
            neutral: train.iter().cloned().filter(|&i| m(i) < 0.2).collect(),
            expr: train.iter().cloned().filter(|&i| m(i) > 0.8).collect(),
            all: train.to_vec(),
        }
    }
    pub(crate) fn empty() -> Self {
        Self { neutral: vec![], expr: vec![], all: vec![] }
    }
}

// Paired-eye index: records where BOTH sides are in `train`, with the 4-dim expr labels of each eye.
// Drives the evenness (L/R symmetry) and consensus-head terms. active-boost weighted like the expr pool.
pub(crate) struct Paired {
    pub(crate) recs: Vec<usize>,
    pub(crate) ll: Vec<[f32; 4]>,
    pub(crate) rl: Vec<[f32; 4]>,
    pub(crate) cum: Vec<f32>,
    pub(crate) total: f32,
}

impl Paired {
    pub(crate) fn build(samples: &[Sample], train: &[usize]) -> Self {
        use std::collections::HashMap;
        let (mut left, mut right) = (HashMap::new(), HashMap::new());
        for &i in train {
            let s = &samples[i];
            let e = [s.label[2], s.label[3], s.label[4], s.label[5]];
            if s.side == 0 { left.insert(s.rec, e); } else { right.insert(s.rec, e); }
        }
        let (mut recs, mut ll, mut rl, mut cum) = (vec![], vec![], vec![], vec![]);
        let mut acc = 0.0f32;
        let mut keys: Vec<usize> = left.keys().cloned().filter(|k| right.contains_key(k)).collect();
        keys.sort_unstable();
        for rec in keys {
            let (l, r) = (left[&rec], right[&rec]);
            let m = l.iter().chain(r.iter()).cloned().fold(0.0f32, f32::max);
            acc += 1.0 + 4.0 * m;
            recs.push(rec);
            ll.push(l);
            rl.push(r);
            cum.push(acc);
        }
        Self { recs, ll, rl, cum, total: acc }
    }
    pub(crate) fn sample(&self, rng: &mut Rng) -> usize {
        let u = rng.f01() * self.total;
        self.cum.partition_point(|&c| c < u).min(self.cum.len().saturating_sub(1))
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.recs.is_empty()
    }
}

// Unlabeled anti-jitter stream = the capture's own FLAG_EXPR_UNLABELED (free-expression gaze-section)
// frames — the same in-domain footage the frontend uses. Yields (cur_strong, cur_weak, nbr_weak,
// pair_ok) for the mean-teacher + temporal-smoothness losses (stage B).
pub(crate) struct Unlabeled {
    pub(crate) items: Vec<(usize, usize)>, // (rec, side)
}

impl Unlabeled {
    pub(crate) fn build(samples: &[Sample], train: &[usize]) -> Self {
        let items = train
            .iter()
            .filter_map(|&i| {
                let s = &samples[i];
                (s.expr_unlabeled > 0.5).then_some((s.rec, s.side))
            })
            .collect();
        Self { items }
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
