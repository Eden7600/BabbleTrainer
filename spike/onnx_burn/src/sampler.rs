// Deterministic frame split + weighted / gaze-stratified sampling (mirrors bin_temporal_dataset).
use crate::data::Sample;

pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn f01(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    pub fn usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// Block holdout (NOT per-frame random): samples are in capture order, so a random per-frame split
// leaks — val frames are near-duplicates of temporally-adjacent train frames. Instead cut the timeline
// into contiguous blocks and hold out every `stride`-th block, spread across the recording so val stays
// representative of all routine sections while only block boundaries touch train. seed offsets which
// blocks are held out.
pub fn split(n: usize, val_frac: f32, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let bsize = 64usize; // small contiguous blocks: val spread finely over the timeline (representative
                         // of every routine section) while only block edges touch train (~minimal leak)
    let stride = (1.0 / val_frac).round().max(2.0) as usize; // hold every stride-th block (~10 for 0.1)
    let off = (seed as usize) % stride;
    let (mut train, mut val) = (Vec::new(), Vec::new());
    for i in 0..n {
        if (i / bsize) % stride == off { val.push(i) } else { train.push(i) }
    }
    if val.is_empty() {
        val.push(n - 1);
    }
    (train, val)
}

// weighted-with-replacement over a subset; weight = 1 + active_boost*max(expr label).
pub struct Weighted {
    idxs: Vec<usize>,
    cum: Vec<f32>,
    total: f32,
}
impl Weighted {
    pub fn expr(samples: &[Sample], subset: &[usize], active_boost: f32) -> Self {
        let mut cum = Vec::with_capacity(subset.len());
        let mut acc = 0.0f32;
        for &i in subset {
            let m = samples[i].label[2..6].iter().cloned().fold(0.0f32, f32::max);
            acc += 1.0 + active_boost * m;
            cum.push(acc);
        }
        Self { idxs: subset.to_vec(), cum, total: acc }
    }
    pub fn sample(&self, rng: &mut Rng) -> usize {
        let u = rng.f01() * self.total;
        let p = self.cum.partition_point(|&c| c < u).min(self.cum.len() - 1);
        self.idxs[p]
    }
}

// gaze 3-way stratified: closed (lid>0.5, not gaze) / expr-gaze / neutral-gaze, budget 0.20 closed.
pub fn gaze_weights(samples: &[Sample], subset: &[usize]) -> Weighted {
    let (closed_ratio, expr_ratio, lid_thresh, active_thresh) = (0.20f32, 0.6f32, 0.5f32, 0.5f32);
    let mut closed = 0usize;
    let mut eg = 0usize;
    let mut ng = 0usize;
    let bucket = |s: &Sample| -> u8 {
        let gaze = s.gaze_usable; // gaze_valid AND not a stuttered/stale-gaze frame
        let active = s.label[2..6].iter().cloned().fold(0.0, f32::max) > active_thresh;
        if gaze {
            if active || s.expr_unlabeled > 0.5 { 1 } else { 2 } // expr-gaze / neut-gaze
        } else if s.label[2] > lid_thresh {
            0 // eyes closed = EyeLid (=1-lid = label[2]) high, not gaze
        } else {
            3
        }
    };
    for &i in subset {
        match bucket(&samples[i]) {
            0 => closed += 1,
            1 => eg += 1,
            2 => ng += 1,
            _ => {}
        }
    }
    let gaze_budget = 1.0 - closed_ratio;
    let e_share = if eg > 0 { expr_ratio * gaze_budget } else { 0.0 };
    let n_share = if ng > 0 { (1.0 - expr_ratio) * gaze_budget } else { 0.0 };
    let scale = if e_share + n_share > 0.0 { gaze_budget / (e_share + n_share) } else { 0.0 };
    let (e_share, n_share) = (e_share * scale, n_share * scale);
    let mut cum = Vec::with_capacity(subset.len());
    let mut acc = 0.0f32;
    for &i in subset {
        let w = match bucket(&samples[i]) {
            0 if closed > 0 => closed_ratio / closed as f32,
            1 if eg > 0 => e_share / eg as f32,
            2 if ng > 0 => n_share / ng as f32,
            _ => 0.0,
        };
        acc += w;
        cum.push(acc);
    }
    if acc == 0.0 {
        return Weighted::expr(samples, subset, 0.0);
    }
    Weighted { idxs: subset.to_vec(), cum, total: acc }
}
