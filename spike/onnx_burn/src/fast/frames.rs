use rayon::prelude::*;

use crate::data::{process_eye, Bin, Sample, S, T};

pub struct Frames {
    data: Vec<Option<Vec<u8>>>, // [rec*2 + side] -> processed 128x128 u8, None if blank
}

impl Frames {
    pub fn decode(bin: &Bin) -> Self {
        let n = bin.frames.len() * 2;
        let data = (0..n).into_par_iter().map(|k| process_eye(bin, k / 2, k % 2)).collect();
        Frames { data }
    }
    pub(crate) fn get(&self, rec: usize, side: usize) -> Option<&[u8]> {
        self.data[rec * 2 + side].as_deref()
    }
    pub(crate) fn stack_into(&self, rec: usize, side: usize, out: &mut Vec<f32>) {
        for k in 0..T {
            let f = if rec >= k { self.get(rec - k, side) } else { None };
            match f {
                Some(px) => out.extend(px.iter().map(|&v| v as f32 / 255.0)),
                None => out.extend(std::iter::repeat(0.0).take(S * S)),
            }
        }
    }
    pub(crate) fn stack_f32(&self, rec: usize, side: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(T * S * S);
        self.stack_into(rec, side, &mut out);
        out
    }
    pub fn raw(&self) -> &[Option<Vec<u8>>] {
        &self.data
    }
    // Temporal-stack frame ids for gpu_aug (+1-shifted, 0 = missing/blank), layout [rec*2+side].
    pub(crate) fn stack_ids(&self, rec: usize, side: usize) -> [u32; T] {
        let mut ids = [0u32; T];
        for (k, id) in ids.iter_mut().enumerate() {
            if rec >= k && self.get(rec - k, side).is_some() {
                *id = ((rec - k) * 2 + side + 1) as u32;
            }
        }
        ids
    }
}

fn raw_gaze(bin: &Bin, rec: usize, side: usize) -> (f32, f32) {
    let f = &bin.frames[rec].fields;
    if side == 0 { (f[5], f[6]) } else { (f[7], f[8]) }
}

pub fn build_samples(bin: &Bin, frames: &Frames) -> Vec<Sample> {
    let mut out = Vec::new();
    for rec in 0..bin.frames.len() {
        let fr = &bin.frames[rec];
        if !fr.good() {
            continue;
        }
        let (l, r) = fr.labels();
        let (gv, eu) = (fr.gaze_valid(), fr.expr_unlabeled());
        for (side, lab) in [(0usize, l), (1usize, r)] {
            if frames.get(rec, side).is_some() {
                let sticky = rec > 0 && raw_gaze(bin, rec - 1, side) == raw_gaze(bin, rec, side);
                let gaze_usable = gv > 0.5 && !sticky;
                out.push(Sample { rec, side, label: lab, gaze_valid: gv, expr_unlabeled: eu, gaze_usable });
            }
        }
    }
    out
}
