use std::collections::HashSet;
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::Arc;
use std::thread;

use crate::aug;
use crate::data::{Sample, S, T};
use crate::gpu_aug::AugSpec;
use crate::sampler::{Rng, Weighted};

use super::frames::Frames;
use super::index::{Paired, Pools, Unlabeled};

pub(crate) type UnlabBatch = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>); // cur_strong, cur_weak, nbr_weak, pair_ok

pub(crate) fn spawn_unlabeled_prefetch(
    frames: Arc<Frames>, unlab: Arc<Unlabeled>, present: Arc<HashSet<(usize, usize)>>, ub: usize,
    workers: usize, seed: u64,
) -> Receiver<UnlabBatch> {
    let (tx, rx) = sync_channel::<UnlabBatch>(workers * 2);
    for wi in 0..workers.max(1) {
        let (frames, unlab, present, tx) = (frames.clone(), unlab.clone(), present.clone(), tx.clone());
        thread::spawn(move || {
            let mut rng = Rng::new(seed.wrapping_add((wi as u64).wrapping_mul(0x85EBCA77)));
            let mut scratch = Vec::new();
            loop {
                let (mut cs, mut cw, mut nw, mut ok) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for _ in 0..ub {
                    let (rec, side) = unlab.items[rng.usize(unlab.items.len())];
                    let mut a = frames.stack_f32(rec, side);
                    aug::aug_stack_expr(&mut a, &mut rng, &mut scratch); // strong view
                    let g = aug::sample_geom_weak(&mut rng); // shared geom for cur/nbr weak views
                    let mut b = frames.stack_f32(rec, side);
                    aug::aug_stack_weak(&mut b, &g, &mut rng);
                    // Only pair with the previous frame if it is a real GOOD, decoded same-side frame
                    // (present in the sample set). Otherwise the temporal hinge would train against a
                    // blank / non-good neighbor; mask it out (pair_ok=0, self as placeholder).
                    let has_nbr = rec >= 1 && present.contains(&(rec - 1, side));
                    let pair_ok = if has_nbr { 1.0 } else { 0.0 };
                    let mut c = frames.stack_f32(if has_nbr { rec - 1 } else { rec }, side);
                    aug::aug_stack_weak(&mut c, &g, &mut rng);
                    cs.extend_from_slice(&a);
                    cw.extend_from_slice(&b);
                    nw.extend_from_slice(&c);
                    ok.push(pair_ok);
                }
                if tx.send((cs, cw, nw, ok)).is_err() {
                    break;
                }
            }
        });
    }
    rx
}

pub(crate) type PairedBatch = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>); // pl, pr, pll(4), prl(4)

pub(crate) fn spawn_paired_prefetch(
    frames: Arc<Frames>, paired: Arc<Paired>, pbatch: usize, workers: usize, seed: u64,
) -> Receiver<PairedBatch> {
    let (tx, rx) = sync_channel::<PairedBatch>(workers * 2);
    for wi in 0..workers.max(1) {
        let (frames, paired, tx) = (frames.clone(), paired.clone(), tx.clone());
        thread::spawn(move || {
            let mut rng = Rng::new(seed.wrapping_add((wi as u64).wrapping_mul(0x9E3779B1)));
            let mut scratch = Vec::new();
            loop {
                let (mut pl, mut pr, mut pll, mut prl) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for _ in 0..pbatch {
                    let k = paired.sample(&mut rng);
                    let rec = paired.recs[k];
                    let mut a = frames.stack_f32(rec, 0);
                    aug::aug_stack_expr(&mut a, &mut rng, &mut scratch);
                    let mut b = frames.stack_f32(rec, 1);
                    aug::aug_stack_expr(&mut b, &mut rng, &mut scratch);
                    pl.extend_from_slice(&a);
                    pr.extend_from_slice(&b);
                    pll.extend_from_slice(&paired.ll[k]);
                    prl.extend_from_slice(&paired.rl[k]);
                }
                if tx.send((pl, pr, pll, prl)).is_err() {
                    break;
                }
            }
        });
    }
    rx
}

// ---- GPU-aug spec workers: same sampling + SAME rng consumption order as the CPU workers, but
// they emit AugSpec rows (indices + params) instead of augmented pixels. ----

pub(crate) type SpecBatch = (Vec<AugSpec>, Vec<f32>);

pub(crate) fn spawn_prefetch_spec(
    frames: Arc<Frames>, samples: Arc<Vec<Sample>>, w: Arc<Weighted>, batch: usize, workers: usize,
    seed: u64, pools: Arc<Pools>,
) -> Receiver<SpecBatch> {
    let (tx, rx) = sync_channel::<SpecBatch>(workers * 2);
    for wi in 0..workers.max(1) {
        let (frames, samples, w, tx, pools) =
            (frames.clone(), samples.clone(), w.clone(), tx.clone(), pools.clone());
        thread::spawn(move || {
            let mut rng = Rng::new(seed.wrapping_add((wi as u64).wrapping_mul(2654435761)));
            loop {
                let idxs: Vec<usize> = (0..batch).map(|_| w.sample(&mut rng)).collect();
                let mut specs = Vec::with_capacity(batch);
                let mut ys = Vec::with_capacity(batch * 4);
                for &i in &idxs {
                    let smp = &samples[i];
                    let r = rng.f01();
                    let neutral_mix = r < 0.5 && !pools.neutral.is_empty() && !pools.expr.is_empty();
                    let cross_mix = !neutral_mix && r < 0.65 && pools.all.len() > 1;
                    if neutral_mix || cross_mix {
                        let (ia, ib) = if neutral_mix {
                            (pools.neutral[rng.usize(pools.neutral.len())], pools.expr[rng.usize(pools.expr.len())])
                        } else {
                            (pools.all[rng.usize(pools.all.len())], pools.all[rng.usize(pools.all.len())])
                        };
                        let lam = rng.f01();
                        let (sa, sb) = (&samples[ia], &samples[ib]);
                        let ids_a = frames.stack_ids(sa.rec, sa.side);
                        let ids_b = frames.stack_ids(sb.rec, sb.side);
                        specs.push(AugSpec::mixup(ids_a, ids_b, lam, &mut rng));
                        for c in 0..4 {
                            ys.push((1.0 - lam) * sa.label[2 + c] + lam * sb.label[2 + c]);
                        }
                    } else {
                        specs.push(AugSpec::strong(frames.stack_ids(smp.rec, smp.side), &mut rng));
                        ys.extend_from_slice(&smp.label[2..6]);
                    }
                }
                if tx.send((specs, ys)).is_err() {
                    break;
                }
            }
        });
    }
    rx
}

pub(crate) type PairedSpecBatch = (Vec<AugSpec>, Vec<AugSpec>, Vec<f32>, Vec<f32>);

pub(crate) fn spawn_paired_prefetch_spec(
    frames: Arc<Frames>, paired: Arc<Paired>, pbatch: usize, workers: usize, seed: u64,
) -> Receiver<PairedSpecBatch> {
    let (tx, rx) = sync_channel::<PairedSpecBatch>(workers * 2);
    for wi in 0..workers.max(1) {
        let (frames, paired, tx) = (frames.clone(), paired.clone(), tx.clone());
        thread::spawn(move || {
            let mut rng = Rng::new(seed.wrapping_add((wi as u64).wrapping_mul(0x9E3779B1)));
            loop {
                let (mut sl, mut sr, mut pll, mut prl) =
                    (Vec::with_capacity(pbatch), Vec::with_capacity(pbatch), Vec::new(), Vec::new());
                for _ in 0..pbatch {
                    let k = paired.sample(&mut rng);
                    let rec = paired.recs[k];
                    sl.push(AugSpec::strong(frames.stack_ids(rec, 0), &mut rng));
                    sr.push(AugSpec::strong(frames.stack_ids(rec, 1), &mut rng));
                    pll.extend_from_slice(&paired.ll[k]);
                    prl.extend_from_slice(&paired.rl[k]);
                }
                if tx.send((sl, sr, pll, prl)).is_err() {
                    break;
                }
            }
        });
    }
    rx
}

pub(crate) type UnlabSpecBatch = (Vec<AugSpec>, Vec<AugSpec>, Vec<AugSpec>, Vec<f32>);

pub(crate) fn spawn_unlabeled_prefetch_spec(
    frames: Arc<Frames>, unlab: Arc<Unlabeled>, present: Arc<HashSet<(usize, usize)>>, ub: usize,
    workers: usize, seed: u64,
) -> Receiver<UnlabSpecBatch> {
    let (tx, rx) = sync_channel::<UnlabSpecBatch>(workers * 2);
    for wi in 0..workers.max(1) {
        let (frames, unlab, present, tx) = (frames.clone(), unlab.clone(), present.clone(), tx.clone());
        thread::spawn(move || {
            let mut rng = Rng::new(seed.wrapping_add((wi as u64).wrapping_mul(0x85EBCA77)));
            loop {
                let (mut cs, mut cw, mut nw, mut ok) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for _ in 0..ub {
                    let (rec, side) = unlab.items[rng.usize(unlab.items.len())];
                    let ids = frames.stack_ids(rec, side);
                    cs.push(AugSpec::strong(ids, &mut rng));
                    let g = aug::sample_geom_weak(&mut rng);
                    cw.push(AugSpec::weak(ids, &g, &mut rng));
                    let has_nbr = rec >= 1 && present.contains(&(rec - 1, side));
                    let nbr_ids = if has_nbr { frames.stack_ids(rec - 1, side) } else { ids };
                    nw.push(AugSpec::weak(nbr_ids, &g, &mut rng));
                    ok.push(if has_nbr { 1.0 } else { 0.0 });
                }
                if tx.send((cs, cw, nw, ok)).is_err() {
                    break;
                }
            }
        });
    }
    rx
}

pub(crate) type Batch = (Vec<f32>, Vec<f32>);

// Training batch with augmentation. Expr: 50% neutral<->expression mixup, 15% cross-frame blend, else
// strong photometric+geometric aug (the linearity + generalization drivers). Gaze: BabbleTrainer gaze
// aug. Eval uses the plain build_batch (no aug).
#[allow(clippy::too_many_arguments)]
fn build_batch_train(
    frames: &Frames, samples: &[Sample], idxs: &[usize], expr: bool, rng: &mut Rng, pools: &Pools,
    scratch: &mut Vec<f32>,
) -> Batch {
    let b = idxs.len();
    let m = if expr { 4 } else { 2 };
    let mut xs = Vec::with_capacity(b * T * S * S);
    let mut ys = Vec::with_capacity(b * m);
    for &i in idxs {
        let s = &samples[i];
        if !expr {
            let mut stack = frames.stack_f32(s.rec, s.side);
            aug::aug_stack_gaze(&mut stack, rng, scratch);
            xs.extend_from_slice(&stack);
            ys.extend_from_slice(&s.label[0..2]);
            continue;
        }
        let r = rng.f01();
        let neutral_mix = r < 0.5 && !pools.neutral.is_empty() && !pools.expr.is_empty();
        let cross_mix = !neutral_mix && r < 0.65 && pools.all.len() > 1;
        if neutral_mix || cross_mix {
            let (ia, ib) = if neutral_mix {
                (pools.neutral[rng.usize(pools.neutral.len())], pools.expr[rng.usize(pools.expr.len())])
            } else {
                (pools.all[rng.usize(pools.all.len())], pools.all[rng.usize(pools.all.len())])
            };
            let lam = rng.f01();
            let (sa, sb) = (&samples[ia], &samples[ib]);
            let fa = frames.stack_f32(sa.rec, sa.side);
            let fb = frames.stack_f32(sb.rec, sb.side);
            xs.extend_from_slice(&aug::mixup_stacks(&fa, &fb, lam, rng, scratch));
            for c in 0..4 {
                ys.push((1.0 - lam) * sa.label[2 + c] + lam * sb.label[2 + c]);
            }
        } else {
            let mut stack = frames.stack_f32(s.rec, s.side);
            aug::aug_stack_expr(&mut stack, rng, scratch);
            xs.extend_from_slice(&stack);
            ys.extend_from_slice(&s.label[2..6]);
        }
    }
    (xs, ys)
}

pub(crate) fn spawn_prefetch(
    frames: Arc<Frames>, samples: Arc<Vec<Sample>>, w: Arc<Weighted>, batch: usize, workers: usize,
    expr: bool, seed: u64, pools: Arc<Pools>,
) -> Receiver<Batch> {
    let (tx, rx) = sync_channel::<Batch>(workers * 2);
    for wi in 0..workers.max(1) {
        let (frames, samples, w, tx, pools) =
            (frames.clone(), samples.clone(), w.clone(), tx.clone(), pools.clone());
        thread::spawn(move || {
            let mut rng = Rng::new(seed.wrapping_add((wi as u64).wrapping_mul(2654435761)));
            let mut scratch = Vec::new();
            loop {
                let idxs: Vec<usize> = (0..batch).map(|_| w.sample(&mut rng)).collect();
                let b = build_batch_train(&frames, &samples, &idxs, expr, &mut rng, &pools, &mut scratch);
                if tx.send(b).is_err() {
                    break;
                }
            }
        });
    }
    rx
}
