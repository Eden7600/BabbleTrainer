// Train-time augmentation, ported from bin_temporal_dataset.py (_aug_stack / _aug_stack_gaze /
// _mixup_frames / _occlude_lids). Operates on f32 [T,S,S] stacks in [0,1]. Geometric + photometric
// params are sampled ONCE per stack so the T temporal frames stay consistent (a moving eye, not noise).
use crate::data::{S, T};
use crate::sampler::Rng;

fn uni(rng: &mut Rng, a: f32, b: f32) -> f32 {
    a + rng.f01() * (b - a)
}

pub struct Geom {
    pub angle: f32, // degrees, CCW (cv2 convention)
    pub tx: f32,    // fraction of width
    pub ty: f32,
    pub scale: f32,
}

fn sample_geom_strong(rng: &mut Rng) -> Geom {
    Geom { angle: uni(rng, -10.0, 10.0), tx: uni(rng, -0.08, 0.08), ty: uni(rng, -0.08, 0.08), scale: uni(rng, 0.85, 1.15) }
}

// cv2.warpAffine(getRotationMatrix2D(center, angle, scale) + translation), INTER_LINEAR,
// BORDER_REPLICATE. warpAffine maps dst<-src via the INVERSE of the forward matrix, so we invert.
fn apply_geom(src: &[f32], dst: &mut [f32], g: &Geom) {
    let (cx, cy) = (S as f32 / 2.0, S as f32 / 2.0);
    let rad = g.angle.to_radians();
    let (a, b) = (g.scale * rad.cos(), g.scale * rad.sin());
    // forward 2x3 (OpenCV): [[a, b, (1-a)cx - b cy + tx*S], [-b, a, b cx + (1-a)cy + ty*S]]
    let (m00, m01, m02) = (a, b, (1.0 - a) * cx - b * cy + g.tx * S as f32);
    let (m10, m11, m12) = (-b, a, b * cx + (1.0 - a) * cy + g.ty * S as f32);
    let det = m00 * m11 - m01 * m10;
    let (i00, i01, i10, i11) = (m11 / det, -m01 / det, -m10 / det, m00 / det);
    let i02 = -(i00 * m02 + i01 * m12);
    let i12 = -(i10 * m02 + i11 * m12);
    let smax = (S - 1) as f32;
    for y in 0..S {
        for x in 0..S {
            let fx = i00 * x as f32 + i01 * y as f32 + i02;
            let fy = i10 * x as f32 + i11 * y as f32 + i12;
            let cxf = fx.clamp(0.0, smax);
            let cyf = fy.clamp(0.0, smax);
            let (x0, y0) = (cxf.floor() as usize, cyf.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(S - 1), (y0 + 1).min(S - 1));
            let (wx, wy) = (cxf - x0 as f32, cyf - y0 as f32);
            let p00 = src[y0 * S + x0];
            let p01 = src[y0 * S + x1];
            let p10 = src[y1 * S + x0];
            let p11 = src[y1 * S + x1];
            let top = p00 + (p01 - p00) * wx;
            let bot = p10 + (p11 - p10) * wx;
            dst[y * S + x] = top + (bot - top) * wy;
        }
    }
}

// separable Gaussian blur (radius = ceil(3*sigma)), BORDER_REPLICATE. scratch reused by caller.
fn gaussian_blur(buf: &mut [f32], sigma: f32, scratch: &mut [f32]) {
    if sigma <= 1e-3 {
        return;
    }
    let r = (3.0 * sigma).ceil() as isize;
    let mut k = vec![0.0f32; (2 * r + 1) as usize];
    let mut sum = 0.0;
    for (i, kv) in k.iter_mut().enumerate() {
        let d = i as f32 - r as f32;
        *kv = (-(d * d) / (2.0 * sigma * sigma)).exp();
        sum += *kv;
    }
    for kv in k.iter_mut() {
        *kv /= sum;
    }
    let clamp = |v: isize| v.clamp(0, S as isize - 1) as usize;
    // horizontal buf -> scratch
    for y in 0..S {
        for x in 0..S {
            let mut acc = 0.0;
            for (j, &kv) in k.iter().enumerate() {
                let xx = clamp(x as isize + j as isize - r);
                acc += kv * buf[y * S + xx];
            }
            scratch[y * S + x] = acc;
        }
    }
    // vertical scratch -> buf
    for y in 0..S {
        for x in 0..S {
            let mut acc = 0.0;
            for (j, &kv) in k.iter().enumerate() {
                let yy = clamp(y as isize + j as isize - r);
                acc += kv * scratch[yy * S + x];
            }
            buf[y * S + x] = acc;
        }
    }
}

fn photometric(frame: &mut [f32], contrast: f32, bright: f32, gamma: f32) {
    for p in frame.iter_mut() {
        let v = (*p * contrast + bright).clamp(0.0, 1.0);
        *p = v.powf(gamma);
    }
}

// _aug_stack: strong geom (shared) + brightness/contrast/gamma + optional blur, on an expr stack.
pub fn aug_stack_expr(stack: &mut [f32], rng: &mut Rng, scratch: &mut Vec<f32>) {
    let g = sample_geom_strong(rng);
    let bright = uni(rng, -0.30, 0.30);
    let contrast = uni(rng, 0.7, 1.3);
    let gamma = uni(rng, 0.6, 1.6);
    let sigma = uni(rng, 0.0, 1.5);
    scratch.resize(S * S, 0.0);
    let mut tmp = vec![0.0f32; S * S];
    for t in 0..T {
        let fr = &mut stack[t * S * S..(t + 1) * S * S];
        tmp.copy_from_slice(fr);
        apply_geom(&tmp, fr, &g);
        photometric(fr, contrast, bright, gamma);
        if sigma > 0.3 {
            gaussian_blur(fr, sigma, scratch);
        }
    }
}

// _aug_stack_gaze: probabilistic spatial/intensity/blur (BabbleTrainer gaze aug). Occlusion off by
// default (GAZE_OCCLUDE_PROB=0.0).
pub fn aug_stack_gaze(stack: &mut [f32], rng: &mut Rng, scratch: &mut Vec<f32>) {
    let do_spatial = rng.f01() < 0.3;
    let do_intensity = rng.f01() < 0.4;
    let do_blur = rng.f01() < 0.3;
    let g = if do_spatial {
        Some(Geom { angle: uni(rng, -12.0, 12.0), tx: uni(rng, -0.172, 0.172), ty: uni(rng, -0.172, 0.172), scale: uni(rng, 0.9, 1.1) })
    } else {
        None
    };
    let (bright, contrast) = (uni(rng, -0.2, 0.2), uni(rng, 0.4, 1.6));
    // cv2 ksize in {3..15}; approximate sigma = 0.3*((k-1)*0.5 - 1) + 0.8 (cv2 default sigma-from-ksize)
    let sigma = if do_blur {
        let ks = [3.0f32, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0][rng.usize(7)];
        0.3 * ((ks - 1.0) * 0.5 - 1.0) + 0.8
    } else {
        0.0
    };
    scratch.resize(S * S, 0.0);
    let mut tmp = vec![0.0f32; S * S];
    for t in 0..T {
        let fr = &mut stack[t * S * S..(t + 1) * S * S];
        if let Some(g) = &g {
            tmp.copy_from_slice(fr);
            apply_geom(&tmp, fr, g);
        }
        if do_intensity {
            for p in fr.iter_mut() {
                *p = (*p * contrast + bright).clamp(0.0, 1.0);
            }
        }
        if sigma > 0.0 {
            gaussian_blur(fr, sigma, scratch);
        }
    }
}

// _mixup_frames: independent strong geom per source, linear blend, then photometric. Returns the
// blended stack. Labels are blended by the caller.
pub fn mixup_stacks(a: &[f32], b: &[f32], lam: f32, rng: &mut Rng, scratch: &mut Vec<f32>) -> Vec<f32> {
    let ga = sample_geom_strong(rng);
    let gb = sample_geom_strong(rng);
    let bright = uni(rng, -0.30, 0.30);
    let contrast = uni(rng, 0.7, 1.3);
    let gamma = uni(rng, 0.6, 1.6);
    let sigma = uni(rng, 0.0, 1.5);
    scratch.resize(S * S, 0.0);
    let mut out = vec![0.0f32; T * S * S];
    let (mut wa, mut wb) = (vec![0.0f32; S * S], vec![0.0f32; S * S]);
    for t in 0..T {
        let sl = t * S * S..(t + 1) * S * S;
        apply_geom(&a[sl.clone()], &mut wa, &ga);
        apply_geom(&b[sl.clone()], &mut wb, &gb);
        let fr = &mut out[sl];
        for i in 0..S * S {
            fr[i] = (1.0 - lam) * wa[i] + lam * wb[i];
        }
        photometric(fr, contrast, bright, gamma);
        if sigma > 0.3 {
            gaussian_blur(fr, sigma, scratch);
        }
    }
    out
}
