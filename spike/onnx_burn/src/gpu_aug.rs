// GPU-side train augmentation. The whole decoded frame store lives on the device once (f16 rows
// holding RAW 0..255 values — exact for u8; /255 happens in f32 on device), prefetch workers only
// SAMPLE indices + aug parameters (AugSpec), and the mega-batch is built on-device: frame gather ->
// bilinear affine warp -> optional mixup blend -> photometric -> separable gaussian blur.
//
// Parity contract with aug.rs (the CPU reference): every spec constructor consumes the Rng in the
// EXACT same order as its CPU counterpart, so the same seed produces the same parameters, and each
// GPU op mirrors the CPU math (clamped-coord bilinear = BORDER_REPLICATE, truncated+renormalized
// gaussian taps, clamp->pow photometric). Verified by the tests at the bottom (CPU vs GPU, same seed).
use burn::prelude::*;
use burn::tensor::{FloatDType, Distribution};

use crate::aug::{sample_geom_weak, Geom};
use crate::data::{S, T};
use crate::sampler::Rng;

const K_BLUR: usize = 19; // fixed tap count; radius 9 covers the gaze aug's max sigma (~2.9)

fn uni(rng: &mut Rng, a: f32, b: f32) -> f32 {
    a + rng.f01() * (b - a)
}

fn sample_geom_strong(rng: &mut Rng) -> Geom {
    Geom { angle: uni(rng, -10.0, 10.0), tx: uni(rng, -0.08, 0.08), ty: uni(rng, -0.08, 0.08), scale: uni(rng, 0.85, 1.15) }
}

// Inverse of the OpenCV-style forward affine used by aug::apply_geom (same matrix construction).
pub fn inv_affine(g: &Geom) -> [f32; 6] {
    let (cx, cy) = (S as f32 / 2.0, S as f32 / 2.0);
    let rad = g.angle.to_radians();
    let (a, b) = (g.scale * rad.cos(), g.scale * rad.sin());
    let (m00, m01, m02) = (a, b, (1.0 - a) * cx - b * cy + g.tx * S as f32);
    let (m10, m11, m12) = (-b, a, b * cx + (1.0 - a) * cy + g.ty * S as f32);
    let det = m00 * m11 - m01 * m10;
    let (i00, i01, i10, i11) = (m11 / det, -m01 / det, -m10 / det, m00 / det);
    let i02 = -(i00 * m02 + i01 * m12);
    let i12 = -(i10 * m02 + i11 * m12);
    [i00, i01, i02, i10, i11, i12]
}

const IDENT: [f32; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

// One mega-batch row: temporal frame ids for source A (and B when mixing), inverse affines, and
// photometric/blur params. Frame ids are +1-shifted; 0 = the zeros row (missing/blank frame).
#[derive(Clone)]
pub struct AugSpec {
    pub ids_a: [u32; T],
    pub ids_b: [u32; T],
    pub inv_a: [f32; 6],
    pub inv_b: [f32; 6],
    pub lam: f32,
    pub bright: f32,
    pub contrast: f32,
    pub gamma: f32,
    pub sigma: f32, // <= 0.3 means no blur (delta kernel)
}

impl AugSpec {
    fn base(ids: [u32; T]) -> Self {
        Self {
            ids_a: ids, ids_b: ids, inv_a: IDENT, inv_b: IDENT,
            lam: 0.0, bright: 0.0, contrast: 1.0, gamma: 1.0, sigma: 0.0,
        }
    }

    // aug_stack_expr: strong geom + bright/contrast/gamma + optional blur. Rng order matches.
    pub fn strong(ids: [u32; T], rng: &mut Rng) -> Self {
        let g = sample_geom_strong(rng);
        let bright = uni(rng, -0.30, 0.30);
        let contrast = uni(rng, 0.7, 1.3);
        let gamma = uni(rng, 0.6, 1.6);
        let sigma = uni(rng, 0.0, 1.5);
        let mut s = Self::base(ids);
        s.inv_a = inv_affine(&g);
        s.inv_b = s.inv_a;
        s.bright = bright;
        s.contrast = contrast;
        s.gamma = gamma;
        s.sigma = if sigma > 0.3 { sigma } else { 0.0 };
        s
    }

    // aug_stack_weak with a GIVEN shared geom: weak brightness only. Rng order matches.
    pub fn weak(ids: [u32; T], g: &Geom, rng: &mut Rng) -> Self {
        let bright = uni(rng, -0.10, 0.10);
        let mut s = Self::base(ids);
        s.inv_a = inv_affine(g);
        s.inv_b = s.inv_a;
        s.bright = bright;
        s
    }

    // mixup_stacks: independent strong geoms, lam blend, then photometric+blur. Rng order matches.
    pub fn mixup(ids_a: [u32; T], ids_b: [u32; T], lam: f32, rng: &mut Rng) -> Self {
        let ga = sample_geom_strong(rng);
        let gb = sample_geom_strong(rng);
        let bright = uni(rng, -0.30, 0.30);
        let contrast = uni(rng, 0.7, 1.3);
        let gamma = uni(rng, 0.6, 1.6);
        let sigma = uni(rng, 0.0, 1.5);
        let mut s = Self::base(ids_a);
        s.ids_b = ids_b;
        s.inv_a = inv_affine(&ga);
        s.inv_b = inv_affine(&gb);
        s.lam = lam;
        s.bright = bright;
        s.contrast = contrast;
        s.gamma = gamma;
        s.sigma = if sigma > 0.3 { sigma } else { 0.0 };
        s
    }

    // aug_stack_gaze: probabilistic spatial/intensity/blur. Rng order matches.
    pub fn gaze(ids: [u32; T], rng: &mut Rng) -> Self {
        let do_spatial = rng.f01() < 0.3;
        let do_intensity = rng.f01() < 0.4;
        let do_blur = rng.f01() < 0.3;
        let g = if do_spatial {
            Some(Geom { angle: uni(rng, -12.0, 12.0), tx: uni(rng, -0.172, 0.172), ty: uni(rng, -0.172, 0.172), scale: uni(rng, 0.9, 1.1) })
        } else {
            None
        };
        let (bright, contrast) = (uni(rng, -0.2, 0.2), uni(rng, 0.4, 1.6));
        let sigma = if do_blur {
            let ks = [3.0f32, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0][rng.usize(7)];
            0.3 * ((ks - 1.0) * 0.5 - 1.0) + 0.8
        } else {
            0.0
        };
        let mut s = Self::base(ids);
        if let Some(g) = &g {
            s.inv_a = inv_affine(g);
            s.inv_b = s.inv_a;
        }
        if do_intensity {
            s.bright = bright;
            s.contrast = contrast;
        }
        s.sigma = sigma;
        s
    }

    // No-op spec (identity warp, no photometric, no blur) — bit-equal to the raw stack.
    pub fn plain(ids: [u32; T]) -> Self {
        Self::base(ids)
    }
}

// Device-resident frame store: [n_frames+1, S*S] f16 rows of RAW 0..255 values; row 0 = zeros.
pub struct GpuFrames<B: Backend> {
    tensor: Tensor<B, 2>,
}

impl<B: Backend> GpuFrames<B> {
    // data[k] = processed 128x128 u8 frame (or None). Uploaded once, in chunks (bounded host RAM).
    pub fn upload(data: &[Option<Vec<u8>>], device: &B::Device) -> Self {
        let n = data.len();
        let mut rows: Vec<Tensor<B, 2>> = Vec::new();
        let chunk = 4096;
        rows.push(Tensor::<B, 2>::zeros([1, S * S], device).cast(FloatDType::F16));
        let mut buf: Vec<f32> = Vec::with_capacity(chunk * S * S);
        let mut in_buf = 0usize;
        for d in data.iter() {
            match d {
                Some(px) => buf.extend(px.iter().map(|&v| v as f32)),
                None => buf.extend(std::iter::repeat(0.0f32).take(S * S)),
            }
            in_buf += 1;
            if in_buf == chunk {
                let t = Tensor::<B, 2>::from_data(TensorData::new(std::mem::take(&mut buf), [in_buf, S * S]), device);
                rows.push(t.cast(FloatDType::F16));
                in_buf = 0;
            }
        }
        if in_buf > 0 {
            let t = Tensor::<B, 2>::from_data(TensorData::new(buf, [in_buf, S * S]), device);
            rows.push(t.cast(FloatDType::F16));
        }
        let tensor = Tensor::cat(rows, 0);
        assert_eq!(tensor.dims(), [n + 1, S * S]);
        Self { tensor }
    }

    // Stack id for (rec, side) given the Frames layout [rec*2+side], +1-shifted; 0 if missing.
    pub fn frame_id(present: impl Fn(usize, usize) -> bool, rec: usize, side: usize) -> u32 {
        if present(rec, side) { (rec * 2 + side + 1) as u32 } else { 0 }
    }
}

// Build the augmented f32 mega-batch [R, T, S, S] on-device from the specs.
pub fn augment<B: Backend>(frames: &GpuFrames<B>, specs: &[AugSpec], device: &B::Device) -> Tensor<B, 4> {
    augment_stage(frames, specs, device, 5)
}

// stage: 1=src_a 2=warp(a) 3=mixed 4=photometric 5=+blur (full). For bisecting parity failures.
pub fn augment_stage<B: Backend>(frames: &GpuFrames<B>, specs: &[AugSpec], device: &B::Device, stage: u8) -> Tensor<B, 4> {
    let r = specs.len();
    let sf = S as f32;

    // Param tensors (tiny uploads).
    let mut ids: Vec<i32> = Vec::with_capacity(r * 2 * T);
    let mut fpar: Vec<f32> = Vec::with_capacity(r * 17);
    for s in specs {
        ids.extend(s.ids_a.iter().map(|&v| v as i32));
        ids.extend(s.ids_b.iter().map(|&v| v as i32));
        fpar.extend_from_slice(&s.inv_a);
        fpar.extend_from_slice(&s.inv_b);
        fpar.push(s.lam);
        fpar.push(s.bright);
        fpar.push(s.contrast);
        fpar.push(s.gamma);
        fpar.push(s.sigma);
    }
    let ids = Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [r, 2 * T]), device);
    let fpar = Tensor::<B, 2>::from_data(TensorData::new(fpar, [r, 17]), device);

    // Gather raw stacks (f16 rows -> f32), sources A and B.
    let gather_src = |ids: Tensor<B, 1, Int>| -> Tensor<B, 4> {
        frames
            .tensor
            .clone()
            .select(0, ids)
            .cast(FloatDType::F32)
            .reshape([r, T, S, S])
            .div_scalar(255.0)
    };
    let src_a = gather_src(ids.clone().slice([0..r, 0..T]).reshape([r * T]));
    if stage == 1 {
        return src_a;
    }
    let src_b = gather_src(ids.slice([0..r, T..2 * T]).reshape([r * T]));

    // Bilinear warp with clamped source coords (= BORDER_REPLICATE), geom shared across T.
    // NOTE: elementwise ops where BOTH operands are stride-0 broadcast/expanded views produce
    // garbage for rows >= 1 on wgpu+fusion (one expanded operand against a contiguous one is
    // fine) — bisected in the tests below. So every per-row map here is built with flat
    // contiguous arange arithmetic + gathers; expands only ever meet contiguous tensors.
    let n3 = r * S * S;
    let pos3 = Tensor::<B, 1, Int>::arange(0..n3 as i64, device);
    let row3 = pos3.clone().div_scalar((S * S) as i64); // spec row per element [n3]
    let spat3 = pos3.clone().sub(row3.clone().mul_scalar((S * S) as i64));
    let yrow = spat3.clone().div_scalar(S as i64);
    let xv = spat3.sub(yrow.clone().mul_scalar(S as i64)).float(); // x coord [n3]
    let yv = yrow.float(); // y coord [n3]
    let col = |i: usize| {
        fpar.clone()
            .slice([0..r, i..i + 1])
            .reshape([r])
            .gather(0, row3.clone()) // per-element row param, contiguous [n3]
    };
    let warp = |src: Tensor<B, 4>, base: usize| -> Tensor<B, 4> {
        let fx = (col(base) * xv.clone() + col(base + 1) * yv.clone() + col(base + 2)).clamp(0.0, sf - 1.0);
        let fy = (col(base + 3) * xv.clone() + col(base + 4) * yv.clone() + col(base + 5)).clamp(0.0, sf - 1.0);
        let x0f = fx.clone().floor();
        let y0f = fy.clone().floor();
        let wx = (fx - x0f.clone()).reshape([r, 1, S, S]).expand([r, T, S, S]);
        let wy = (fy - y0f.clone()).reshape([r, 1, S, S]).expand([r, T, S, S]);
        let x0 = x0f.int(); // [n3]
        let y0 = y0f.int();
        let x1 = x0.clone().add_scalar(1).clamp(0, S as i64 - 1);
        let y1 = y0.clone().add_scalar(1).clamp(0, S as i64 - 1);
        // flat index into [R*T*S*S], built with ZERO broadcasts: pos -> (rt, spatial) via flat
        // arange arithmetic; the per-row spatial map (y0*S+x0 etc, [r*S*S]) is looked up per
        // element with a gather. (Broadcast/expand-built Int index tensors scramble rows >= 1 on
        // wgpu+fusion; float broadcasts are fine — see the probes in tests.)
        let n_all = r * T * S * S;
        let pos = Tensor::<B, 1, Int>::arange(0..n_all as i64, device);
        let rt = pos.clone().div_scalar((S * S) as i64); // rt-plane id = flat row into [r*T]
        let spat = pos.clone().sub(rt.clone().mul_scalar((S * S) as i64)); // y*S + x within plane
        let t_of = rt.clone().sub(rt.clone().div_scalar(T as i64).mul_scalar(T as i64));
        let sp_row = rt.clone().sub(t_of).div_scalar(T as i64); // spec row r_i, [N]
        // per-element index into the [r*S*S] spatial map: r_i*(S*S) + spat
        let map_idx = sp_row.mul_scalar((S * S) as i64).add(spat);
        let plane_base = rt.mul_scalar((S * S) as i64);
        let flat = src.reshape([r * T * S * S]);
        let tap = |yi: &Tensor<B, 1, Int>, xi: &Tensor<B, 1, Int>| -> Tensor<B, 4> {
            let sp_map = yi.clone().mul_scalar(S as i64).add(xi.clone()); // [r*S*S]
            let sp_all = sp_map.gather(0, map_idx.clone()); // [N] spatial offset per element
            let idx = plane_base.clone().add(sp_all);
            flat.clone().gather(0, idx).reshape([r, T, S, S])
        };
        let p00 = tap(&y0, &x0);
        let p01 = tap(&y0, &x1);
        let p10 = tap(&y1, &x0);
        let p11 = tap(&y1, &x1);
        let top = p00.clone() + (p01 - p00) * wx.clone();
        let bot = p10.clone() + (p11 - p10) * wx;
        top.clone() + (bot - top) * wy
    };
    let wa = warp(src_a, 0);
    if stage == 2 {
        return wa;
    }
    let wb = warp(src_b, 6);

    // Mixup blend: out = (1-lam)*a + lam*b (lam=0 for non-mix rows -> exactly a).
    let per_row = |i: usize| fpar.clone().slice([0..r, i..i + 1]).reshape([r, 1, 1, 1]).expand([r, T, S, S]);
    let lam = per_row(12);
    let mixed = wa.clone() + (wb - wa) * lam;
    if stage == 3 {
        return mixed;
    }

    // Photometric: clamp(p*contrast + bright, 0, 1) ^ gamma (gamma==1 rows bypass pow exactly).
    let bright = per_row(13);
    let contrast = per_row(14);
    let gamma = fpar.clone().slice([0..r, 15..16]).reshape([r, 1, 1, 1]);
    let lin = (mixed * contrast + bright).clamp(0.0, 1.0);
    let g_is1 = gamma.clone().equal_elem(1.0);
    let powed = lin.clone().powf(gamma.expand([r, T, S, S]));
    let photo = powed.mask_where(g_is1.expand([r, T, S, S]), lin);
    if stage == 4 {
        return photo;
    }

    // Separable gaussian blur, per-row sigma; taps outside ceil(3*sigma) are excluded then the
    // kernel renormalized (matches the CPU truncation); sigma<=0.3 rows get a delta kernel (no-op).
    let radius = (K_BLUR / 2) as f32;
    let sigma = fpar.slice([0..r, 16..17]); // [r,1]
    // flat contiguous build (see stride-0 note above): [r*K] elements, per-element sigma gather.
    let nk = r * K_BLUR;
    let posk = Tensor::<B, 1, Int>::arange(0..nk as i64, device);
    let rowk = posk.clone().div_scalar(K_BLUR as i64);
    let jv = posk.sub(rowk.clone().mul_scalar(K_BLUR as i64)).float(); // tap index [r*K]
    let sigv = sigma.reshape([r]).gather(0, rowk); // per-element sigma [r*K]
    let d = jv.clone().sub_scalar(radius);
    let sig_safe = sigv.clone().clamp_min(1e-3);
    let w = (d.clone().powf_scalar(2.0).neg() / (sig_safe.clone() * sig_safe * 2.0)).exp();
    let rmask = d.abs().lower_equal(sigv.clone().mul_scalar(3.0).ceil()).float();
    let active = sigv.greater_elem(0.3).float(); // [r*K]
    let delta = jv.equal_elem((K_BLUR / 2) as f32).float();
    let w = (w * rmask).reshape([r, K_BLUR]);
    let w = (w.clone() / w.sum_dim(1).expand([r, K_BLUR])).reshape([nk]);
    let w = (w * active.clone() + delta * (active.neg().add_scalar(1.0))).reshape([r, K_BLUR]);

    // Replicate-pad + two grouped conv2d passes (groups = R*T, weight [R*T,1,k,1]/[R*T,1,1,k]).
    let rt = r * T;
    let x = photo.reshape([1, rt, S, S]);
    let wk = w.reshape([r, 1, K_BLUR]).repeat_dim(1, T).reshape([rt, 1, 1, K_BLUR]);
    let pad_n = K_BLUR / 2;
    // horizontal: replicate-pad W then conv [1,1,1,K]
    let first = x.clone().slice([0..1, 0..rt, 0..S, 0..1]).repeat_dim(3, pad_n);
    let last = x.clone().slice([0..1, 0..rt, 0..S, S - 1..S]).repeat_dim(3, pad_n);
    let xp = Tensor::cat(vec![first, x, last], 3);
    let opts = burn::tensor::ops::ConvOptions::new([1, 1], [0, 0], [1, 1], rt);
    let h = burn::tensor::module::conv2d(xp, wk.clone(), None, opts.clone());
    // vertical: replicate-pad H then conv [1,1,K,1]
    let first = h.clone().slice([0..1, 0..rt, 0..1, 0..S]).repeat_dim(2, pad_n);
    let last = h.clone().slice([0..1, 0..rt, S - 1..S, 0..S]).repeat_dim(2, pad_n);
    let hp = Tensor::cat(vec![first, h, last], 2);
    let wkv = wk.reshape([rt, 1, K_BLUR, 1]);
    let v = burn::tensor::module::conv2d(hp, wkv, None, opts);
    v.reshape([r, T, S, S])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aug;
    use burn::backend::{wgpu::WgpuDevice, Wgpu};

    // Fake frame store: n random frames, some missing.
    fn fake_frames(n: usize, rng: &mut Rng) -> Vec<Option<Vec<u8>>> {
        (0..n)
            .map(|k| {
                if k % 7 == 3 {
                    None
                } else {
                    Some((0..S * S).map(|_| (rng.f01() * 255.0) as u8).collect())
                }
            })
            .collect()
    }

    fn cpu_stack(data: &[Option<Vec<u8>>], ids: &[u32; T]) -> Vec<f32> {
        let mut out = Vec::with_capacity(T * S * S);
        for &id in ids {
            match id {
                0 => out.extend(std::iter::repeat(0.0f32).take(S * S)),
                v => out.extend(data[(v - 1) as usize].as_ref().unwrap().iter().map(|&p| p as f32 / 255.0)),
            }
        }
        out
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn gpu_aug_matches_cpu() {
        type B = Wgpu;
        let device = WgpuDevice::default();
        let mut rng = Rng::new(99);
        let data = fake_frames(64, &mut rng);
        let frames = GpuFrames::<B>::upload(&data, &device);
        let id_of = |k: usize| -> u32 { if data[k].is_some() { (k + 1) as u32 } else { 0 } };
        let ids1: [u32; T] = [id_of(10), id_of(9), id_of(8), id_of(7)];
        let ids2: [u32; T] = [id_of(20), id_of(19), id_of(18), id_of(17)];
        let ids3: [u32; T] = [id_of(31), id_of(30), id_of(29), id_of(28)];

        let mut scratch = Vec::new();

        // CPU references, sampling with seed X; GPU specs sampling with the SAME seed.
        let mut c = Rng::new(4242);
        let mut g = Rng::new(4242);

        // strong
        let mut cpu1 = cpu_stack(&data, &ids1);
        aug::aug_stack_expr(&mut cpu1, &mut c, &mut scratch);
        let s1 = AugSpec::strong(ids1, &mut g);
        // weak (shared geom)
        let geom_c = aug::sample_geom_weak(&mut c);
        let mut cpu2 = cpu_stack(&data, &ids2);
        aug::aug_stack_weak(&mut cpu2, &geom_c, &mut c);
        let geom_g = sample_geom_weak(&mut g);
        let s2 = AugSpec::weak(ids2, &geom_g, &mut g);
        // mixup
        let (fa, fb) = (cpu_stack(&data, &ids1), cpu_stack(&data, &ids3));
        let lam = 0.37f32;
        let cpu3 = aug::mixup_stacks(&fa, &fb, lam, &mut c, &mut scratch);
        let s3 = AugSpec::mixup(ids1, ids3, lam, &mut g);
        // gaze
        let mut cpu4 = cpu_stack(&data, &ids2);
        aug::aug_stack_gaze(&mut cpu4, &mut c, &mut scratch);
        let s4 = AugSpec::gaze(ids2, &mut g);
        // plain
        let cpu5 = cpu_stack(&data, &ids3);
        let s5 = AugSpec::plain(ids3);

        let mut scratch = Vec::new();

        // CPU references, sampling with seed X; GPU specs sampling with the SAME seed.
        let mut c = Rng::new(4242);
        let mut g = Rng::new(4242);

        // strong
        let mut cpu1 = cpu_stack(&data, &ids1);
        aug::aug_stack_expr(&mut cpu1, &mut c, &mut scratch);
        let s1 = AugSpec::strong(ids1, &mut g);
        // weak (shared geom)
        let geom_c = aug::sample_geom_weak(&mut c);
        let mut cpu2 = cpu_stack(&data, &ids2);
        aug::aug_stack_weak(&mut cpu2, &geom_c, &mut c);
        let geom_g = sample_geom_weak(&mut g);
        let s2 = AugSpec::weak(ids2, &geom_g, &mut g);
        // mixup
        let (fa, fb) = (cpu_stack(&data, &ids1), cpu_stack(&data, &ids3));
        let lam = 0.37f32;
        let cpu3 = aug::mixup_stacks(&fa, &fb, lam, &mut c, &mut scratch);
        let s3 = AugSpec::mixup(ids1, ids3, lam, &mut g);
        // gaze
        let mut cpu4 = cpu_stack(&data, &ids2);
        aug::aug_stack_gaze(&mut cpu4, &mut c, &mut scratch);
        let s4 = AugSpec::gaze(ids2, &mut g);
        // plain
        let cpu5 = cpu_stack(&data, &ids3);
        let s5 = AugSpec::plain(ids3);

        // stage bisect on the photo probe rows (identity warps, bright 0.0 / 0.2)
        {
            let n = T * S * S;
            let base = cpu_stack(&data, &ids1);
            let mut pa = AugSpec::plain(ids1);
            let mut pb = AugSpec::plain(ids1);
            pa.bright = 0.0;
            pb.bright = 0.2;
            for stage in 1..=4u8 {
                let o = augment_stage(&frames, &[pa.clone(), pb.clone()], &device, stage)
                    .into_data().to_vec::<f32>().unwrap();
                println!("stage {stage}: row0 {:.2e} row1-vs-base {:.2e}", max_diff(&o[0..n], &base), max_diff(&o[n..2*n], &base));
            }
        }
        // select probe: two identical id rows through the frame gather must produce equal halves
        {
            let n = T * S * S;
            let idsl: Vec<i32> = ids1.iter().chain(ids1.iter()).map(|&v| v as i32).collect();
            let idst = Tensor::<Wgpu, 1, Int>::from_data(TensorData::new(idsl, [2 * T]), &device);
            let sel = frames.tensor.clone().select(0, idst).cast(burn::tensor::FloatDType::F32).div_scalar(255.0);
            let vv = sel.into_data().to_vec::<f32>().unwrap();
            let base = cpu_stack(&data, &ids1);
            println!("select probe: half0-vs-cpu {:.2e} half1-vs-cpu {:.2e}", max_diff(&vv[0..n], &base), max_diff(&vv[n..2*n], &base));
        }
        // stage probes: which per-row pathway collapses to row 0?
        {
            let n = T * S * S;
            // (a) photometric only: identity warp, brights 0.0 vs 0.2
            let mut pa = AugSpec::plain(ids1);
            let mut pb = AugSpec::plain(ids1);
            pa.bright = 0.0;
            pb.bright = 0.2;
            let o = augment(&frames, &[pa, pb], &device).into_data().to_vec::<f32>().unwrap();
            let base = cpu_stack(&data, &ids1);
            let d0 = max_diff(&o[0..n], &base);
            let shifted: Vec<f32> = base.iter().map(|&x| (x + 0.2).min(1.0)).collect();
            let d1 = max_diff(&o[n..2 * n], &shifted);
            let d1_base = max_diff(&o[n..2 * n], &base);
            let r1_mean: f32 = o[n..2 * n].iter().sum::<f32>() / n as f32;
            let b_mean: f32 = base.iter().sum::<f32>() / n as f32;
            // is row1 equal to row0's OUTPUT? (content copied across rows)
            let d1_row0 = max_diff(&o[n..2 * n], &o[0..n]);
            println!("photo probe: row0 {d0:.2e} row1-vs-shifted {d1:.2e} row1-vs-base {d1_base:.2e} row1-vs-row0out {d1_row0:.2e} r1_mean {r1_mean:.3} base_mean {b_mean:.3}");
            // (b) warp only: rows with different pure-translation warps, no photo
            let mut wa_ = AugSpec::plain(ids1);
            let mut wb_ = AugSpec::plain(ids1);
            wa_.inv_a = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
            wb_.inv_a = [1.0, 0.0, 5.0, 0.0, 1.0, 0.0]; // shift +5px in x (src coords)
            wa_.inv_b = wa_.inv_a;
            wb_.inv_b = wb_.inv_a;
            let o = augment(&frames, &[wa_, wb_], &device).into_data().to_vec::<f32>().unwrap();
            let d0 = max_diff(&o[0..n], &base);
            // expected row1: src x+5 clamped
            let mut exp1 = vec![0.0f32; n];
            for t in 0..T {
                for y in 0..S {
                    for x in 0..S {
                        exp1[t * S * S + y * S + x] = base[t * S * S + y * S + (x + 5).min(S - 1)];
                    }
                }
            }
            let d1 = max_diff(&o[n..2 * n], &exp1);
            println!("warp probe: row0 {d0:.2e} row1 {d1:.2e}");
        }
        let out = augment(&frames, &[s1, s2, s3, s4, s5], &device);
        let v = out.into_data().to_vec::<f32>().unwrap();
        let n = T * S * S;
        for (i, (name, cpu)) in [("strong", &cpu1), ("weak", &cpu2), ("mixup", &cpu3), ("gaze", &cpu4), ("plain", &cpu5)]
            .iter()
            .enumerate()
        {
            let d = max_diff(&v[i * n..(i + 1) * n], cpu);
            println!("{name}: max diff {d:.2e}");
            assert!(d < 5e-4, "{name} diverged: {d}");
        }
    }
}
