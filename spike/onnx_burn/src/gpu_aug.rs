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

// Device-resident frame store: [n_frames+1, S*S/4] i32 rows, 4 u8 pixels packed per word
// (byte k = bits 8k..8k+8), row 0 = zeros (missing-frame sentinel). ~733 MB for a 45k-eye capture.
pub struct GpuFrames<B: Backend> {
    packed: Tensor<B, 2, Int>,
}

/// Backend hook for the fused augmentation kernels (burn-cubecl kernel::aug). Implemented for any
/// fusion-wrapped cubecl backend (wgpu, cuda); the mega-batch never exists on the host.
pub trait AugBackend: Backend + Sized {
    fn aug_upload(data: &[Option<Vec<u8>>], device: &Self::Device) -> GpuFrames<Self>;
    fn aug_batch(frames: &GpuFrames<Self>, specs: &[AugSpec], device: &Self::Device) -> Tensor<Self, 4>;
}

use burn::tensor::{DType, TensorPrimitive};
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime};
use burn_fusion::stream::StreamId;
use burn_fusion::Fusion;

const W_FRAME: usize = S * S / 4;

impl<R: CubeRuntime, BT: BoolElement> AugBackend for Fusion<CubeBackend<R, f32, i32, BT>> {
    fn aug_upload(data: &[Option<Vec<u8>>], device: &Self::Device) -> GpuFrames<Self> {
        let n = data.len();
        let mut rows: Vec<Tensor<Self, 2, Int>> = Vec::new();
        let chunk = 8192;
        let mut buf: Vec<i32> = Vec::with_capacity(chunk * W_FRAME);
        buf.extend(std::iter::repeat(0i32).take(W_FRAME)); // sentinel zeros row
        let mut in_buf = 1usize;
        let flush = |buf: &mut Vec<i32>, in_buf: &mut usize, rows: &mut Vec<Tensor<Self, 2, Int>>| {
            if *in_buf > 0 {
                let t = Tensor::from_data(TensorData::new(std::mem::take(buf), [*in_buf, W_FRAME]), device);
                rows.push(t);
                *in_buf = 0;
            }
        };
        for d in data.iter() {
            match d {
                Some(px) => {
                    for w in px.chunks_exact(4) {
                        buf.push(i32::from_le_bytes([w[0], w[1], w[2], w[3]]));
                    }
                }
                None => buf.extend(std::iter::repeat(0i32).take(W_FRAME)),
            }
            in_buf += 1;
            if in_buf == chunk {
                flush(&mut buf, &mut in_buf, &mut rows);
            }
        }
        flush(&mut buf, &mut in_buf, &mut rows);
        let packed = Tensor::cat(rows, 0);
        assert_eq!(packed.dims(), [n + 1, W_FRAME]);
        GpuFrames { packed }
    }

    fn aug_batch(frames: &GpuFrames<Self>, specs: &[AugSpec], device: &Self::Device) -> Tensor<Self, 4> {
        type CB<R2, BT2> = CubeBackend<R2, f32, i32, BT2>;
        let r = specs.len();
        let mut ids: Vec<i32> = Vec::with_capacity(r * 2 * T);
        let mut fpar: Vec<f32> = Vec::with_capacity(r * 17);
        for sp in specs {
            ids.extend(sp.ids_a.iter().map(|&v| v as i32));
            ids.extend(sp.ids_b.iter().map(|&v| v as i32));
            fpar.extend_from_slice(&sp.inv_a);
            fpar.extend_from_slice(&sp.inv_b);
            fpar.push(sp.lam);
            fpar.push(sp.bright);
            fpar.push(sp.contrast);
            fpar.push(sp.gamma);
            fpar.push(sp.sigma);
        }
        let ids_t = Tensor::<Self, 2, Int>::from_data(TensorData::new(ids, [r, 2 * T]), device);
        let fpar_t = Tensor::<Self, 2>::from_data(TensorData::new(fpar, [r, 17]), device);

        let f_prim = frames.packed.clone().into_primitive();
        let client = f_prim.client.clone();
        let f_cube = client.resolve_tensor_int::<CB<R, BT>>(f_prim);
        let ids_cube = client.resolve_tensor_int::<CB<R, BT>>(ids_t.into_primitive());
        let fpar_cube = match fpar_t.into_primitive() {
            TensorPrimitive::Float(t) => client.resolve_tensor_float::<CB<R, BT>>(t),
            _ => unreachable!(),
        };

        let out = burn_cubecl::kernel::aug::fused_aug::<R>(f_cube, ids_cube, fpar_cube, r, T, S);
        let handle: burn_cubecl_fusion::CubeFusionHandle<R> = out.into();
        let id = client.register_tensor_handle(handle);
        let ft = burn_fusion::FusionTensor::new(id, burn::prelude::Shape::new([r, T, S, S]), DType::F32, client, StreamId::current());
        Tensor::from_primitive(TensorPrimitive::Float(ft))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aug;
    use burn::backend::Wgpu;

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

    // Five pipelines (strong/weak/mixup/gaze/plain), CPU aug.rs vs the fused GPU kernels, with the
    // SAME rng seed on both sides (spec constructors consume the rng in the same order).
    #[test]
    fn gpu_aug_matches_cpu() {
        type B = Wgpu;
        let device = Default::default();
        let mut rng = Rng::new(99);
        let data = fake_frames(64, &mut rng);
        let frames = <B as AugBackend>::aug_upload(&data, &device);
        let id_of = |k: usize| -> u32 { if data[k].is_some() { (k + 1) as u32 } else { 0 } };
        let ids1: [u32; T] = [id_of(10), id_of(9), id_of(8), id_of(7)];
        let ids2: [u32; T] = [id_of(20), id_of(19), id_of(18), id_of(17)];
        let ids3: [u32; T] = [id_of(31), id_of(30), id_of(29), id_of(28)];

        let mut scratch = Vec::new();
        let mut c = Rng::new(4242);
        let mut g = Rng::new(4242);

        let mut cpu1 = cpu_stack(&data, &ids1);
        aug::aug_stack_expr(&mut cpu1, &mut c, &mut scratch);
        let s1 = AugSpec::strong(ids1, &mut g);

        let geom_c = aug::sample_geom_weak(&mut c);
        let mut cpu2 = cpu_stack(&data, &ids2);
        aug::aug_stack_weak(&mut cpu2, &geom_c, &mut c);
        let geom_g = sample_geom_weak(&mut g);
        let s2 = AugSpec::weak(ids2, &geom_g, &mut g);

        let (fa, fb) = (cpu_stack(&data, &ids1), cpu_stack(&data, &ids3));
        let lam = 0.37f32;
        let cpu3 = aug::mixup_stacks(&fa, &fb, lam, &mut c, &mut scratch);
        let s3 = AugSpec::mixup(ids1, ids3, lam, &mut g);

        let mut cpu4 = cpu_stack(&data, &ids2);
        aug::aug_stack_gaze(&mut cpu4, &mut c, &mut scratch);
        let s4 = AugSpec::gaze(ids2, &mut g);

        let cpu5 = cpu_stack(&data, &ids3);
        let s5 = AugSpec::plain(ids3);

        let out = <B as AugBackend>::aug_batch(&frames, &[s1, s2, s3, s4, s5], &device);
        let v = out.into_data().to_vec::<f32>().unwrap();
        let n = T * S * S;
        for (i, (name, cpu)) in [("strong", &cpu1), ("weak", &cpu2), ("mixup", &cpu3), ("gaze", &cpu4), ("plain", &cpu5)]
            .iter()
            .enumerate()
        {
            let d = max_diff(&v[i * n..(i + 1) * n], cpu);
            println!("{name}: max diff {d:.2e}");
        }
        let mut worst = 0.0f32;
        for (i, (_, cpu)) in [("strong", &cpu1), ("weak", &cpu2), ("mixup", &cpu3), ("gaze", &cpu4), ("plain", &cpu5)].iter().enumerate() {
            worst = worst.max(max_diff(&v[i * n..(i + 1) * n], cpu));
        }
        assert!(worst < 5e-4, "diverged: {worst}");
    }
}
