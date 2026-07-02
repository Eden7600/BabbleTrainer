use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorData};

mod model {
    pub mod backbone {
        include!(concat!(env!("OUT_DIR"), "/model/backbone.rs"));
    }
}
use model::backbone::Model;

type B = Wgpu;

fn read_npy_f32(path: &str) -> (Vec<f32>, Vec<usize>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
    let shape: Vec<usize> = npy.shape().iter().map(|&d| d as usize).collect();
    let data: Vec<f32> = npy.into_vec().unwrap();
    (data, shape)
}

fn main() {
    let device = Default::default();
    println!("backend: wgpu (DX12/Vulkan)");
    let model: Model<B> = Model::default();

    let (xd, xs) = read_npy_f32("../input_ref.npy");
    let input = Tensor::<B, 4>::from_data(TensorData::new(xd, xs), &device);
    let out = model.forward(input);
    let out_data: Vec<f32> = out.to_data().to_vec().unwrap();

    let (ref_d, _rs) = read_npy_f32("../ort_out.npy");
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let max_abs = out_data.iter().zip(&ref_d).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    let max_sig = out_data.iter().zip(&ref_d)
        .map(|(a, b)| (sig(*a) - sig(*b)).abs()).fold(0.0, f32::max);
    let finite = out_data.iter().all(|x| x.is_finite());
    println!("wgpu out: {out_data:?}");
    println!("ort  out: {ref_d:?}");
    println!("max|wgpu - ort|   = {max_abs:.3e}  (logits)");
    println!("max|sigmoid diff| = {max_sig:.3e}  (deployed-space)");
    println!("SMOKE: {}", if finite && max_abs < 2e-3 && max_sig < 2e-4 { "PASS" } else { "FAIL" });
}
