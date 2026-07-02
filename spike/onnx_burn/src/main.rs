use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};

mod model {
    pub mod backbone {
        include!(concat!(env!("OUT_DIR"), "/model/backbone.rs"));
    }
}
use model::backbone::Model;

type B = NdArray<f32>;

fn read_npy_f32(path: &str) -> (Vec<f32>, Vec<usize>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
    let shape: Vec<usize> = npy.shape().iter().map(|&d| d as usize).collect();
    let data: Vec<f32> = npy.into_vec().unwrap();
    (data, shape)
}

fn main() {
    let device = Default::default();
    let model: Model<B> = Model::default();

    let (xd, xs) = read_npy_f32("../input_ref.npy");
    println!("input shape {xs:?}");
    let input = Tensor::<B, 4>::from_data(TensorData::new(xd, xs), &device);

    let out = model.forward(input);
    let out_data: Vec<f32> = out.to_data().to_vec().unwrap();

    let (ref_d, _rs) = read_npy_f32("../ort_out.npy");
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let max_abs = out_data.iter().zip(&ref_d).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    let max_rel = out_data.iter().zip(&ref_d)
        .map(|(a, b)| (a - b).abs() / (b.abs() + 1e-6)).fold(0.0, f32::max);
    let max_sig = out_data.iter().zip(&ref_d)
        .map(|(a, b)| (sig(*a) - sig(*b)).abs()).fold(0.0, f32::max);
    println!("burn out: {out_data:?}");
    println!("ort  out: {ref_d:?}");
    println!("max|burn - ort|      = {max_abs:.3e}  (logits)");
    println!("max rel err          = {max_rel:.3e}");
    println!("max|sigmoid diff|    = {max_sig:.3e}  (deployed-space)");
    // f32 conv accumulation differs across runtimes; 1e-3 atol on logits is the standard bar
    // (burn's own model-check CI), and what actually ships is the sigmoid-space value.
    println!("PARITY: {}", if max_abs < 1e-3 && max_sig < 1e-4 { "PASS" } else { "FAIL" });
}
