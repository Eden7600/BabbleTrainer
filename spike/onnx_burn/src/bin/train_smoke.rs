// End-to-end supervised training smoke on real user_cal.bin (CPU autodiff). Loss must decrease.
use burn::backend::{Autodiff, NdArray};
use onnx_burn::data::{build_samples, parse_bin, EyeCache};
use onnx_burn::train::fit_supervised;

type AB = Autodiff<NdArray<f32>>;

const BIN: &str = "../../../ModelData/user_cal.bin";

fn main() {
    let device = Default::default();
    let bin = parse_bin(BIN).expect("parse bin");
    let cache = EyeCache::default();
    let samples = build_samples(&bin, &cache);
    println!("samples: {}", samples.len());
    let steps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let batch: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    fit_supervised::<AB>(&bin, &samples, steps, batch, 1e-3, &device);
}
