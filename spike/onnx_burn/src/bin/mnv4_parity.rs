use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use onnx_burn::mnv4::Mnv4;

type B = NdArray<f32>;

fn read_npy_f32(path: &str) -> (Vec<f32>, Vec<usize>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let npy = npyz::NpyFile::new(&bytes[..]).unwrap();
    let shape: Vec<usize> = npy.shape().iter().map(|&d| d as usize).collect();
    let data: Vec<f32> = npy.into_vec().unwrap();
    (data, shape)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn main() {
    let device = Default::default();
    let mut model: Mnv4<B> = Mnv4::new(4, 4, &device);

    // timm safetensors carries no container-type, so the adapter can't infer BN norm-renaming;
    // remap BN weight/bias -> gamma/beta explicitly (conv/*_head names never end in bn/bn1/norm_head).
    let mut store = SafetensorsStore::from_file("../mnv4_weights.safetensors")
        .with_from_adapter(PyTorchToBurnAdapter)
        .with_key_remapping(r"(bn1|bn|norm_head)\.weight$", "${1}.gamma")
        .with_key_remapping(r"(bn1|bn|norm_head)\.bias$", "${1}.beta")
        .allow_partial(true)
        .validate(false);
    let res = model.load_from(&mut store).expect("load safetensors");
    println!("=== load result ===\n{res}");

    let (xd, xs) = read_npy_f32("../input_ref.npy");
    let input = Tensor::<B, 4>::from_data(TensorData::new(xd, xs), &device);

    let mut trace: Vec<(String, Vec<f32>)> = Vec::new();
    let _ = model.forward_trace(input, &mut trace);

    println!("=== block-by-block parity vs timm intermediates ===");
    let mut worst = 0.0f32;
    let mut first_fail: Option<String> = None;
    for (name, got) in &trace {
        let path = if name == "logits" || name == "feat" {
            format!("../intermediates/{name}.npy")
        } else if name == "stem" || name == "conv_head" {
            format!("../intermediates/{name}.npy")
        } else {
            format!("../intermediates/{name}.npy")
        };
        match std::fs::metadata(&path) {
            Ok(_) => {
                let (refd, _) = read_npy_f32(&path);
                let d = max_abs(got, &refd);
                worst = worst.max(if d.is_finite() { d } else { 0.0 });
                let flag = if d < 2e-3 { "ok" } else { "FAIL" };
                if d >= 2e-3 && first_fail.is_none() {
                    first_fail = Some(name.clone());
                }
                println!("{name:12} len={:6} max|diff|={d:.3e}  {flag}", got.len());
            }
            Err(_) => println!("{name:12} (no ref .npy)"),
        }
    }
    println!("\nworst max|diff| across all stages = {worst:.3e}");
    match first_fail {
        Some(n) => println!("PARITY: FAIL (first divergence at {n})"),
        None => println!("PARITY: PASS"),
    }
}
