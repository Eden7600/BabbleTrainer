// Interim export bridge: save Burn weights as PyTorch-compatible safetensors (gamma->weight, beta->bias,
// Linear transposed) via BurnToPyTorchAdapter. A Python step loads these into the timm tower + MicroChad
// and runs the existing export_dual_onnx.py. Param paths already mirror timm.
use burn::prelude::*;
use burn_store::{BurnToPyTorchAdapter, ModuleSnapshot, SafetensorsStore};

pub fn save_pytorch<B: Backend, M: ModuleSnapshot<B>>(model: &M, path: &str) {
    let mut store = SafetensorsStore::from_file(path)
        .with_to_adapter(BurnToPyTorchAdapter)
        .allow_partial(true)
        .overwrite(true);
    model.save_into(&mut store).expect("save pytorch-format safetensors");
    println!("wrote {path}");
}

// Same, but rewrites Burn keys via one regex before writing (e.g. strip the `heads.` Vec prefix so the
// consensus heads land on the Python nn.ModuleList keys `0.fc1.*` / `1.fc1.*`).
pub fn save_pytorch_remap<B: Backend, M: ModuleSnapshot<B>>(model: &M, path: &str, from: &str, to: &str) {
    let mut store = SafetensorsStore::from_file(path)
        .with_to_adapter(BurnToPyTorchAdapter)
        .with_key_remapping(from, to)
        .allow_partial(true)
        .overwrite(true);
    model.save_into(&mut store).expect("save pytorch-format safetensors");
    println!("wrote {path}");
}
