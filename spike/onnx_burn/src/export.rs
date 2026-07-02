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
