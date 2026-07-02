// Load trained Burn checkpoints (.mpk) and emit PyTorch-format safetensors for the Python export bridge.
use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use onnx_burn::export::save_pytorch;
use onnx_burn::gaze::MicroChad;
use onnx_burn::mnv4::Mnv4;

type B = NdArray<f32>;

fn main() {
    let device = Default::default();
    let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let expr = Mnv4::<B>::new(4, 4, &device)
        .load_file("../ckpt_expr.mpk", &rec, &device)
        .expect("load expr ckpt");
    save_pytorch(&expr, "../expr_pt.safetensors");

    let gaze = MicroChad::<B>::new(4, 2, &device)
        .load_file("../ckpt_gaze.mpk", &rec, &device)
        .expect("load gaze ckpt");
    save_pytorch(&gaze, "../gaze_pt.safetensors");
}
