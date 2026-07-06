// Load trained Burn checkpoints (.mpk) and write the deployment ONNX natively — no Python bridge.
use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use onnx_burn::gaze::MicroChad;
use onnx_burn::heads::ConsensusHeads;
use onnx_burn::mnv4::Mnv4;
use onnx_burn::onnx_export::export_merged;

type B = NdArray<f32>;

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "../burn_native.onnx".into());
    let device = Default::default();
    let rec = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let expr = Mnv4::<B>::new(4, 4, &device)
        .load_file("../ckpt_expr.mpk", &rec, &device)
        .expect("load expr ckpt");
    let gaze = MicroChad::<B>::new(4, 2, &device)
        .load_file("../ckpt_gaze.mpk", &rec, &device)
        .expect("load gaze ckpt");

    // Consensus heads are optional (only trained when the capture has paired eyes): 15-dim
    // brow-fused export when present, 12-dim otherwise.
    let cons = std::path::Path::new("../ckpt_cons.mpk").exists().then(|| {
        ConsensusHeads::<B>::new(1280, &device)
            .load_file("../ckpt_cons.mpk", &rec, &device)
            .expect("load cons ckpt")
    });

    export_merged(&expr, &gaze, cons.as_ref(), &out).expect("write onnx");
    println!("exported {} ({}-dim)", out, if cons.is_some() { 15 } else { 12 });
}
