use onnx_burn::{data, descriptor};
use serde_json::{Value, json};
use std::path::Path;

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

// Source of truth: Baballonia repo src/Baballonia.Descriptors/Schema/. Bump deliberately on sync.
const PINNED_SCHEMA_SHA256: &str = "951f3ee94ab742280c029dc31a1511c0bcc52f0dc0f49843b558ad766d37e638";

#[test]
fn schema_matches_pinned_hash() {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("schema/model_descriptor.v1.schema.json"),
    )
    .unwrap();
    let actual = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(actual, PINNED_SCHEMA_SHA256, "re-sync schema from Baballonia and bump the pin deliberately");
}

fn load_gray(name: &str) -> Vec<u8> {
    image::open(fixture(name)).unwrap().to_luma8().into_raw()
}

#[test]
fn equalize_hist_v1_golden() {
    let mut px = load_gray("golden_input_128.png");
    let expected = load_gray("golden_equalize_hist_v1.png");
    let chain = descriptor::build_preprocess_chain(&descriptor::preprocess_chain()).unwrap();
    for (f, params) in &chain {
        f(&mut px, params);
    }
    assert_eq!(px, expected, "equalize_hist v1 not byte-identical to golden");
}

#[test]
fn decode_ops_golden() {
    let fx: Value = serde_json::from_slice(&std::fs::read(fixture("golden_decode_ops.json")).unwrap()).unwrap();
    for (key, spec) in fx.as_object().unwrap() {
        let (op, v) = key.rsplit_once("_v").unwrap();
        let chain = json!([{ "op": op, "v": v.parse::<i64>().unwrap(), "params": spec["params"] }]);
        for case in spec["cases"].as_array().unwrap() {
            let (input, expected) = (case[0].as_f64().unwrap(), case[1].as_f64().unwrap());
            let got = descriptor::apply_decode(input, &chain).unwrap();
            assert_eq!(got, expected, "{key}({input})");
        }
    }
}

#[test]
fn unknown_op_rejected() {
    assert!(descriptor::build_preprocess_chain(&json!([{ "op": "nope", "v": 1 }])).is_err());
    assert!(descriptor::apply_decode(0.5, &json!([{ "op": "nope", "v": 1 }])).is_err());
}

#[test]
fn model_descriptor_matches_committed_sample() {
    let names: Vec<&str> = ["right", "left"]
        .iter()
        .flat_map(|s| {
            ["EyeY", "EyeX", "EyeLid", "EyeWiden", "EyeSquint", "EyeBrow"]
                .map(|e| format!("{s}{e}"))
        })
        .map(|n| Box::leak(n.into_boxed_str()) as &str)
        .chain(["combinedEyeWiden", "combinedEyeSquint", "combinedEyeBrow"])
        .collect();
    let mut got = descriptor::build_model_descriptor(&names);
    let mut sample: Value =
        serde_json::from_slice(&std::fs::read(fixture("sample_model_descriptor.json")).unwrap()).unwrap();
    // Provenance varies (trainer name/version/git/timestamp); compare everything else.
    got.as_object_mut().unwrap().remove("provenance");
    sample.as_object_mut().unwrap().remove("provenance");
    assert_eq!(got, sample, "Rust descriptor diverges from the trainer-committed sample");
}

// --- bin metadata pseudo-record ---

const FLAG_GOOD_DATA: i32 = 1 << 30;

fn record(state: i32, left: &[u8], right: &[u8]) -> Vec<u8> {
    let mut r = vec![0u8; 88];
    r.extend_from_slice(&state.to_le_bytes());
    r.extend_from_slice(&(left.len() as i32).to_le_bytes());
    r.extend_from_slice(&(right.len() as i32).to_le_bytes());
    r.extend_from_slice(left);
    r.extend_from_slice(right);
    r
}

fn tiny_jpeg() -> Vec<u8> {
    let img = image::GrayImage::from_pixel(8, 8, image::Luma([128u8]));
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Jpeg).unwrap();
    buf.into_inner()
}

fn bin_descriptor() -> Value {
    json!({
        "schema": 1, "kind": "capture_bin",
        "rig": { "slots": {
            "jpeg_left": { "stream": "eye_right", "kind": "eye.right" },
            "jpeg_right": { "stream": "eye_left", "kind": "eye.left" } } },
        "pixels": { "state": "raw", "preprocess": [] },
        "labels": { "gaze_range_deg": 45.0, "yaw_sign": "negated_from_overlay", "lid_convention": "1_open" }
    })
}

fn write_bin(path: &Path, desc: Option<&Value>) {
    let mut bytes = Vec::new();
    if let Some(d) = desc {
        let payload = [b"BBCM".as_slice(), d.to_string().as_bytes()].concat();
        bytes.extend_from_slice(&record(1 << 20, &payload, &[]));
    }
    let j = tiny_jpeg();
    for _ in 0..3 {
        bytes.extend_from_slice(&record(FLAG_GOOD_DATA, &j, &j));
    }
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn metadata_roundtrip_and_skip() {
    let dir = std::env::temp_dir().join("bb_desc_test");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("a.bin");
    write_bin(&p, Some(&bin_descriptor()));
    let bin = data::parse_bin(p.to_str().unwrap()).unwrap();
    assert_eq!(bin.frames.len(), 3, "metadata record must not appear as a frame");
    assert_eq!(bin.descriptor, Some(bin_descriptor()));
}

#[test]
fn legacy_bin_has_no_descriptor() {
    let dir = std::env::temp_dir().join("bb_desc_test");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("legacy.bin");
    write_bin(&p, None);
    let bin = data::parse_bin(p.to_str().unwrap()).unwrap();
    assert_eq!(bin.frames.len(), 3);
    assert!(bin.descriptor.is_none());
}

// Legacy real-capture smoke: BB_REAL_BIN=<path> cargo test — parses, no descriptor, frames intact.
#[test]
fn real_legacy_bin_parses_unchanged() {
    let Ok(path) = std::env::var("BB_REAL_BIN") else { return };
    let bin = data::parse_bin(&path).unwrap();
    assert!(bin.descriptor.is_none(), "legacy bin must have no descriptor");
    assert!(bin.frames.iter().any(|f| f.good()));
    let eye = bin
        .frames
        .iter()
        .enumerate()
        .filter(|(_, f)| f.good())
        .find_map(|(i, _)| data::process_eye(&bin, i, 1))
        .expect("at least one processable eye");
    assert_eq!(eye.len(), data::S * data::S);
}

#[test]
fn contradicting_bin_refused() {
    let dir = std::env::temp_dir().join("bb_desc_test");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("bad.bin");
    let mut bad = bin_descriptor();
    bad["rig"]["slots"]["jpeg_left"]["kind"] = json!("eye.left");
    bad["labels"]["gaze_range_deg"] = json!(30.0);
    write_bin(&p, Some(&bad));
    let Err(err) = data::parse_bin(p.to_str().unwrap()) else {
        panic!("contradicting bin descriptor must be refused");
    };
    let err = err.to_string();
    assert!(err.contains("jpeg_left") && err.contains("gaze_range_deg"), "{err}");
}
