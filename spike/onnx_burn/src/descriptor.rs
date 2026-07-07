// Descriptor support (descriptor design §1/§4, P4): op registries mirroring the C#/Python sides,
// capture-bin metadata pseudo-record consumption, and the model descriptor stamped at export.
// Schema source of truth: Baballonia repo; vendored copy in schema/ (pinned-hash test).

use serde_json::{Value, json};

pub const METADATA_MAGIC: &[u8; 4] = b"BBCM";
const FLAG_VERSION_MASK: i32 = 0xF << 20;
const FLAG_GOOD_DATA: i32 = 1 << 30;

pub const GAZE_RANGE_DEG: f64 = 45.0;
pub const MODEL_FAMILY: &str = "ET-dual";
pub const MODEL_REVISION: i64 = 3;

// Preprocess contract of the exported model. Single source of truth: process_eye applies this
// chain AND the exporter stamps it verbatim. Default == legacy behavior (global equalizeHist).
pub fn preprocess_chain() -> Value {
    json!([{ "op": "equalize_hist", "v": 1, "params": {} }])
}

// --- op registries (byte-identical contract with cv2/OpenCvSharp; golden test in tests/) ---

// cv2.equalizeHist: LUT[i] = round((cdf[i]-cdf_min) * 255 / (total-cdf_min)).
pub fn equalize_hist(px: &mut [u8]) {
    let mut hist = [0u32; 256];
    for &p in px.iter() {
        hist[p as usize] += 1;
    }
    let total = px.len() as u32;
    let mut cdf = [0u32; 256];
    let mut acc = 0u32;
    for i in 0..256 {
        acc += hist[i];
        cdf[i] = acc;
    }
    let cdf_min = cdf.iter().copied().find(|&c| c > 0).unwrap_or(0);
    let denom = (total - cdf_min).max(1) as f32;
    let mut lut = [0u8; 256];
    for i in 0..256 {
        let v = ((cdf[i].saturating_sub(cdf_min)) as f32 * 255.0 / denom).round();
        lut[i] = v.clamp(0.0, 255.0) as u8;
    }
    for p in px.iter_mut() {
        *p = lut[*p as usize];
    }
}

type PreprocessFn = fn(&mut [u8], &Value);

fn op_key(spec: &Value) -> Option<(&str, i64)> {
    Some((spec.get("op")?.as_str()?, spec.get("v")?.as_i64()?))
}

fn preprocess_op(op: &str, v: i64) -> Option<PreprocessFn> {
    match (op, v) {
        ("equalize_hist", 1) => Some(|px, _| equalize_hist(px)),
        _ => None,
    }
}

// Ordered {op,v,params} list -> composed fn. Unknown (op, v) = hard error.
pub fn build_preprocess_chain(chain: &Value) -> Result<Vec<(PreprocessFn, Value)>, String> {
    let list = chain.as_array().ok_or("preprocess chain must be an array")?;
    list.iter()
        .map(|spec| {
            let (op, v) = op_key(spec).ok_or_else(|| format!("malformed op spec {spec}"))?;
            let f = preprocess_op(op, v).ok_or_else(|| format!("unknown preprocess op {op} v{v}"))?;
            Ok((f, spec.get("params").cloned().unwrap_or(json!({}))))
        })
        .collect()
}

pub fn apply_default_preprocess(px: &mut [u8]) {
    use std::sync::OnceLock;
    static CHAIN: OnceLock<Vec<(PreprocessFn, Value)>> = OnceLock::new();
    for (f, params) in CHAIN.get_or_init(|| build_preprocess_chain(&preprocess_chain()).unwrap()) {
        f(px, params);
    }
}

pub fn apply_decode(mut value: f64, chain: &Value) -> Result<f64, String> {
    for spec in chain.as_array().ok_or("decode chain must be an array")? {
        let (op, v) = op_key(spec).ok_or_else(|| format!("malformed op spec {spec}"))?;
        value = match (op, v) {
            ("denorm_pm_deg", 1) => {
                let range = spec.pointer("/params/range_deg").and_then(Value::as_f64).unwrap_or(45.0);
                value * range
            }
            ("clamp01", 1) => value.clamp(0.0, 1.0),
            _ => return Err(format!("unknown decode op {op} v{v}")),
        };
    }
    Ok(value)
}

// --- capture-bin metadata pseudo-record (design §4) ---

pub fn is_metadata_record(state: i32, payload_head: &[u8]) -> bool {
    (state & FLAG_VERSION_MASK) != 0
        && (state & FLAG_GOOD_DATA) == 0
        && payload_head.len() >= 4
        && &payload_head[..4] == METADATA_MAGIC
}

pub fn parse_metadata_payload(payload: &[u8]) -> Option<Value> {
    serde_json::from_slice(payload.get(4..)?).ok()
}

// Legacy/default conventions every current bin is trained under (design §3 traps).
const EXPECTED_SLOT_KINDS: [(&str, &str); 2] = [("jpeg_left", "eye.right"), ("jpeg_right", "eye.left")];
const EXPECTED_YAW_SIGN: &str = "negated_from_overlay";
const EXPECTED_LID_CONVENTION: &str = "1_open";

// Refuse (Err) a bin whose declared conventions contradict this trainer's assumptions.
// None descriptor (legacy bin) = today's defaults, always fine.
pub fn check_bin_compat(desc: Option<&Value>) -> Result<(), String> {
    let Some(desc) = desc else { return Ok(()) };
    let mut errors = Vec::new();
    for (slot, want) in EXPECTED_SLOT_KINDS {
        if let Some(kind) = desc.pointer(&format!("/rig/slots/{slot}/kind")).and_then(Value::as_str)
            && kind != want
        {
            errors.push(format!("rig.slots.{slot}.kind={kind:?}, trainer assumes {want:?}"));
        }
    }
    if let Some(rng) = desc.pointer("/labels/gaze_range_deg").and_then(Value::as_f64)
        && rng != GAZE_RANGE_DEG
    {
        errors.push(format!("labels.gaze_range_deg={rng}, trainer assumes {GAZE_RANGE_DEG}"));
    }
    if let Some(yaw) = desc.pointer("/labels/yaw_sign").and_then(Value::as_str)
        && yaw != EXPECTED_YAW_SIGN
    {
        errors.push(format!("labels.yaw_sign={yaw:?}, trainer assumes {EXPECTED_YAW_SIGN:?}"));
    }
    if let Some(lid) = desc.pointer("/labels/lid_convention").and_then(Value::as_str)
        && lid != EXPECTED_LID_CONVENTION
    {
        errors.push(format!("labels.lid_convention={lid:?}, trainer assumes {EXPECTED_LID_CONVENTION:?}"));
    }
    if let Some(pre) = desc.pointer("/pixels/preprocess").and_then(Value::as_array)
        && !pre.is_empty()
    {
        errors.push(format!("pixels.preprocess={pre:?}, trainer assumes raw pixels"));
    }
    if errors.is_empty() { Ok(()) } else { Err(errors.join("; ")) }
}

// --- model descriptor (design §1), stamped by onnx_export ---

fn git_describe() -> Option<String> {
    let out = std::process::Command::new("git")
        .args(["describe", "--always", "--dirty"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

fn utc_now_iso() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    // Howard Hinnant's civil_from_days
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, (rem / 60) % 60, rem % 60)
}

fn output_entry(name: &str, index: usize) -> Value {
    const PER_EYE_KINDS: [(&str, &str); 6] = [
        ("EyeY", "gaze.pitch"), ("EyeX", "gaze.yaw"), ("EyeLid", "lid"),
        ("EyeWiden", "eye.widen"), ("EyeSquint", "eye.squint"), ("EyeBrow", "brow.angry"),
    ];
    for (suffix, base) in PER_EYE_KINDS {
        for side in ["left", "right", "combined"] {
            if name != format!("{side}{suffix}") {
                continue;
            }
            let kind = if side == "combined" { base.to_string() } else { format!("{base}.{side}") };
            return if base.starts_with("gaze.") {
                json!({ "index": index, "name": name, "kind": kind,
                        "decode": [{ "op": "denorm_pm_deg", "v": 1, "params": { "range_deg": GAZE_RANGE_DEG } }],
                        "units": "deg", "range": [-GAZE_RANGE_DEG, GAZE_RANGE_DEG],
                        "blendable": true, "confidence": null })
            } else {
                json!({ "index": index, "name": name, "kind": kind,
                        "decode": [{ "op": "clamp01", "v": 1, "params": {} }],
                        "units": "norm", "range": [0.0, 1.0], "blendable": true, "confidence": null })
            };
        }
    }
    panic!("no output kind mapping for blendshape name {name:?}");
}

// Full schema-1 model descriptor for the exported dual-eye ONNX. Mirrors BabbleTrainer's
// descriptors.build_model_descriptor (committed sample: tests/fixtures/sample_model_descriptor.json).
pub fn build_model_descriptor(names: &[&str]) -> Value {
    let (frames, size) = (crate::data::T, crate::data::S);
    let packing: Vec<Value> = (0..frames)
        .flat_map(|t| {
            [json!({ "ch": 2 * t, "stream": "eye_right", "t": t }),
             json!({ "ch": 2 * t + 1, "stream": "eye_left", "t": t })]
        })
        .collect();
    let mut trainer = json!({ "name": "BabbleBurn", "version": env!("CARGO_PKG_VERSION") });
    if let Some(git) = git_describe() {
        trainer["git"] = json!(git);
    }
    json!({
        "schema": 1,
        "provenance": { "trainer": trainer, "created_utc": utc_now_iso() },
        "model": { "family": MODEL_FAMILY, "revision": MODEL_REVISION, "roles": ["eyes"] },
        "input": {
            "tensor": "input", "layout": "NCHW", "dtype": "float32",
            "width": size, "height": size, "channels": 2 * frames,
            "range": [0.0, 1.0],
            "normalization": { "scale": "1/255", "mean": null, "std": null },
            "streams": {
                "eye_left": { "kind": "eye.left", "color": "gray" },
                "eye_right": { "kind": "eye.right", "color": "gray" }
            },
            "sync": { "preferred": "same_device", "max_skew_ms": 10 },
            "temporal": { "frames": frames, "order": "newest_first" },
            "packing": packing,
            "preprocess": { "per_stream": { "*": preprocess_chain() } }
        },
        "output": {
            "tensor": "output",
            "outputs": names.iter().enumerate().map(|(i, n)| output_entry(n, i)).collect::<Vec<_>>(),
            "post": []
        },
        "runtime": { "batch": "dynamic", "warmup_frames": frames }
    })
}
