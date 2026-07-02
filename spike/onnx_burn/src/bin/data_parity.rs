// Verify the Rust .bin parser + preprocessing vs the Python reference (spike/dataref/refs.npz).
// Labels must be exact; frame pixels are ~equal (JPEG decoder differs from cv2/libjpeg-turbo).
use npyz::npz::NpzArchive;
use onnx_burn::data::{build_samples, parse_bin, process_eye, EyeCache, S};

const BIN: &str = "../../../ModelData/user_cal.bin";
const REFS: &str = "../dataref/refs.npz";

fn npz_f32(a: &mut NpzArchive<std::io::BufReader<std::fs::File>>, name: &str) -> (Vec<f32>, Vec<usize>) {
    let arr = a.by_name(name).unwrap().unwrap();
    let shape = arr.shape().iter().map(|&d| d as usize).collect();
    (arr.into_vec::<f32>().unwrap(), shape)
}
fn npz_u8(a: &mut NpzArchive<std::io::BufReader<std::fs::File>>, name: &str) -> Vec<u8> {
    a.by_name(name).unwrap().unwrap().into_vec::<u8>().unwrap()
}

fn main() {
    let bin = parse_bin(BIN).expect("parse bin");
    let good = bin.frames.iter().filter(|f| f.good()).count();
    println!("parsed {} records, {} good", bin.frames.len(), good);

    let mut z = NpzArchive::open(REFS).expect("open refs.npz");
    let (ll, _) = npz_f32(&mut z, "left_labels");
    let (rl, _) = npz_f32(&mut z, "right_labels");
    let (gv, _) = npz_f32(&mut z, "gaze_valid");
    let f0 = npz_u8(&mut z, "frames_side0");
    let f1 = npz_u8(&mut z, "frames_side1");

    // first 16 good records, in file order
    let good_idx: Vec<usize> = (0..bin.frames.len()).filter(|&i| bin.frames[i].good()).take(16).collect();

    let mut lab_max = 0f32;
    let mut gv_ok = true;
    for (n, &rec) in good_idx.iter().enumerate() {
        let (l, r) = bin.frames[rec].labels();
        for c in 0..6 {
            lab_max = lab_max.max((l[c] - ll[n * 6 + c]).abs());
            lab_max = lab_max.max((r[c] - rl[n * 6 + c]).abs());
        }
        if (bin.frames[rec].gaze_valid() - gv[n]).abs() > 1e-6 {
            gv_ok = false;
        }
    }
    println!("label max|diff|   = {lab_max:.3e}  (must be ~0)");
    println!("gaze_valid match  = {gv_ok}");

    // frame pixel agreement (approx; JPEG decoder differs). Report mean abs diff + % within 3 levels.
    let cache = EyeCache::default();
    let frame_stats = |side: usize, refbuf: &[u8]| -> (f32, f32) {
        let mut sum = 0f64;
        let mut within = 0usize;
        let mut cnt = 0usize;
        for (n, &rec) in good_idx.iter().enumerate() {
            let got = process_eye(&bin, rec, side).unwrap();
            let refimg = &refbuf[n * S * S..(n + 1) * S * S];
            for i in 0..S * S {
                let d = (got[i] as i32 - refimg[i] as i32).unsigned_abs();
                sum += d as f64;
                if d <= 3 {
                    within += 1;
                }
                cnt += 1;
            }
        }
        ((sum / cnt as f64) as f32, within as f32 / cnt as f32 * 100.0)
    };
    let (m0, w0) = frame_stats(0, &f0);
    let (m1, w1) = frame_stats(1, &f1);
    println!("side0 frame mean|diff|={m0:.3} levels, {w0:.1}% within 3");
    println!("side1 frame mean|diff|={m1:.3} levels, {w1:.1}% within 3");

    let samples = build_samples(&bin, &cache);
    println!("built {} labeled eye-samples", samples.len());

    let pass = lab_max < 1e-5 && gv_ok && m0 < 3.0 && m1 < 3.0;
    println!("DATA PARITY: {}", if pass { "PASS" } else { "FAIL" });
}
