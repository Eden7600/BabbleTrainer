// .bin capture parsing + cv2-matching deterministic preprocessing + temporal stacking.
// Mirrors bin_reader.py / bin_temporal_dataset.py. Labels are exact arithmetic; frame pixels are
// approximately equal to cv2 (JPEG decoder differs from libjpeg-turbo) but equalizeHist matches cv2.

pub const S: usize = 128; // DUAL_INPUT_SIZE
pub const T: usize = 4; // TEMPORAL_FRAMES
const HEADER: usize = 100; // FRAME_SIZE
const MAX_JPEG: i32 = 10 * 1024 * 1024;
const FLAG_GOOD_DATA: i32 = 1 << 30;
const FLAG_GAZE_DATA: i32 = 1 << 0;
const FLAG_EXPR_UNLABELED: i32 = 1 << 1;
const GAZE_RANGE_DEG: f32 = 45.0;
const MIRROR_SIDE: usize = 0; // TRAIN_MIRROR_BIN_SIDE (side 0 = left jpeg gets flipped)

pub struct Frame {
    pub off: usize,        // byte offset of the JPEG-left start (header end)
    pub jl: usize,
    pub jr: usize,
    pub state: i32,
    pub fields: [f32; 16], // routine_pitch..dilate
}

impl Frame {
    pub fn good(&self) -> bool {
        self.state & FLAG_GOOD_DATA != 0
    }
    pub fn gaze_valid(&self) -> f32 {
        if self.state & FLAG_GAZE_DATA != 0 { 1.0 } else { 0.0 }
    }
    pub fn expr_unlabeled(&self) -> f32 {
        if self.good() && self.state & FLAG_EXPR_UNLABELED != 0 { 1.0 } else { 0.0 }
    }

    // frame_targets_dual(mirror_side=0) -> (left[6], right[6])
    pub fn labels(&self) -> ([f32; 6], [f32; 6]) {
        let f = &self.fields;
        let norm = |deg: f32| ((deg + GAZE_RANGE_DEG) / (2.0 * GAZE_RANGE_DEG)).clamp(0.0, 1.0);
        let (widen, squint, brow) = (f[13], f[14], f[12]);
        let (mut yaw_l, yaw_r) = (norm(f[6]), norm(f[8]));
        let (pitch_l, pitch_r) = (norm(f[5]), norm(f[7]));
        if MIRROR_SIDE == 0 {
            yaw_l = 1.0 - yaw_l;
        }
        let left = [pitch_l, yaw_l, 1.0 - f[9], widen, squint, brow];
        let right = [pitch_r, yaw_r, 1.0 - f[10], widen, squint, brow];
        (left, right)
    }
}

pub struct Bin {
    pub bytes: Vec<u8>,
    pub frames: Vec<Frame>,
}

pub fn parse_bin(path: &str) -> std::io::Result<Bin> {
    let bytes = std::fs::read(path)?;
    let mut frames = Vec::new();
    let mut pos = 0usize;
    while pos + HEADER <= bytes.len() {
        let h = &bytes[pos..pos + HEADER];
        let f32a = |i: usize| f32::from_le_bytes(h[i * 4..i * 4 + 4].try_into().unwrap());
        let i32a = |b: usize| i32::from_le_bytes(h[b..b + 4].try_into().unwrap());
        let mut fields = [0f32; 16];
        for (i, v) in fields.iter_mut().enumerate() {
            *v = f32a(i);
        }
        let state = i32a(88);
        let jl = i32a(92);
        let jr = i32a(96);
        if jl < 0 || jr < 0 || jl > MAX_JPEG || jr > MAX_JPEG {
            break;
        }
        let (jl, jr) = (jl as usize, jr as usize);
        let data_start = pos + HEADER;
        if data_start + jl + jr > bytes.len() {
            break;
        }
        frames.push(Frame { off: data_start, jl, jr, state, fields });
        pos = data_start + jl + jr;
    }
    Ok(Bin { bytes, frames })
}

fn decode_gray(buf: &[u8]) -> Option<Vec<u8>> {
    image::load_from_memory_with_format(buf, image::ImageFormat::Jpeg)
        .ok()
        .map(|im| im.to_luma8().into_raw())
}

fn std_dev(px: &[u8]) -> f32 {
    let n = px.len() as f32;
    let mean = px.iter().map(|&p| p as f32).sum::<f32>() / n;
    let var = px.iter().map(|&p| { let d = p as f32 - mean; d * d }).sum::<f32>() / n;
    var.sqrt()
}

// cv2.equalizeHist: LUT[i] = round((cdf[i]-cdf_min) * 255 / (total-cdf_min)).
fn equalize_hist(px: &mut [u8]) {
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

fn area_resize_to_s(px: &[u8], w: usize, h: usize) -> Vec<u8> {
    // INTER_AREA analogue (box average). Rarely used: this capture is already 128x128.
    if w == S && h == S {
        return px.to_vec();
    }
    let img = image::GrayImage::from_raw(w as u32, h as u32, px.to_vec()).unwrap();
    image::imageops::resize(&img, S as u32, S as u32, image::imageops::FilterType::Triangle).into_raw()
}

fn flip_h(px: &mut [u8]) {
    for row in px.chunks_mut(S) {
        row.reverse();
    }
}

// Decode+process one eye to canonical 128x128 uint8, or None if blank/unreadable.
// side 0 = left jpeg (mirror side, flipped), side 1 = right jpeg.
pub fn process_eye(bin: &Bin, rec: usize, side: usize) -> Option<Vec<u8>> {
    let fr = &bin.frames[rec];
    let (start, len) = if side == 0 { (fr.off, fr.jl) } else { (fr.off + fr.jl, fr.jr) };
    let raw = decode_gray(&bin.bytes[start..start + len])?;
    // is_blank on the pre-resize decoded pixels (std < 2.0)
    if std_dev(&raw) < 2.0 {
        return None;
    }
    // decoded is (h,w); this capture is square 128. Guard non-128 with a resize.
    let side_len = (raw.len() as f64).sqrt() as usize;
    let mut img = area_resize_to_s(&raw, side_len, side_len);
    equalize_hist(&mut img);
    if side == MIRROR_SIDE {
        flip_h(&mut img);
    }
    Some(img)
}

// current + 3 predecessors (zero-padded, no cross-bin: single-bin so guard rec-k >= 0), each 128x128,
// as a flat [T*S*S] uint8 in [t0(cur), t1, t2, t3] channel order.
pub fn temporal_stack(bin: &Bin, rec: usize, side: usize, cache: &EyeCache) -> Vec<u8> {
    let mut out = vec![0u8; T * S * S];
    for k in 0..T {
        if rec >= k {
            if let Some(f) = cache.get(bin, rec - k, side) {
                out[k * S * S..(k + 1) * S * S].copy_from_slice(&f);
            }
        }
    }
    out
}

// Lazily-filled per-eye processed-frame cache (Some(bytes) processed, None = blank/absent).
use std::cell::RefCell;
use std::collections::HashMap;
pub struct EyeCache {
    map: RefCell<HashMap<(usize, usize), Option<Vec<u8>>>>,
}
impl Default for EyeCache {
    fn default() -> Self {
        Self { map: RefCell::new(HashMap::new()) }
    }
}
impl EyeCache {
    pub fn get(&self, bin: &Bin, rec: usize, side: usize) -> Option<Vec<u8>> {
        if let Some(v) = self.map.borrow().get(&(rec, side)) {
            return v.clone();
        }
        let v = process_eye(bin, rec, side);
        self.map.borrow_mut().insert((rec, side), v.clone());
        v
    }
}

// A labeled per-eye sample referencing a record + side (built over good, non-blank eyes).
// gaze_usable = gaze_valid AND not a stuttered/stale gaze frame (raw gaze identical to the previous
// record's, which corrupts gaze labels — see the capture stutter issue).
pub struct Sample {
    pub rec: usize,
    pub side: usize,
    pub label: [f32; 6],
    pub gaze_valid: f32,
    pub expr_unlabeled: f32,
    pub gaze_usable: bool,
}

fn raw_gaze(fr: &Frame, side: usize) -> (f32, f32) {
    if side == 0 { (fr.fields[5], fr.fields[6]) } else { (fr.fields[7], fr.fields[8]) }
}

// Build the sample index (mirror of build_index's good+non-blank per-eye enumeration).
pub fn build_samples(bin: &Bin, cache: &EyeCache) -> Vec<Sample> {
    let mut out = Vec::new();
    for rec in 0..bin.frames.len() {
        let fr = &bin.frames[rec];
        if !fr.good() {
            continue;
        }
        let (l, r) = fr.labels();
        let (gv, eu) = (fr.gaze_valid(), fr.expr_unlabeled());
        for (side, lab) in [(0usize, l), (1usize, r)] {
            if cache.get(bin, rec, side).is_some() {
                let sticky = rec > 0 && raw_gaze(&bin.frames[rec - 1], side) == raw_gaze(fr, side);
                let gaze_usable = gv > 0.5 && !sticky;
                out.push(Sample { rec, side, label: lab, gaze_valid: gv, expr_unlabeled: eu, gaze_usable });
            }
        }
    }
    out
}
