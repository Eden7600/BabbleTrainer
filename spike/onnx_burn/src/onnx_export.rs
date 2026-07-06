// Native ONNX export: builds the MergedDualEye graph (model_dual.py) straight from the trained Burn
// modules and writes an opset-18 ModelProto — no torch/Python in the loop. Protobuf encoding is
// hand-rolled (ONNX needs ~10 message types); BN is folded into conv weights, matching what
// torch.onnx.export produces under do_constant_folding. Graph contract mirrors export_dual_onnx.py:
// input "input" [batch,8,128,128] gray01, output "output" [batch,12|15], same metadata_props.

use burn::nn::conv::Conv2d;
use burn::nn::{BatchNorm, Linear, PaddingConfig2d};
use burn::prelude::*;

use crate::gaze::MicroChad;
use crate::heads::ConsensusHeads;
use crate::mnv4::Mnv4;

// ---- protobuf primitives (proto3 wire format: varint=0, fixed32=5, len-delimited=2) ----

fn uvarint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(b);
            return;
        }
        out.push(b | 0x80);
    }
}

fn tag(out: &mut Vec<u8>, field: u64, wire: u64) {
    uvarint(out, (field << 3) | wire);
}

fn pb_varint(out: &mut Vec<u8>, field: u64, v: u64) {
    tag(out, field, 0);
    uvarint(out, v);
}

fn pb_bytes(out: &mut Vec<u8>, field: u64, data: &[u8]) {
    tag(out, field, 2);
    uvarint(out, data.len() as u64);
    out.extend_from_slice(data);
}

fn pb_str(out: &mut Vec<u8>, field: u64, s: &str) {
    pb_bytes(out, field, s.as_bytes());
}

fn pb_f32(out: &mut Vec<u8>, field: u64, v: f32) {
    tag(out, field, 5);
    out.extend_from_slice(&v.to_le_bytes());
}

// ---- ONNX graph builder ----

pub enum Attr<'a> {
    I(&'a str, i64),
    Is(&'a str, &'a [i64]),
    F(&'a str, f32),
}

// AttributeProto: name=1 f=2 i=3 type=20 ints=8 (type enum: FLOAT=1 INT=2 INTS=7)
fn attr_bytes(a: &Attr) -> Vec<u8> {
    let mut b = Vec::new();
    match a {
        Attr::I(name, v) => {
            pb_str(&mut b, 1, name);
            tag(&mut b, 3, 0);
            uvarint(&mut b, *v as u64);
            pb_varint(&mut b, 20, 2);
        }
        Attr::Is(name, vs) => {
            pb_str(&mut b, 1, name);
            for v in *vs {
                tag(&mut b, 8, 0);
                uvarint(&mut b, *v as u64);
            }
            pb_varint(&mut b, 20, 7);
        }
        Attr::F(name, v) => {
            pb_str(&mut b, 1, name);
            pb_f32(&mut b, 2, *v);
            pb_varint(&mut b, 20, 1);
        }
    }
    b
}

pub struct OnnxGraph {
    nodes: Vec<u8>,  // encoded NodeProto stream
    inits: Vec<u8>,  // encoded TensorProto stream
    n: usize,        // unique-name counter
    // weight-initializer names keyed by ParamId: the tower runs once per eye, weights emit once
    cache: std::collections::HashMap<String, (String, Option<String>)>,
}

impl Default for OnnxGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl OnnxGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), inits: Vec::new(), n: 0, cache: Default::default() }
    }

    fn name(&mut self, prefix: &str) -> String {
        self.n += 1;
        format!("{prefix}_{}", self.n)
    }

    // TensorProto: dims=1 data_type=2 name=8 raw_data=9 (f32=1, i64=7; raw is little-endian)
    pub fn init_f32(&mut self, prefix: &str, dims: &[usize], data: &[f32]) -> String {
        let name = self.name(prefix);
        let mut t = Vec::new();
        for d in dims {
            pb_varint(&mut t, 1, *d as u64);
        }
        pb_varint(&mut t, 2, 1);
        pb_str(&mut t, 8, &name);
        let mut raw = Vec::with_capacity(data.len() * 4);
        for v in data {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        pb_bytes(&mut t, 9, &raw);
        pb_bytes(&mut self.inits, 5, &t); // GraphProto.initializer = 5
        name
    }

    pub fn init_i64(&mut self, prefix: &str, vals: &[i64]) -> String {
        let name = self.name(prefix);
        let mut t = Vec::new();
        pb_varint(&mut t, 1, vals.len() as u64);
        pb_varint(&mut t, 2, 7);
        pb_str(&mut t, 8, &name);
        let mut raw = Vec::with_capacity(vals.len() * 8);
        for v in vals {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        pb_bytes(&mut t, 9, &raw);
        pb_bytes(&mut self.inits, 5, &t);
        name
    }

    fn init_scalar(&mut self, prefix: &str, v: f32) -> String {
        let name = self.name(prefix);
        let mut t = Vec::new();
        pb_varint(&mut t, 2, 1);
        pb_str(&mut t, 8, &name);
        pb_bytes(&mut t, 9, &v.to_le_bytes());
        pb_bytes(&mut self.inits, 5, &t);
        name
    }

    // NodeProto: input=1 output=2 name=3 op_type=4 attribute=5
    pub fn node(&mut self, op: &str, ins: &[&str], attrs: &[Attr]) -> String {
        let out = self.name(op);
        let mut nd = Vec::new();
        for i in ins {
            pb_str(&mut nd, 1, i);
        }
        pb_str(&mut nd, 2, &out);
        pb_str(&mut nd, 3, &out);
        pb_str(&mut nd, 4, op);
        for a in attrs {
            pb_bytes(&mut nd, 5, &attr_bytes(a));
        }
        pb_bytes(&mut self.nodes, 1, &nd); // GraphProto.node = 1
        out
    }

    // op helpers (each returns the output tensor name)

    pub fn relu(&mut self, x: &str) -> String {
        self.node("Relu", &[x], &[])
    }

    pub fn sigmoid(&mut self, x: &str) -> String {
        self.node("Sigmoid", &[x], &[])
    }

    pub fn add(&mut self, a: &str, b: &str) -> String {
        self.node("Add", &[a, b], &[])
    }

    pub fn concat(&mut self, ins: &[&str]) -> String {
        self.node("Concat", ins, &[Attr::I("axis", 1)])
    }

    pub fn flatten(&mut self, x: &str) -> String {
        self.node("Flatten", &[x], &[Attr::I("axis", 1)])
    }

    pub fn global_avg_pool(&mut self, x: &str) -> String {
        self.node("GlobalAveragePool", &[x], &[])
    }

    pub fn maxpool2(&mut self, x: &str) -> String {
        self.node("MaxPool", &[x], &[
            Attr::Is("kernel_shape", &[2, 2]),
            Attr::Is("strides", &[2, 2]),
            Attr::Is("pads", &[0, 0, 0, 0]),
        ])
    }

    // GlobalMaxPool over H,W. ReduceMax because axes moved to an input at opset 18.
    pub fn reduce_max_hw(&mut self, x: &str) -> String {
        let axes = self.init_i64("axes", &[2, 3]);
        self.node("ReduceMax", &[x, &axes], &[Attr::I("keepdims", 1)])
    }

    pub fn gather_ch(&mut self, x: &str, idx: &[i64]) -> String {
        let i = self.init_i64("idx", idx);
        self.node("Gather", &[x, &i], &[Attr::I("axis", 1)])
    }

    pub fn slice(&mut self, x: &str, axis: i64, start: i64, end: i64, step: i64) -> String {
        let s = self.init_i64("starts", &[start]);
        let e = self.init_i64("ends", &[end]);
        let a = self.init_i64("axes", &[axis]);
        let st = self.init_i64("steps", &[step]);
        self.node("Slice", &[x, &s, &e, &a, &st], &[])
    }

    // torch.flip(x, dims=[-1]) as exported: negative-step Slice over W.
    pub fn flip_w(&mut self, x: &str) -> String {
        self.slice(x, 3, -1, i64::MIN + 1, -1)
    }

    pub fn one_minus(&mut self, x: &str) -> String {
        let one = self.init_scalar("one", 1.0);
        self.node("Sub", &[&one, x], &[])
    }

    pub fn clip01(&mut self, x: &str) -> String {
        let lo = self.init_scalar("zero", 0.0);
        let hi = self.init_scalar("one", 1.0);
        self.node("Clip", &[x, &lo, &hi], &[])
    }

    // Burn Linear weight is [d_in, d_out], so Gemm runs with the default transB=0.
    pub fn linear<B: Backend>(&mut self, prefix: &str, x: &str, lin: &Linear<B>) -> String {
        let key = format!("{:?}", lin.weight.id);
        let (wn, bn) = match self.cache.get(&key) {
            Some(hit) => hit.clone(),
            None => {
                let w = lin.weight.val();
                let wn = self.init_f32(&format!("{prefix}_w"), &w.dims(), &to_vec(w.to_data()));
                let bias = lin.bias.as_ref().expect("linear bias").val();
                let bn = self.init_f32(&format!("{prefix}_b"), &bias.dims(), &to_vec(bias.to_data()));
                self.cache.insert(key, (wn.clone(), Some(bn.clone())));
                (wn, Some(bn))
            }
        };
        self.node("Gemm", &[x, &wn, &bn.unwrap()], &[])
    }

    // Conv straight from the module (bias kept as-is; MicroChad's convs have no BN).
    pub fn conv<B: Backend>(&mut self, prefix: &str, x: &str, conv: &Conv2d<B>) -> String {
        let w = conv.weight.val();
        let b = conv.bias.as_ref().map(|b| to_vec(b.val().to_data()));
        self.conv_raw(prefix, x, conv, &to_vec(w.to_data()), b)
    }

    // Conv with the following BatchNorm folded in (eval-mode fold, per out-channel):
    //   w' = w * gamma/sqrt(var+eps), b' = beta - mean * gamma/sqrt(var+eps)
    pub fn conv_bn<B: Backend>(&mut self, prefix: &str, x: &str, conv: &Conv2d<B>, bn: &BatchNorm<B>) -> String {
        let w = to_vec(conv.weight.val().to_data());
        let dims = conv.weight.val().dims();
        let gamma = to_vec(bn.gamma.val().to_data());
        let beta = to_vec(bn.beta.val().to_data());
        let mean = to_vec(bn.running_mean.value().to_data());
        let var = to_vec(bn.running_var.value().to_data());
        let cout = dims[0];
        let per = w.len() / cout;
        let mut wf = w;
        let mut bf = vec![0f32; cout];
        for o in 0..cout {
            let s = gamma[o] / (var[o] + 1e-5f32).sqrt();
            for v in &mut wf[o * per..(o + 1) * per] {
                *v *= s;
            }
            bf[o] = beta[o] - mean[o] * s;
        }
        self.conv_raw(prefix, x, conv, &wf, Some(bf))
    }

    fn conv_raw<B: Backend>(&mut self, prefix: &str, x: &str, conv: &Conv2d<B>, w: &[f32], b: Option<Vec<f32>>) -> String {
        let key = format!("{:?}", conv.weight.id);
        let (wn, bn) = match self.cache.get(&key) {
            Some(hit) => hit.clone(),
            None => {
                let dims = conv.weight.val().dims();
                let wn = self.init_f32(&format!("{prefix}_w"), &dims, w);
                let bn = b.map(|b| self.init_f32(&format!("{prefix}_b"), &[b.len()], &b));
                self.cache.insert(key, (wn.clone(), bn.clone()));
                (wn, bn)
            }
        };
        let (pt, pl, pb, pr) = match conv.padding {
            PaddingConfig2d::Explicit(t, l, bo, r) => (t as i64, l as i64, bo as i64, r as i64),
            _ => (0, 0, 0, 0),
        };
        let k = [conv.kernel_size[0] as i64, conv.kernel_size[1] as i64];
        let s = [conv.stride[0] as i64, conv.stride[1] as i64];
        let attrs = [
            Attr::Is("kernel_shape", &k),
            Attr::Is("strides", &s),
            Attr::Is("pads", &[pt, pl, pb, pr]),
            Attr::Is("dilations", &[1, 1]),
            Attr::I("group", conv.groups as i64),
        ];
        match bn {
            Some(bn) => self.node("Conv", &[x, &wn, &bn], &attrs),
            None => self.node("Conv", &[x, &wn], &attrs),
        }
    }

    // ValueInfoProto: name=1 type=2; TypeProto.tensor_type=1 {elem_type=1 shape=2};
    // Dimension: dim_value=1 | dim_param=2. First dim is the dynamic "batch_size".
    fn value_info(name: &str, tail_dims: &[usize]) -> Vec<u8> {
        let mut shape = Vec::new();
        let mut d0 = Vec::new();
        pb_str(&mut d0, 2, "batch_size");
        pb_bytes(&mut shape, 1, &d0);
        for d in tail_dims {
            let mut dv = Vec::new();
            pb_varint(&mut dv, 1, *d as u64);
            pb_bytes(&mut shape, 1, &dv);
        }
        let mut tensor = Vec::new();
        pb_varint(&mut tensor, 1, 1); // f32
        pb_bytes(&mut tensor, 2, &shape);
        let mut ty = Vec::new();
        pb_bytes(&mut ty, 1, &tensor);
        let mut vi = Vec::new();
        pb_str(&mut vi, 1, name);
        pb_bytes(&mut vi, 2, &ty);
        vi
    }

    // Assemble ModelProto (ir_version 8, opset 18) and write it. `names` -> blendshape_names metadata.
    pub fn finish(self, path: &str, out_dim: usize, names: &[&str]) -> std::io::Result<()> {
        let mut g = Vec::new();
        g.extend_from_slice(&self.nodes);
        pb_str(&mut g, 2, "babbleburn");
        g.extend_from_slice(&self.inits);
        pb_bytes(&mut g, 11, &Self::value_info("input", &[8, 128, 128]));
        pb_bytes(&mut g, 12, &Self::value_info("output", &[out_dim]));

        let mut m = Vec::new();
        pb_varint(&mut m, 1, 8); // ir_version
        pb_str(&mut m, 2, "babbleburn");
        let mut opset = Vec::new();
        pb_str(&mut opset, 1, "");
        pb_varint(&mut opset, 2, 18);
        pb_bytes(&mut m, 8, &opset);
        pb_bytes(&mut m, 7, &g);
        let json = format!(
            "[{}]",
            names.iter().map(|n| format!("\"{n}\"")).collect::<Vec<_>>().join(", ")
        );
        for (k, v) in [
            ("blendshape_names", json.as_str()),
            ("input_layout", "interleaved_LR_t0..t3_8x128x128_gray01"),
            ("gaze_range_deg", "45.0"),
        ] {
            let mut kv = Vec::new();
            pb_str(&mut kv, 1, k);
            pb_str(&mut kv, 2, v);
            pb_bytes(&mut m, 14, &kv);
        }
        std::fs::write(path, m)
    }
}

fn to_vec(data: burn::tensor::TensorData) -> Vec<f32> {
    data.to_vec().expect("f32 tensor data")
}

// Route the final value to the graph output name via Identity (node outputs are baked at emit time).
fn to_output(g: &mut OnnxGraph, x: &str) {
    let mut nd = Vec::new();
    pb_str(&mut nd, 1, x);
    pb_str(&mut nd, 2, "output");
    pb_str(&mut nd, 3, "output_identity");
    pb_str(&mut nd, 4, "Identity");
    pb_bytes(&mut g.nodes, 1, &nd);
}

const BLENDSHAPES_12: [&str; 12] = [
    "rightEyeY", "rightEyeX", "rightEyeLid", "rightEyeWiden", "rightEyeSquint", "rightEyeBrow",
    "leftEyeY", "leftEyeX", "leftEyeLid", "leftEyeWiden", "leftEyeSquint", "leftEyeBrow",
];
const BLENDSHAPES_15: [&str; 15] = [
    "rightEyeY", "rightEyeX", "rightEyeLid", "rightEyeWiden", "rightEyeSquint", "rightEyeBrow",
    "leftEyeY", "leftEyeX", "leftEyeLid", "leftEyeWiden", "leftEyeSquint", "leftEyeBrow",
    "combinedEyeWiden", "combinedEyeSquint", "combinedEyeBrow",
];

// Brow fusion (EXPR_FUSE_CHANNELS=[3], source=consensus): both per-eye brow slots read the combined
// brow (col 14) — MergedDualEye.gather_idx.
const FUSE_IDX: [i64; 15] = [0, 1, 2, 3, 4, 14, 6, 7, 8, 9, 10, 14, 12, 13, 14];

/// Export the merged dual-eye model: tower on both eyes (right = mirrored), rightEye yaw un-flipped,
/// [right6, left6] clamped to [0,1], plus the consensus block + brow fusion when heads are present.
/// Input channel split mirrors config.py SWAP_INPUT_EYES=True: [1,3,5,7] unflipped -> leftEye*,
/// [0,2,4,6] hflipped -> rightEye*.
pub fn export_merged<B: Backend>(
    expr: &Mnv4<B>,
    gaze: &MicroChad<B>,
    cons: Option<&ConsensusHeads<B>>,
    path: &str,
) -> std::io::Result<()> {
    let mut g = OnnxGraph::new();

    let left_in = g.gather_ch("input", &[1, 3, 5, 7]);
    let picked = g.gather_ch("input", &[0, 2, 4, 6]);
    let right_in = g.flip_w(&picked);

    let (feat_l, expr_l) = expr.onnx(&mut g, "l", &left_in);
    let gaze_l = gaze.onnx(&mut g, "l", &left_in);
    let (feat_r, expr_r) = expr.onnx(&mut g, "r", &right_in);
    let gaze_r = gaze.onnx(&mut g, "r", &right_in);

    // right eye ran on a mirrored frame: un-mirror yaw (EyeX -> 1-x), keep the rest
    let right6 = g.concat(&[&gaze_r, &expr_r]);
    let r_y = g.slice(&right6, 1, 0, 1, 1);
    let r_x = g.slice(&right6, 1, 1, 2, 1);
    let r_x = g.one_minus(&r_x);
    let r_rest = g.slice(&right6, 1, 2, i64::MAX, 1);

    let merged = g.concat(&[&r_y, &r_x, &r_rest, &gaze_l, &expr_l]);
    let merged = g.clip01(&merged);

    match cons {
        None => {
            to_output(&mut g, &merged);
            g.finish(path, 12, &BLENDSHAPES_12)
        }
        Some(heads) => {
            let combined = heads.onnx(&mut g, &feat_l, &feat_r);
            let combined = g.clip01(&combined);
            let full = g.concat(&[&merged, &combined]);
            let fused = g.gather_ch(&full, &FUSE_IDX);
            to_output(&mut g, &fused);
            g.finish(path, 15, &BLENDSHAPES_15)
        }
    }
}
