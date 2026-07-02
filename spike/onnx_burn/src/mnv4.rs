// Hand-written MobileNetV4-conv-small (width 0.5), 4-channel temporal stem, trainable BatchNorm.
// Param paths mirror timm keys so the timm<->Burn map is mechanical (PyTorchToBurnAdapter handles
// weight->gamma/beta + Linear transpose). Blueprint: spike/blocks_structure.txt / weight_keys.txt.

use burn::module::{Ignored, Module};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::prelude::*;
use burn::tensor::activation::relu;

fn bn<B: Backend>(c: usize, dev: &B::Device) -> BatchNorm<B> {
    BatchNormConfig::new(c).with_epsilon(1e-5).with_momentum(0.1).init(dev)
}

fn conv2d<B: Backend>(cin: usize, cout: usize, k: usize, stride: usize, groups: usize, dev: &B::Device) -> Conv2d<B> {
    Conv2dConfig::new([cin, cout], [k, k])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(k / 2, k / 2, k / 2, k / 2))
        .with_groups(groups)
        .with_bias(false)
        .init(dev)
}

// conv -> bn -> optional relu (UIR sub-conv: dw_start/pw_exp/dw_mid/pw_proj)
#[derive(Module, Debug)]
pub struct ConvBn<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B>,
    act: Ignored<bool>,
}

impl<B: Backend> ConvBn<B> {
    fn make(cin: usize, cout: usize, k: usize, stride: usize, groups: usize, act: bool, dev: &B::Device) -> Self {
        Self { conv: conv2d(cin, cout, k, stride, groups, dev), bn: bn(cout, dev), act: Ignored(act) }
    }
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let h = self.bn.forward(self.conv.forward(x));
        if *self.act { relu(h) } else { h }
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    conv: Option<Conv2d<B>>,       // ConvBnAct: bare conv (timm blocks.s.b.conv)
    bn1: Option<BatchNorm<B>>,  // ConvBnAct: bn1 (relu)
    dw_start: Option<ConvBn<B>>,   // UIR (no act)
    pw_exp: Option<ConvBn<B>>,     // UIR (relu)
    dw_mid: Option<ConvBn<B>>,     // UIR (relu)
    pw_proj: Option<ConvBn<B>>,    // UIR (no act)
    residual: Ignored<bool>,
}

impl<B: Backend> Block<B> {
    fn cba(cin: usize, cout: usize, k: usize, stride: usize, dev: &B::Device) -> Self {
        Self {
            conv: Some(conv2d(cin, cout, k, stride, 1, dev)),
            bn1: Some(bn(cout, dev)),
            dw_start: None, pw_exp: None, dw_mid: None, pw_proj: None,
            residual: Ignored(false),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn uir(cin: usize, cout: usize, exp: usize, dw_start_k: Option<usize>,
           dw_mid: Option<(usize, usize)>, residual: bool, dev: &B::Device) -> Self {
        Self {
            conv: None,
            bn1: None,
            dw_start: dw_start_k.map(|k| ConvBn::make(cin, cin, k, 1, cin, false, dev)),
            pw_exp: Some(ConvBn::make(cin, exp, 1, 1, 1, true, dev)),
            dw_mid: dw_mid.map(|(k, s)| ConvBn::make(exp, exp, k, s, exp, true, dev)),
            pw_proj: Some(ConvBn::make(exp, cout, 1, 1, 1, false, dev)),
            residual: Ignored(residual),
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let inp = x.clone();
        let mut h = x;
        if let (Some(c), Some(b)) = (&self.conv, &self.bn1) {
            h = relu(b.forward(c.forward(h)));
        } else {
            if let Some(m) = &self.dw_start { h = m.forward(h); }
            if let Some(m) = &self.pw_exp { h = m.forward(h); }
            if let Some(m) = &self.dw_mid { h = m.forward(h); }
            if let Some(m) = &self.pw_proj { h = m.forward(h); }
        }
        if *self.residual { h + inp } else { h }
    }
}

#[derive(Module, Debug)]
pub struct Mnv4<B: Backend> {
    conv_stem: Conv2d<B>,
    bn1: BatchNorm<B>,
    blocks: Vec<Vec<Block<B>>>,
    conv_head: Conv2d<B>,
    norm_head: BatchNorm<B>,
    classifier: Linear<B>,
    pool: Ignored<AdaptiveAvgPool2d>,
}

impl<B: Backend> Mnv4<B> {
    pub fn new(in_ch: usize, num_classes: usize, dev: &B::Device) -> Self {
        let s2 = |cin, cout, exp, ds, dm, res| Block::uir(cin, cout, exp, ds, dm, res, dev);
        let blocks = vec![
            // stage 0 (ConvBnAct)
            vec![Block::cba(32, 16, 3, 2, dev), Block::cba(16, 16, 1, 1, dev)],
            // stage 1 (ConvBnAct)
            vec![Block::cba(16, 48, 3, 2, dev), Block::cba(48, 32, 1, 1, dev)],
            // stage 2 (UIR)
            vec![
                s2(32, 48, 96, Some(5), Some((5, 2)), false),
                s2(48, 48, 96, None, Some((3, 1)), true),
                s2(48, 48, 96, None, Some((3, 1)), true),
                s2(48, 48, 96, None, Some((3, 1)), true),
                s2(48, 48, 96, None, Some((3, 1)), true),
                s2(48, 48, 192, Some(3), None, true),
            ],
            // stage 3 (UIR)
            vec![
                s2(48, 64, 288, Some(3), Some((3, 2)), false),
                s2(64, 64, 256, Some(5), Some((5, 1)), true),
                s2(64, 64, 256, None, Some((5, 1)), true),
                s2(64, 64, 192, None, Some((5, 1)), true),
                s2(64, 64, 256, None, Some((3, 1)), true),
                s2(64, 64, 256, None, Some((3, 1)), true),
            ],
            // stage 4 (ConvBnAct)
            vec![Block::cba(64, 480, 1, 1, dev)],
        ];
        Self {
            conv_stem: conv2d(in_ch, 32, 3, 2, 1, dev),
            bn1: bn(32, dev),
            blocks,
            conv_head: conv2d(480, 1280, 1, 1, 1, dev),
            norm_head: bn(1280, dev),
            classifier: LinearConfig::new(1280, num_classes).with_bias(true).init(dev),
            pool: Ignored(AdaptiveAvgPool2dConfig::new([1, 1]).init()),
        }
    }

    pub fn forward_features(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut h = relu(self.bn1.forward(self.conv_stem.forward(x)));
        for stage in &self.blocks {
            for blk in stage {
                h = blk.forward(h);
            }
        }
        // MobileNetV3-style head: pool FIRST, then conv_head(1x1) acts as an FC, then norm+relu.
        let pooled = self.pool.forward(h); // [B,480,1,1]
        let h = relu(self.norm_head.forward(self.conv_head.forward(pooled))); // [B,1280,1,1]
        let [b, c, _, _] = h.dims();
        h.reshape([b, c])
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        self.classifier.forward(self.forward_features(x))
    }

    // Collect per-stage outputs for block-by-block parity. stem/conv_head are recorded CONV-ONLY to
    // match the reference hooks (which tapped conv_stem / conv_head before their BN+ReLU).
    pub fn forward_trace(&self, x: Tensor<B, 4>, trace: &mut Vec<(String, Vec<f32>)>) -> Tensor<B, 2> {
        let stem_conv = self.conv_stem.forward(x);
        trace.push(("stem".into(), stem_conv.to_data().to_vec().unwrap()));
        let mut h = relu(self.bn1.forward(stem_conv));
        for (si, stage) in self.blocks.iter().enumerate() {
            for (bi, blk) in stage.iter().enumerate() {
                h = blk.forward(h);
                trace.push((format!("block_{si}_{bi}"), h.to_data().to_vec().unwrap()));
            }
        }
        let pooled = self.pool.forward(h); // [B,480,1,1]
        let head_conv = self.conv_head.forward(pooled); // [B,1280,1,1] (matches ref conv_head hook)
        trace.push(("conv_head".into(), head_conv.to_data().to_vec().unwrap()));
        let hh = relu(self.norm_head.forward(head_conv));
        let [b, c, _, _] = hh.dims();
        let feat = hh.reshape([b, c]);
        trace.push(("feat".into(), feat.to_data().to_vec().unwrap()));
        let logits = self.classifier.forward(feat);
        trace.push(("logits".into(), logits.to_data().to_vec().unwrap()));
        logits
    }
}
