// Loss terms as pure tensor ops. Supervised path first (reg + range hinge); MT/temporal/evenness/
// consensus added alongside the curriculum. LAMBDA_RANGE = 0.01.
use burn::prelude::*;

pub const LAMBDA_RANGE: f32 = 0.01;

// Per-step loss constants built ONCE (device tensors), hoisted out of the training loop instead of
// rebuilt via from_floats every step: the deconfuse matrix [4,4], evenness weights, temporal margins.
pub struct ExprConsts<B: Backend> {
    pub deconf: Tensor<B, 2>, // [4,4] EXPR_DECONFUSE rows=drivers, cols=targets
    pub even_w: Tensor<B, 2>, // [1,4]
    pub margin: Tensor<B, 2>, // [1,4]
}

impl<B: Backend> ExprConsts<B> {
    pub fn new(device: &B::Device) -> Self {
        let deconf = Tensor::<B, 1>::from_floats(
            [0.0f32, 0.0, 0.5, 0.7, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.7, 0.0, 0.7, 0.0],
            device,
        )
        .reshape([4, 4]);
        let even_w = Tensor::<B, 1>::from_floats([0.3f32, 1.0, 0.3, 0.3], device).reshape([1, 4]);
        let margin = Tensor::<B, 1>::from_floats([0.35f32, 0.10, 0.10, 0.10], device).reshape([1, 4]);
        Self { deconf, even_w, margin }
    }
}

pub fn reg_mse<B: Backend>(pred: Tensor<B, 2>, tgt: Tensor<B, 2>) -> Tensor<B, 1> {
    (pred - tgt).powf_scalar(2.0).mean()
}

// (p - clamp(p,0,1))^2 keeps raw expr outputs near [0,1] (exact-grad range hinge).
pub fn range_hinge<B: Backend>(pred: Tensor<B, 2>) -> Tensor<B, 1> {
    let clamped = pred.clone().clamp(0.0, 1.0);
    (pred - clamped).powf_scalar(2.0).mean()
}

pub fn supervised<B: Backend>(pred: Tensor<B, 2>, tgt: Tensor<B, 2>) -> Tensor<B, 1> {
    reg_mse(pred.clone(), tgt) + range_hinge(pred).mul_scalar(LAMBDA_RANGE)
}

// EXPR_DECONFUSE rows=drivers(lid,widen,squint,brow), cols=targets. weight=prod_d clamp_min(1-tgt_d*row,0)
// `deconf` is the [4,4] matrix, prebuilt once (ExprConsts) rather than from_floats per step.
pub fn deconfuse_weight<B: Backend>(tgt: Tensor<B, 2>, deconf: &Tensor<B, 2>) -> Tensor<B, 2> {
    let [b, _] = tgt.dims();
    let mut w: Tensor<B, 2> = Tensor::ones([b, 4], &tgt.device());
    for d in 0..4 {
        let td = tgt.clone().slice([0..b, d..d + 1]); // [b,1]
        let row = deconf.clone().slice([d..d + 1, 0..4]); // [1,4]
        let f = td.mul(row).neg().add_scalar(1.0).clamp_min(0.0); // 1 - tgt_d*row, clamped
        w = w.mul(f);
    }
    w
}

pub fn supervised_deconfuse<B: Backend>(pred: Tensor<B, 2>, tgt: Tensor<B, 2>, deconf: &Tensor<B, 2>) -> Tensor<B, 1> {
    let w = deconfuse_weight(tgt.clone(), deconf);
    let sq = (pred.clone() - tgt).powf_scalar(2.0);
    let reg = (sq * w.clone()).sum() / w.sum().clamp_min(1.0);
    reg + range_hinge(pred).mul_scalar(LAMBDA_RANGE)
}

// agree-gated L/R symmetry pull (Mechanism C). even_w = [0.3,1,0.3,0.3] (prebuilt).
pub fn evenness<B: Backend>(
    pl: Tensor<B, 2>, pr: Tensor<B, 2>, ll: Tensor<B, 2>, lr: Tensor<B, 2>, even_w: &Tensor<B, 2>,
) -> Tensor<B, 1> {
    let agree = (ll - lr).abs().neg().add_scalar(1.0).clamp(0.0, 1.0);
    let diff2 = (pl - pr).powf_scalar(2.0);
    (even_w.clone() * agree.clone() * diff2).sum() / agree.sum().clamp_min(1.0)
}

pub fn mt_consistency<B: Backend>(pred_s: Tensor<B, 2>, t_pred: Tensor<B, 2>) -> Tensor<B, 1> {
    (pred_s - t_pred).powf_scalar(2.0).mean()
}

// temporal smoothness margin hinge. margin=[0.35,0.10,0.10,0.10] (prebuilt), gated by pair_ok [B].
pub fn temporal<B: Backend>(
    pw: Tensor<B, 2>, pn: Tensor<B, 2>, pair_ok: Tensor<B, 1>, margin: &Tensor<B, 2>,
) -> Tensor<B, 1> {
    let [b, _] = pw.dims();
    let d = (pw - pn).abs();
    let l = (d - margin.clone()).clamp_min(0.0).powf_scalar(2.0).mean_dim(1).reshape([b]);
    (l * pair_ok.clone()).sum() / pair_ok.sum().clamp_min(1.0)
}

pub fn consensus_term<B: Backend>(head_out: Tensor<B, 2>, tgt: Tensor<B, 2>) -> Tensor<B, 1> {
    (head_out - tgt).powf_scalar(2.0).mean()
}
