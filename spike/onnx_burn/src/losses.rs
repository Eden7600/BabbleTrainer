// Loss terms as pure tensor ops. Supervised path first (reg + range hinge); MT/temporal/evenness/
// consensus added alongside the curriculum. LAMBDA_RANGE = 0.01.
use burn::prelude::*;

pub const LAMBDA_RANGE: f32 = 0.01;

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
