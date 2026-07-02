// ConsensusHead: fc-relu-fc-sigmoid on concatenated L/R (detached) features (model_dual.py).
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{relu, sigmoid};

#[derive(Module, Debug)]
pub struct ConsensusHead<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> ConsensusHead<B> {
    pub fn new(feat_dim: usize, out: usize, hidden: usize, dev: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(feat_dim * 2, hidden).init(dev),
            fc2: LinearConfig::new(hidden, out).init(dev),
        }
    }

    pub fn forward(&self, feat_left: Tensor<B, 2>, feat_right: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = Tensor::cat(vec![feat_left, feat_right], 1);
        sigmoid(self.fc2.forward(relu(self.fc1.forward(x))))
    }
}

// CONSENSUS_GROUPS = [[1,2],[3]] -> heads producing 2 and 1 outputs (widen/squint, brow).
pub fn build_consensus_heads<B: Backend>(feat_dim: usize, dev: &B::Device) -> Vec<ConsensusHead<B>> {
    [2usize, 1]
        .iter()
        .map(|&out| ConsensusHead::new(feat_dim, out, 64, dev))
        .collect()
}
