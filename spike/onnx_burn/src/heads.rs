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

// Two consensus heads as one Module (group [1,2]=widen/squint -> 2 outs, group [3]=brow -> 1 out).
// Burn keys `heads.0.fc1.*` / `heads.1.fc1.*` remap to the Python ModuleList keys `0.fc1.*` / `1.fc1.*`.
#[derive(Module, Debug)]
pub struct ConsensusHeads<B: Backend> {
    heads: Vec<ConsensusHead<B>>,
}

impl<B: Backend> ConsensusHeads<B> {
    pub fn new(feat_dim: usize, dev: &B::Device) -> Self {
        Self { heads: build_consensus_heads(feat_dim, dev) }
    }
    // group 0 -> [widen, squint] (2), group 1 -> [brow] (1)
    pub fn forward(&self, i: usize, fl: Tensor<B, 2>, fr: Tensor<B, 2>) -> Tensor<B, 2> {
        self.heads[i].forward(fl, fr)
    }
}
