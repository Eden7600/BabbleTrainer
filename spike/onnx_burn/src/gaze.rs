// MicroChad: the tiny 6-conv gaze net (model_dual.py MicroChad). 4-ch temporal in -> 2 gaze outs.
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::prelude::*;
use burn::tensor::activation::{relu, sigmoid};

#[derive(Module, Debug)]
pub struct MicroChad<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    conv5: Conv2d<B>,
    conv6: Conv2d<B>,
    fc_gaze: Linear<B>,
    pool: MaxPool2d,
}

impl<B: Backend> MicroChad<B> {
    pub fn new(in_ch: usize, out: usize, dev: &B::Device) -> Self {
        let c = |i: usize, o: usize| {
            Conv2dConfig::new([i, o], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(dev)
        };
        Self {
            conv1: c(in_ch, 14),
            conv2: c(14, 21),
            conv3: c(21, 32),
            conv4: c(32, 47),
            conv5: c(47, 70),
            conv6: c(70, 106),
            fc_gaze: LinearConfig::new(106, out).init(dev),
            pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let mut h = self.pool.forward(relu(self.conv1.forward(x)));
        h = self.pool.forward(relu(self.conv2.forward(h)));
        h = self.pool.forward(relu(self.conv3.forward(h)));
        h = self.pool.forward(relu(self.conv4.forward(h)));
        h = self.pool.forward(relu(self.conv5.forward(h)));
        h = relu(self.conv6.forward(h));
        let g = h.max_dim(3).max_dim(2); // GlobalMaxPool over H,W -> [B,106,1,1]
        let [b, c, _, _] = g.dims();
        sigmoid(self.fc_gaze.forward(g.reshape([b, c])))
    }
}
