# Per-op forward benchmark on torch-directml (same ops/shapes as op_bench.rs) to compare per-op.
import os
import sys
import time

os.environ.setdefault("BACKBONE", "mobilenetv4_conv_small_050.e3000_r224_in1k")
BT = r"C:\Users\dan2wik\Documents\GitHub\Babal-training\BabbleTrainer"
sys.path.insert(0, BT)

import torch
import torch.nn as nn
import device as Dev

dev = Dev.get_device()
batch, iters = 64, 40
print(f"torch-directml per-op forward (batch {batch}):  device {dev}")


def time_op(name, mod, shape):
    mod = mod.to(dev).eval()
    x = torch.rand(*shape, device=dev)
    with torch.no_grad():
        for _ in range(20):
            _ = float(mod(x).sum().item())
        t = time.perf_counter()
        for _ in range(iters):
            _ = float(mod(x).sum().item())
    print(f"{name:28} {(time.perf_counter()-t)*1000/iters:6.2f} ms/fwd")


time_op("stem conv3x3 s2 4->32 @128", nn.Conv2d(4, 32, 3, 2, 1, bias=False), (batch, 4, 128, 128))
time_op("pointwise 1x1 32->96 @32", nn.Conv2d(32, 96, 1, bias=False), (batch, 32, 32, 32))
time_op("depthwise 3x3 g96 @32", nn.Conv2d(96, 96, 3, 1, 1, groups=96, bias=False), (batch, 96, 32, 32))
time_op("depthwise 5x5 g96 @16", nn.Conv2d(96, 96, 5, 1, 2, groups=96, bias=False), (batch, 96, 16, 16))
time_op("pointwise 1x1 64->256 @4", nn.Conv2d(64, 256, 1, bias=False), (batch, 64, 4, 4))
time_op("batchnorm 96 @32", nn.BatchNorm2d(96), (batch, 96, 32, 32))
