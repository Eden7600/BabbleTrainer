# Per-step benchmark of the SAME MNv4 expr net (fwd+MSE+bwd+ForeachAdamW step) on torch-directml.
import os
import sys
import time

os.environ.setdefault("BACKBONE", "mobilenetv4_conv_small_050.e3000_r224_in1k")
BT = r"C:\Users\dan2wik\Documents\GitHub\Babal-training\BabbleTrainer"
sys.path.insert(0, BT)

import torch
import device as Dev
from model_dual import build_expr_net

dev = Dev.get_device()
print(f"device: {dev}  (torch {torch.__version__})")

for batch in [32, 64, 96, 128]:
    net = build_expr_net(pretrained=False).to(dev).train()
    opt = Dev.make_adamw([{"params": list(net.parameters()), "lr": 1e-4}], dev, 1e-4)
    x = torch.rand(batch, 4, 128, 128, device=dev)
    y = torch.rand(batch, 4, device=dev)
    for _ in range(15):
        opt.zero_grad()
        loss = ((net(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    _ = float(loss.item())  # drain
    t = time.perf_counter()
    it = 50
    for _ in range(it):
        opt.zero_grad()
        loss = ((net(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    _ = float(loss.item())  # drain queue before stopping clock
    ms = (time.perf_counter() - t) * 1000 / it
    print(f"directml TRAIN batch {batch:3}: {ms:6.1f} ms/step | {ms/batch:.2f} ms/sample")
