#!/usr/bin/env python
# Interim export bridge: Burn-format safetensors (PyTorch-keyed) -> timm/.pt -> existing ONNX export.
import os
import sys

os.environ.setdefault("BACKBONE", "mobilenetv4_conv_small_050.e3000_r224_in1k")
BT = r"C:\Users\dan2wik\Documents\GitHub\Babal-training\BabbleTrainer"
sys.path.insert(0, BT)

import torch
from safetensors.torch import load_file

import config as C
from model_dual import build_expr_net, build_gaze_net
from export_dual_onnx import export

HERE = os.path.dirname(os.path.abspath(__file__))
EXPR_ST = os.path.join(HERE, "expr_pt.safetensors")
GAZE_ST = os.path.join(HERE, "gaze_pt.safetensors")
EXPR_PT = os.path.join(HERE, "burn_expr.pt")
GAZE_PT = os.path.join(HERE, "burn_gaze.pt")
OUT_ONNX = os.path.join(HERE, "burn_model.onnx")


def load_into(net, st_path, tag):
    sd = load_file(st_path)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    miss = [k for k in missing if "num_batches_tracked" not in k]
    print(f"[{tag}] loaded {len(sd)} tensors | missing(non-nbt)={len(miss)} unexpected={len(unexpected)}")
    if miss:
        print(f"[{tag}] MISSING: {miss[:8]}{' ...' if len(miss) > 8 else ''}")
    if unexpected:
        print(f"[{tag}] UNEXPECTED: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    return net


def main():
    expr = load_into(build_expr_net(pretrained=False), EXPR_ST, "expr")
    gaze = load_into(build_gaze_net(), GAZE_ST, "gaze")
    expr_sd = {k: v.cpu() for k, v in expr.state_dict().items()}
    gaze_sd = {k: v.cpu() for k, v in gaze.state_dict().items()}
    torch.save({"teacher": expr_sd, "student": expr_sd, "backbone": C.BACKBONE, "step": 0}, EXPR_PT)
    torch.save({"gaze": gaze_sd, "step": 0, "arch": "microchad"}, GAZE_PT)
    print(f"wrote {EXPR_PT} + {GAZE_PT}")
    export(GAZE_PT, EXPR_PT, OUT_ONNX, check=False)
    print(f"exported ONNX: {OUT_ONNX}")
    print(f"\neval vs goalpost:\n  python eval_model.py {EXPR_PT} {GAZE_PT} --tag burn")


if __name__ == "__main__":
    main()
