"""
Burn-import fidelity spike: reference artifact generator for the BabbleTrainer
expression backbone (mobilenetv4_conv_small_050).

ANALYSIS/GENERATION ONLY. Runs on CPU. Idempotent.
"""
import os
import sys
import warnings
from collections import Counter

import numpy as np
import torch
import timm
import onnx
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Setup: force CPU, deterministic input, output paths.
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")
torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))  # be gentle

HERE = os.path.dirname(os.path.abspath(__file__))
def p(name):
    return os.path.join(HERE, name)

MODEL_NAME = "mobilenetv4_conv_small_050.e3000_r224_in1k"

print("=" * 70)
print("BabbleBurn spike / make_ref.py")
print("=" * 70)
print(f"torch        : {torch.__version__}")
print(f"timm         : {timm.__version__}")
print(f"onnx         : {onnx.__version__}")
print(f"onnxruntime  : {ort.__version__}")
print(f"numpy        : {np.__version__}")
print(f"device       : {DEVICE}")
print(f"output dir   : {HERE}")
print("-" * 70)

# ---------------------------------------------------------------------------
# 2. Build the EXACT trainer backbone, with pretrained fallback.
# ---------------------------------------------------------------------------
used_pretrained = True
try:
    net = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        in_chans=4,
        num_classes=4,
        drop_rate=0.2,
    )
    print(f"[model] created with pretrained=True")
except Exception as e:  # noqa: BLE001 - want any download/network failure
    print(f"[model] pretrained=True FAILED: {type(e).__name__}: {e}")
    print(f"[model] retrying with pretrained=False ...")
    used_pretrained = False
    net = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        in_chans=4,
        num_classes=4,
        drop_rate=0.2,
    )
    print(f"[model] created with pretrained=False")

net = net.to(DEVICE)
net.eval()

# ---------------------------------------------------------------------------
# 3. Fixed input.
# ---------------------------------------------------------------------------
torch.manual_seed(0)
x = torch.rand(1, 4, 128, 128, device=DEVICE)
np.save(p("input_ref.npy"), x.numpy())
print(f"[input] x shape={tuple(x.shape)} dtype={x.dtype} -> input_ref.npy")

# ---------------------------------------------------------------------------
# 4. Export to ONNX (opset 17). Surface any op-related warnings verbatim.
# ---------------------------------------------------------------------------
onnx_path = p("backbone.onnx")
captured_warnings = []
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    torch.onnx.export(
        net,
        x,
        onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "B"}, "logits": {0: "B"}},
    )
    captured_warnings = list(wlist)

print(f"[onnx] exported -> backbone.onnx (opset 17)")
if captured_warnings:
    print("[onnx] export warnings (verbatim):")
    for w in captured_warnings:
        print(f"    {w.category.__name__}: {w.message}")
else:
    print("[onnx] export produced no python warnings")

# ---------------------------------------------------------------------------
# 5. ONNXRuntime vs torch parity.
# ---------------------------------------------------------------------------
with torch.no_grad():
    torch_out = net(x).cpu().numpy()
np.save(p("torch_out.npy"), torch_out)

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
ort_out = sess.run(["logits"], {"input": x.numpy()})[0]
np.save(p("ort_out.npy"), ort_out)

max_abs_diff = float(np.max(np.abs(ort_out - torch_out)))
print(f"[parity] torch_out shape={torch_out.shape}  ort_out shape={ort_out.shape}")
print(f"[parity] max|ort - torch| = {max_abs_diff:.3e}")

# ---------------------------------------------------------------------------
# 6. Pre-logits feature (expr_feat_dim).
# ---------------------------------------------------------------------------
with torch.no_grad():
    feat = net.forward_head(net.forward_features(x), pre_logits=True)
feat_np = feat.cpu().numpy()
np.save(p("feat_ref.npy"), feat_np)
expr_feat_dim = int(feat_np.reshape(feat_np.shape[0], -1).shape[1])
print(f"[feat] pre-logits feature shape={feat_np.shape}  expr_feat_dim={expr_feat_dim}")

# ---------------------------------------------------------------------------
# 7. Weights manifest + module statistics.
# ---------------------------------------------------------------------------
manifest_lines = []
n_param_tensors = 0
for name, prm in net.named_parameters():
    manifest_lines.append(f"{name}  {tuple(prm.shape)}  {prm.dtype}")
    n_param_tensors += 1

manifest_lines.append("")  # separator
manifest_lines.append("# --- buffers ---")
n_buffer_tensors = 0
for name, buf in net.named_buffers():
    manifest_lines.append(f"{name}  {tuple(buf.shape)}  {buf.dtype}")
    n_buffer_tensors += 1

# Module inspection.
n_conv = 0
n_depthwise = 0
n_bn = 0
activation_types = Counter()

# Known activation class names to look for.
ACT_NAMES = {
    "ReLU", "ReLU6", "GELU", "SiLU", "Swish", "Hardswish", "HardSwish",
    "Hardsigmoid", "HardSigmoid", "Sigmoid", "Tanh", "Mish", "LeakyReLU",
    "ELU", "PReLU", "GELUTanh", "QuickGELU",
}

for mod_name, mod in net.named_modules():
    cls = type(mod).__name__
    if isinstance(mod, torch.nn.Conv2d):
        n_conv += 1
        is_dw = (mod.groups > 1) or (mod.groups == mod.in_channels and mod.in_channels > 1)
        if is_dw:
            n_depthwise += 1
    # BatchNorm2d and timm's BatchNormAct2d (subclass of BatchNorm2d, but be safe)
    if isinstance(mod, torch.nn.BatchNorm2d) or cls in ("BatchNormAct2d", "BatchNorm2d"):
        n_bn += 1
    if cls in ACT_NAMES:
        activation_types[cls] += 1

manifest_lines.append("")
manifest_lines.append("# --- counts ---")
manifest_lines.append(f"total_param_tensors  {n_param_tensors}")
manifest_lines.append(f"total_buffer_tensors  {n_buffer_tensors}")
manifest_lines.append(f"conv2d_modules  {n_conv}")
manifest_lines.append(f"depthwise_conv2d_modules  {n_depthwise}")
manifest_lines.append(f"batchnorm_modules  {n_bn}")
act_summary = ", ".join(f"{k}:{v}" for k, v in sorted(activation_types.items())) or "(none detected)"
manifest_lines.append(f"activation_types  {act_summary}")
manifest_lines.append(f"used_pretrained  {used_pretrained}")
manifest_lines.append(f"expr_feat_dim  {expr_feat_dim}")

with open(p("weights_manifest.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(manifest_lines) + "\n")

print("-" * 70)
print(f"[manifest] total_param_tensors    = {n_param_tensors}")
print(f"[manifest] total_buffer_tensors   = {n_buffer_tensors}")
print(f"[manifest] conv2d_modules         = {n_conv}")
print(f"[manifest] depthwise_conv2d_modules = {n_depthwise}")
print(f"[manifest] batchnorm_modules      = {n_bn}")
print(f"[manifest] activation_types       = {act_summary}")
print(f"[manifest] -> weights_manifest.txt")

# ---------------------------------------------------------------------------
# 8. ONNX op_type histogram (key output for burn-import coverage prediction).
# ---------------------------------------------------------------------------
model_onnx = onnx.load(onnx_path)
op_counter = Counter(node.op_type for node in model_onnx.graph.node)

optype_lines = []
print("-" * 70)
print("[onnx] unique op_types (sorted):")
for op, cnt in sorted(op_counter.items()):
    line = f"{op}  {cnt}"
    optype_lines.append(line)
    print(f"    {line}")
optype_lines.append("")
optype_lines.append(f"# total_nodes  {sum(op_counter.values())}")
optype_lines.append(f"# unique_op_types  {len(op_counter)}")

with open(p("onnx_optypes.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(optype_lines) + "\n")
print(f"[onnx] total nodes = {sum(op_counter.values())}, unique op_types = {len(op_counter)}")
print(f"[onnx] -> onnx_optypes.txt")

# ---------------------------------------------------------------------------
# Existence + size confirmation for artifacts.
# ---------------------------------------------------------------------------
print("-" * 70)
print("[artifacts] existence / sizes:")
for fname in ("input_ref.npy", "torch_out.npy", "ort_out.npy", "feat_ref.npy", "backbone.onnx"):
    fpath = p(fname)
    ok = os.path.exists(fpath)
    size = os.path.getsize(fpath) if ok else -1
    print(f"    {'OK ' if ok else 'MISS'} {fname:18s} {size:>12,d} bytes" if ok
          else f"    MISS {fname}")

print("=" * 70)
print("SUMMARY")
print(f"  used_pretrained    : {used_pretrained}")
print(f"  max|ort - torch|   : {max_abs_diff:.3e}")
print(f"  expr_feat_dim      : {expr_feat_dim}")
print(f"  onnx unique optypes: {len(op_counter)}")
print("=" * 70)
