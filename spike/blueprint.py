"""
Exact structural blueprint + per-block reference activations for
timm's mobilenetv4_conv_small_050, to guide a from-scratch Burn (Rust)
reimplementation with trainable BatchNorm.

ANALYSIS/GENERATION ONLY. CPU only. Idempotent.

Reuses the fixed input at spike/input_ref.npy so everything is consistent
with the earlier spike (make_ref.py).
"""
import os
import sys
from collections import Counter, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import timm
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Setup: force CPU. Do NOT touch the GPU.
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")
torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))  # be gentle on the box
torch.manual_seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))
def p(*names):
    return os.path.join(HERE, *names)

INTER_DIR = p("intermediates")
os.makedirs(INTER_DIR, exist_ok=True)

MODEL_NAME = "mobilenetv4_conv_small_050.e3000_r224_in1k"

print("=" * 74)
print("BabbleBurn spike / blueprint.py")
print("=" * 74)
print(f"python : {sys.version.split()[0]}")
print(f"torch  : {torch.__version__}")
print(f"timm   : {timm.__version__}")
print(f"numpy  : {np.__version__}")
print(f"device : {DEVICE}")
print(f"outdir : {HERE}")
print("-" * 74)

# ---------------------------------------------------------------------------
# Build the exact trainer backbone, with pretrained fallback.
# ---------------------------------------------------------------------------
used_pretrained = True
try:
    net = timm.create_model(
        MODEL_NAME, pretrained=True, in_chans=4, num_classes=4, drop_rate=0.2,
    )
    print("[model] created with pretrained=True")
except Exception as e:  # noqa: BLE001
    print(f"[model] pretrained=True FAILED: {type(e).__name__}: {e}")
    print("[model] retrying with pretrained=False ...")
    used_pretrained = False
    net = timm.create_model(
        MODEL_NAME, pretrained=False, in_chans=4, num_classes=4, drop_rate=0.2,
    )
    print("[model] created with pretrained=False")

net = net.to(DEVICE).eval()

# ---------------------------------------------------------------------------
# Fixed input (reuse existing input_ref.npy).
# ---------------------------------------------------------------------------
input_path = p("input_ref.npy")
if not os.path.exists(input_path):
    raise FileNotFoundError(f"expected {input_path} to already exist")
x_np = np.load(input_path).astype(np.float32)
x = torch.from_numpy(x_np).to(DEVICE)
print(f"[input] reused input_ref.npy shape={tuple(x.shape)} dtype={x.dtype}")

# ---------------------------------------------------------------------------
# Helper: classify + describe leaf modules.
# ---------------------------------------------------------------------------
ACT_NAMES = {
    "ReLU", "ReLU6", "GELU", "SiLU", "Swish", "Hardswish", "HardSwish",
    "Hardsigmoid", "HardSigmoid", "Sigmoid", "Tanh", "Mish", "LeakyReLU",
    "ELU", "PReLU", "GELUTanh", "QuickGELU", "Identity",
}

def _as_pair(v):
    if isinstance(v, (tuple, list)):
        return tuple(v)
    return (v, v)

def conv_cfg(m: nn.Conv2d) -> str:
    k = _as_pair(m.kernel_size)
    s = _as_pair(m.stride)
    pad = _as_pair(m.padding)
    depthwise = (m.groups == m.in_channels and m.in_channels > 1)
    return (f"in={m.in_channels}, out={m.out_channels}, "
            f"k={k[0]}x{k[1]}, stride={s[0]}x{s[1]}, pad={pad[0]}x{pad[1]}, "
            f"groups={m.groups}, bias={m.bias is not None}, depthwise={depthwise}")

def bn_inner_act(m) -> str:
    """For timm BatchNormAct2d, report the inner activation module type."""
    act = getattr(m, "act", None)
    if act is None:
        return "None"
    cls = type(act).__name__
    return "None" if cls == "Identity" else cls

def bn_cfg(m) -> str:
    mom = m.momentum
    return (f"features={m.num_features}, eps={m.eps}, "
            f"momentum={mom}, act={bn_inner_act(m)}")

def linear_cfg(m: nn.Linear) -> str:
    return f"in={m.in_features}, out={m.out_features}, bias={m.bias is not None}"

def pool_cfg(m) -> str:
    cls = type(m).__name__
    bits = [cls]
    for attr in ("output_size", "pool_type", "flatten"):
        if hasattr(m, attr):
            bits.append(f"{attr}={getattr(m, attr)!r}")
    return ", ".join(bits)

def is_leaf_compute(m) -> bool:
    """A leaf that has weights or does compute we care about."""
    cls = type(m).__name__
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        return True
    if isinstance(m, nn.BatchNorm2d) or cls in ("BatchNormAct2d", "BatchNorm2d"):
        return True
    if cls in ACT_NAMES and cls != "Identity":
        return True
    if "Pool" in cls or cls in ("Flatten", "SelectAdaptivePool2d"):
        return True
    if cls in ("Dropout", "DropPath"):
        return True
    return False

def describe_leaf(name, m) -> str:
    cls = type(m).__name__
    if isinstance(m, nn.Conv2d):
        return f"{name} | {cls} | {conv_cfg(m)}"
    if isinstance(m, nn.BatchNorm2d) or cls in ("BatchNormAct2d", "BatchNorm2d"):
        return f"{name} | {cls} | {bn_cfg(m)}"
    if isinstance(m, nn.Linear):
        return f"{name} | {cls} | {linear_cfg(m)}"
    if "Pool" in cls or cls in ("Flatten", "SelectAdaptivePool2d"):
        return f"{name} | {cls} | {pool_cfg(m)}"
    if cls in ("Dropout", "DropPath"):
        rate = getattr(m, "p", getattr(m, "drop_prob", None))
        return f"{name} | {cls} | p={rate}"
    return f"{name} | {cls} |"

# ---------------------------------------------------------------------------
# ARTIFACT 1: blueprint.txt -- full architecture in EXECUTION ORDER, grouped.
# ---------------------------------------------------------------------------
def top_group(name: str) -> str:
    """Map a dotted module name to its top-level container header."""
    if name == "":
        return "(root)"
    parts = name.split(".")
    if parts[0] == "blocks" and len(parts) >= 2:
        return f"blocks.{parts[1]}"
    return parts[0]

# named_modules() yields in definition/registration order which for timm's
# feature-forward models matches execution order.
leaf_lines_by_group = OrderedDict()
group_order = []
# names of BatchNormAct2d modules -- their inner .act/.drop children are folded
# into the parent line, so we suppress them to avoid a redundant bare-ReLU line.
bna_names = {
    n for n, m in net.named_modules()
    if type(m).__name__ in ("BatchNormAct2d", "BatchNorm2d") or isinstance(m, nn.BatchNorm2d)
}
def _child_of_bna(name: str) -> bool:
    return any(name == bn + ".act" or name == bn + ".drop" for bn in bna_names)

for name, m in net.named_modules():
    if name == "":
        continue
    if not is_leaf_compute(m):
        continue
    cls = type(m).__name__
    is_bn = isinstance(m, nn.BatchNorm2d) or cls in ("BatchNormAct2d", "BatchNorm2d")
    # Emit BatchNormAct2d itself (it has .act/.drop children but IS the compute
    # node the Burn port needs -- trainable BN + fused activation).
    if not is_bn:
        # skip container-ish modules that also matched (only true leaves)
        if len(list(m.children())) > 0:
            continue
        # suppress the redundant inner act/drop child of a BatchNormAct2d
        if _child_of_bna(name):
            continue
    g = top_group(name)
    if g not in leaf_lines_by_group:
        leaf_lines_by_group[g] = []
        group_order.append(g)
    leaf_lines_by_group[g].append(describe_leaf(name, m))

with open(p("blueprint.txt"), "w", encoding="utf-8") as f:
    f.write(f"# mobilenetv4_conv_small_050 architecture in EXECUTION ORDER\n")
    f.write(f"# model={MODEL_NAME}  in_chans=4  num_classes=4  drop_rate=0.2\n")
    f.write(f"# pretrained={used_pretrained}\n")
    f.write(f"# format: <dotted.name> | <ModuleType> | <config>\n")
    f.write("=" * 74 + "\n")
    for g in group_order:
        f.write(f"\n### {g}\n")
        for line in leaf_lines_by_group[g]:
            f.write(line + "\n")
print(f"[artifact] blueprint.txt written ({sum(len(v) for v in leaf_lines_by_group.values())} leaf lines)")

# ---------------------------------------------------------------------------
# ARTIFACT 2: blocks_structure.txt -- block-level connectivity map.
# ---------------------------------------------------------------------------
def block_has_skip(blk) -> (bool, str):
    """Determine whether the block adds a residual skip on forward."""
    for attr in ("has_skip",):
        if hasattr(blk, attr):
            return bool(getattr(blk, attr)), f"attr:{attr}"
    for attr in ("use_residual", "use_res_connect"):
        if hasattr(blk, attr):
            return bool(getattr(blk, attr)), f"attr:{attr}"
    return None, "unknown"

def sub_convs(blk):
    """Ordered (subname, Conv2d) pairs inside a block, in named_modules order."""
    out = []
    for sub, m in blk.named_modules():
        if isinstance(m, nn.Conv2d):
            out.append((sub, m))
    return out

def block_io_channels(blk):
    convs = sub_convs(blk)
    if not convs:
        return (None, None)
    in_ch = convs[0][1].in_channels
    out_ch = convs[-1][1].out_channels
    return in_ch, out_ch

def block_total_stride(blk):
    s = 1
    for _, m in sub_convs(blk):
        st = _as_pair(m.stride)[0]
        s *= st
    return s

bs_lines = []
bs_lines.append("# BLOCK-LEVEL connectivity map for the Burn forward")
bs_lines.append(f"# model={MODEL_NAME}  input={tuple(x.shape)}")
bs_lines.append("# For each block: class, in->out channels, spatial stride,")
bs_lines.append("# residual-skip add (True/False), ordered sub-convs (k/stride/groups).")
bs_lines.append("=" * 74)

n_blocks = 0
n_residual = 0
block_class_names = Counter()
# running spatial size for a nice trace (input is 128x128, stem stride 2)
for si, stage in enumerate(net.blocks):
    bs_lines.append(f"\n## stage blocks.{si}  ({len(stage)} block(s))")
    for bi, blk in enumerate(stage):
        n_blocks += 1
        cls = type(blk).__name__
        block_class_names[cls] += 1
        in_ch, out_ch = block_io_channels(blk)
        stride = block_total_stride(blk)
        skip, skip_src = block_has_skip(blk)
        # fallback inference if attribute absent
        if skip is None:
            skip = (in_ch == out_ch) and (stride == 1)
            skip_src = "inferred(in==out & stride==1)"
        if skip:
            n_residual += 1
        bs_lines.append(
            f"blocks.{si}.{bi} | {cls} | in={in_ch} -> out={out_ch} | "
            f"stride={stride} | residual_add={skip} [{skip_src}]"
        )
        for sub, m in sub_convs(blk):
            k = _as_pair(m.kernel_size)
            st = _as_pair(m.stride)
            dw = (m.groups == m.in_channels and m.in_channels > 1)
            bs_lines.append(
                f"    - {sub}: Conv2d k={k[0]}x{k[1]} stride={st[0]} "
                f"groups={m.groups} in={m.in_channels} out={m.out_channels} "
                f"depthwise={dw}"
            )

with open(p("blocks_structure.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(bs_lines) + "\n")
print(f"[artifact] blocks_structure.txt written ({n_blocks} blocks)")

# ---------------------------------------------------------------------------
# ARTIFACT 3: mnv4_weights.safetensors + weight_keys.txt
# ---------------------------------------------------------------------------
sd = net.state_dict()
# safetensors requires contiguous tensors; also drop shared-storage issues.
clean_sd = OrderedDict()
for k, v in sd.items():
    t = v.detach().cpu().contiguous().clone()
    clean_sd[k] = t
save_file(clean_sd, p("mnv4_weights.safetensors"))
print(f"[artifact] mnv4_weights.safetensors written ({len(clean_sd)} tensors)")

with open(p("weight_keys.txt"), "w", encoding="utf-8") as f:
    for k, v in sd.items():
        f.write(f"{k}  {tuple(v.shape)}  {str(v.dtype).replace('torch.','')}\n")
print(f"[artifact] weight_keys.txt written ({len(sd)} keys)")

# ---------------------------------------------------------------------------
# ARTIFACT 4: intermediates/ -- per-stage reference activations via hooks.
# ---------------------------------------------------------------------------
captured = OrderedDict()  # label -> np array
hooks = []

def mk_hook(label):
    def _hook(module, inp, out):
        t = out
        if isinstance(t, (tuple, list)):
            t = t[0]
        captured[label] = t.detach().cpu().float().numpy()
    return _hook

# conv_stem output. In timm MNV4, stem is conv_stem then bn1. The "conv_stem
# output" per the spec = output of the conv_stem Conv2d module.
if hasattr(net, "conv_stem"):
    hooks.append(net.conv_stem.register_forward_hook(mk_hook("stem")))

# each top-level block's output
for si, stage in enumerate(net.blocks):
    for bi, blk in enumerate(stage):
        hooks.append(blk.register_forward_hook(mk_hook(f"block_{si}_{bi}")))

# conv_head output (if present)
if hasattr(net, "conv_head") and net.conv_head is not None and not isinstance(net.conv_head, nn.Identity):
    hooks.append(net.conv_head.register_forward_hook(mk_hook("conv_head")))

with torch.no_grad():
    # pooled pre-logits feature + logits, both from the same forward path.
    feats = net.forward_features(x)
    pre_logits = net.forward_head(feats, pre_logits=True)
    logits = net.forward_head(feats, pre_logits=False)

captured["feat"] = pre_logits.detach().cpu().float().numpy()
captured["logits"] = logits.detach().cpu().float().numpy()

for h in hooks:
    h.remove()

# Save every captured intermediate and print shapes.
print("-" * 74)
print("[intermediates] saved reference activations:")
inter_shapes = OrderedDict()
for label, arr in captured.items():
    arr = np.ascontiguousarray(arr.astype(np.float32))
    np.save(p("intermediates", f"{label}.npy"), arr)
    inter_shapes[label] = arr.shape
    print(f"    {label:16s} {str(arr.shape):24s} -> intermediates/{label}.npy")

# ---------------------------------------------------------------------------
# Head / stem config extraction for the SUMMARY.
# ---------------------------------------------------------------------------
stem_conv = net.conv_stem
stem_cfg = conv_cfg(stem_conv)
conv_head_out = None
if hasattr(net, "conv_head") and isinstance(net.conv_head, nn.Conv2d):
    conv_head_out = net.conv_head.out_channels
classifier = net.get_classifier() if hasattr(net, "get_classifier") else net.classifier
clf_in = classifier.in_features
clf_out = classifier.out_features
pre_logits_dim = int(np.prod(inter_shapes["feat"][1:]))

# Distinct block classes
distinct_block_classes = sorted(block_class_names.keys())

# Per-stage width-multiplier channel counts (out channels of each stage's
# last block == the stage output channel count for the 0.5 width model).
stage_out_channels = []
for si, stage in enumerate(net.blocks):
    _, oc = block_io_channels(stage[-1])
    stage_out_channels.append((f"blocks.{si}", oc, len(stage)))

# ---------------------------------------------------------------------------
# ARTIFACT 5: SUMMARY (printed + appended to blueprint.txt tail as a header).
# ---------------------------------------------------------------------------
summary = []
summary.append("=" * 74)
summary.append("SUMMARY  (mobilenetv4_conv_small_050 / width-mult 0.5)")
summary.append("=" * 74)
summary.append(f"pretrained            : {used_pretrained}")
summary.append(f"input                 : {tuple(x.shape)} (reused input_ref.npy)")
summary.append(f"total blocks          : {n_blocks}")
summary.append(f"blocks w/ residual add: {n_residual}")
summary.append(f"distinct block classes: {distinct_block_classes}")
summary.append(f"block class counts    : {dict(block_class_names)}")
summary.append("-" * 74)
summary.append(f"STEM  conv_stem       : {stem_cfg}")
summary.append(f"      bn1            : {bn_cfg(net.bn1)}")
summary.append("-" * 74)
summary.append(f"HEAD  conv_head out   : {conv_head_out}")
if hasattr(net, "norm_head"):
    summary.append(f"      norm_head      : {bn_cfg(net.norm_head)}")
summary.append(f"      classifier     : in={clf_in}, out={clf_out}, bias={classifier.bias is not None}")
summary.append(f"      pre-logits dim : {pre_logits_dim}")
summary.append("-" * 74)
summary.append("WIDTH-MULT (0.5) STAGE OUTPUT CHANNELS (hardcode these):")
summary.append(f"      stem out       : {stem_conv.out_channels}")
for gname, oc, nb in stage_out_channels:
    summary.append(f"      {gname:9s} out : {oc:4d}  ({nb} block(s))")
summary.append(f"      conv_head out  : {conv_head_out}")
summary.append(f"      classifier     : {clf_in} -> {clf_out}")
summary.append("-" * 74)
summary.append("INTERMEDIATE SHAPES:")
for label, shp in inter_shapes.items():
    summary.append(f"      {label:16s} {shp}")
summary.append("=" * 74)

summary_text = "\n".join(summary)
print(summary_text)

# Also stash the summary at the top of blueprint.txt tail for convenience.
with open(p("blueprint.txt"), "a", encoding="utf-8") as f:
    f.write("\n\n")
    f.write(summary_text + "\n")

# ---------------------------------------------------------------------------
# Artifact existence / size confirmation.
# ---------------------------------------------------------------------------
print("\n[artifacts] existence / sizes:")
artifact_files = [
    "blueprint.txt", "blocks_structure.txt",
    "mnv4_weights.safetensors", "weight_keys.txt",
]
for fname in artifact_files:
    fp = p(fname)
    ok = os.path.exists(fp)
    size = os.path.getsize(fp) if ok else -1
    print(f"    {'OK ' if ok else 'MISS'} {fname:28s} {size:>12,d} bytes")
for label in inter_shapes:
    fp = p("intermediates", f"{label}.npy")
    size = os.path.getsize(fp)
    print(f"    OK  intermediates/{label+'.npy':22s} {size:>12,d} bytes {inter_shapes[label]}")

print("=" * 74)
print("DONE.")
