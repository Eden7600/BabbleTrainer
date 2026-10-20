import json

import numpy as np
import torch

import config as C
from model_dual import build_eye_tower, MergedDualEye, build_consensus_heads, expr_feat_dim, gaze_net_for


def _load_consensus(ec, tower):
    if not (isinstance(ec, dict) and ec.get("consensus") and C.DUAL_CONSENSUS_HEAD):
        return None
    heads = build_consensus_heads(expr_feat_dim(tower.expr))
    heads.load_state_dict(ec["consensus"])
    return heads.eval()


def load_merged_tower(gaze_ckpt, expr_ckpt, device="cpu"):
    gc = torch.load(gaze_ckpt, map_location=device, weights_only=True)
    ec = torch.load(expr_ckpt, map_location=device, weights_only=True)
    backbone = ec.get("backbone", C.BACKBONE)
    tower = build_eye_tower(backbone=backbone, pretrained=False)
    expr_sd = ec["teacher"] if isinstance(ec, dict) and "teacher" in ec else ec
    gaze_sd = gc.get("gaze", gc.get("student", gc)) if isinstance(gc, dict) else gc
    assert set(expr_sd) == set(tower.expr.state_dict()), "expr ckpt keys don't match tower.expr"
    tower.expr.load_state_dict(expr_sd)
    tower.gaze = gaze_net_for(gaze_sd)   # the selected checkpoint may be the max-pooled baseline
    tower.gaze.load_state_dict(gaze_sd)
    consensus = _load_consensus(ec, tower)
    return tower.to(device).eval(), backbone, (consensus.to(device) if consensus is not None else None)


def export(gaze_ckpt, expr_ckpt, out_path, check=False, presence=None):
    """Merge the gaze + expr checkpoints (and optional presence module) and export the single ONNX."""
    tower, backbone, consensus = load_merged_tower(gaze_ckpt, expr_ckpt, "cpu")
    _export_tower(tower, backbone, out_path, check, consensus, presence)


def _export_tower(tower, backbone, out_path, check=False, consensus=None, presence=None):
    import onnx
    device = "cpu"
    merged = MergedDualEye(tower, consensus, presence).to(device).eval()
    names = C.output_names(consensus is not None, presence is not None, getattr(presence, "pupil", False))

    S, IN = C.DUAL_INPUT_SIZE, 2 * C.TEMPORAL_FRAMES
    dummy = torch.randn(1, IN, S, S, device=device)
    kw = dict(export_params=True, opset_version=18, do_constant_folding=True,
              input_names=["input"], output_names=["output"],
              dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}})
    try:
        torch.onnx.export(merged, dummy, out_path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(merged, dummy, out_path, **kw)

    model = onnx.load(out_path)
    for key, value in [
        ("blendshape_names", json.dumps(names)),
        ("input_layout", f"interleaved_LR_t0..t{C.TEMPORAL_FRAMES - 1}_{IN}x{S}x{S}_gray01"),
        ("gaze_range_deg", str(C.GAZE_RANGE_DEG)),
    ]:
        mp = model.metadata_props.add()
        mp.key, mp.value = key, value
    onnx.save(model, out_path)

    print(f"Exported {out_path}")
    print(f"  backbone={backbone}  input=(B,{IN},{S},{S})  output=(B,{len(names)})"
          f"{'  +consensus(15)' if consensus is not None else ''}")
    print(f"  blendshape_names={names}")
    if merged.fuse_channels:
        fused = [C.DUAL_PER_EYE[C.EXPR_SLICE[0] + c] for c in merged.fuse_channels]
        print(f"  FUSED (both eyes share one value, source={merged.fuse_source}): {fused}")
    if check:
        _check(merged, out_path, dummy, names)


def _check(merged, out_path, dummy, names):
    import onnxruntime as ort
    with torch.no_grad():
        ref = merged(dummy).numpy()
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got = sess.run(["output"], {"input": dummy.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    ort_names = json.loads(sess.get_modelmeta().custom_metadata_map["blendshape_names"])
    print(f"  ORT round-trip: shape={got.shape} max|torch-ort|={diff:.2e} names_ok={ort_names == names}")
    assert got.shape == ref.shape and diff < 1e-4, "ONNX output disagrees with PyTorch"
    assert got.shape[1] == len(names), f"output width {got.shape[1]} != {len(names)} names"
    got2 = sess.run(["output"], {"input": torch.cat([dummy, dummy], 0).numpy()})[0]
    assert got2.shape[0] == 2, "dynamic batch axis broken"
    print("  dynamic batch OK")
