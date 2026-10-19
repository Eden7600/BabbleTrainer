"""Append the headset-worn outputs to an existing eye ONNX without retraining.

  python add_presence.py eye.onnx eye_presence.onnx [--presence runs/seg/seg_presence.pt]

Every original output is kept bit-for-bit; rightEyePresent/leftEyePresent are appended and named in metadata.
New trainings get the same outputs automatically when presence_model.pt sits beside the trainer.
"""
import argparse
import io
import json

import numpy as np
import onnx
import torch
from onnx import helper

import config as C
from model_dual import load_presence, presence_outputs


class _PresenceOnly(torch.nn.Module):
    def __init__(self, presence):
        super().__init__()
        self.presence = presence

    def forward(self, x):
        return presence_outputs(self.presence, x)


def add_presence(src, dst, presence_ckpt):
    base = onnx.load(src)
    meta = {p.key: p for p in base.metadata_props}
    names = json.loads(meta["blendshape_names"].value)
    if names[-len(C.PRESENCE_NAMES):] == C.PRESENCE_NAMES:
        raise ValueError(f"{src} already has presence outputs")
    graph = base.graph
    in_name, out_name = graph.input[0].name, graph.output[0].name
    opset = max(o.version for o in base.opset_import if o.domain in ("", "ai.onnx"))

    presence = load_presence(presence_ckpt)
    buf = io.BytesIO()
    kw = dict(input_names=[in_name], output_names=["presence"], opset_version=opset,
              dynamic_axes={in_name: {0: "batch"}, "presence": {0: "batch"}})
    dummy = torch.zeros(1, 2 * C.TEMPORAL_FRAMES, C.DUAL_INPUT_SIZE, C.DUAL_INPUT_SIZE)
    try:
        torch.onnx.export(_PresenceOnly(presence).eval(), dummy, buf, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(_PresenceOnly(presence).eval(), dummy, buf, **kw)
    sub = onnx.compose.add_prefix(onnx.load_from_string(buf.getvalue()), "presence/")
    prefixed_in = "presence/" + in_name
    for node in sub.graph.node:  # the presence branch reads the eye model's own input tensor
        node.input[:] = [in_name if i == prefixed_in else i for i in node.input]

    base_out = out_name + "/without_presence"
    for node in graph.node:
        node.output[:] = [base_out if o == out_name else o for o in node.output]
        node.input[:] = [base_out if i == out_name else i for i in node.input]
    graph.node.extend(sub.graph.node)
    graph.initializer.extend(sub.graph.initializer)
    graph.node.append(helper.make_node("Concat", [base_out, "presence/presence"], [out_name], axis=1))
    dims = graph.output[0].type.tensor_type.shape.dim
    if len(dims) == 2 and dims[1].HasField("dim_value"):
        dims[1].dim_value += len(C.PRESENCE_NAMES)
    meta["blendshape_names"].value = json.dumps(names + C.PRESENCE_NAMES)
    onnx.checker.check_model(base)
    onnx.save(base, dst)

    # Original columns must be unchanged and the appended ones must match PyTorch.
    import onnxruntime as ort
    x = np.random.default_rng(0).random((2, 2 * C.TEMPORAL_FRAMES, C.DUAL_INPUT_SIZE, C.DUAL_INPUT_SIZE), np.float32)
    old = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, {in_name: x})[0]
    new = ort.InferenceSession(dst, providers=["CPUExecutionProvider"]).run(None, {in_name: x})[0]
    with torch.no_grad():
        ref = _PresenceOnly(presence)(torch.from_numpy(x)).numpy()
    assert new.shape == (2, len(names) + 2), new.shape
    assert np.array_equal(new[:, :len(names)], old), "original outputs changed"
    assert np.abs(new[:, len(names):] - ref).max() < 1e-4, "presence outputs disagree with PyTorch"
    return names + C.PRESENCE_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--presence", default="runs/seg/seg_presence.pt")
    args = ap.parse_args()
    print("outputs:", add_presence(args.src, args.dst, args.presence))


if __name__ == "__main__":
    main()
