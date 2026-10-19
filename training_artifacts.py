"""Bounded parity fixtures; the desktop verifies ONNX with its existing runtime."""
import json
from pathlib import Path
import numpy as np
import torch
import config as C
from capture_manifest import sha256
from evaluation import paired_sample_indices, paired_inputs


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@torch.no_grad()
def write_artifacts(model_path, merged, paired, run_id, report):
    model_path = Path(model_path)
    folder = model_path.parent
    x, _, _ = paired_inputs(paired, paired_sample_indices(paired, 32))
    inputs = x.numpy().astype("<f4")
    expected = merged.cpu().eval()(x).numpy().astype("<f4")
    names = C.output_names(merged.consensus is not None, getattr(merged, "presence", None) is not None)
    if not 1 <= len(inputs) <= 32 or expected.shape != (len(inputs), len(names)):
        raise ValueError("Invalid parity fixture shape")
    if not np.isfinite(inputs).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite parity fixture")
    input_path, output_path = folder / "parity-input.f32", folder / "parity-output.f32"
    inputs.tofile(input_path); expected.tofile(output_path)
    report_path = folder / "training-report.json"
    write_json(report_path, report)
    def entry(path, **extra):
        return dict(path=path.name, sha256=sha256(path), **extra)
    manifest = dict(schema_version=1, run_id=run_id, model=entry(model_path),
        input=entry(input_path, name="input", shape=list(inputs.shape)),
        output=entry(output_path, name="output", shape=list(expected.shape)),
        output_names=names, provenance="held_out" if paired.base.independent else "non_independent",
        report=entry(report_path))
    write_json(folder / "training-artifact.json", manifest)
    return manifest
