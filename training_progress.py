"""Versioned progress for the desktop; ordinary logs may still share stdout."""
import json
import math

_run_id = None
_stage = "preparing"


def configure(run_id=None):
    global _run_id
    _run_id = run_id


def emit(stage, completed_steps=None, total_steps=None, metrics=None, **fields):
    global _stage
    _stage = stage
    if _run_id is None:
        return
    event = dict(schema_version=1, run_id=_run_id, type="progress", stage=stage)
    if completed_steps is not None:
        if not 0 <= completed_steps <= total_steps:
            raise ValueError("Invalid progress counters")
        event.update(completed_steps=completed_steps, total_steps=total_steps)
    if metrics:
        if any(not math.isfinite(float(v)) for v in metrics.values()):
            raise ValueError("Training produced a non-finite metric")
        event["metrics"] = metrics
    event.update(fields)
    print(json.dumps(event, allow_nan=False, separators=(",", ":")), flush=True)


def structured():
    return _run_id is not None


def fail(message):
    emit(_stage, type="failure", message=message)
