"""Evaluate a frozen, reviewed replay comparison; never trains or promotes a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def finite(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    return float(value)


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def evaluate(protocol, results):
    if protocol.get("schema_version") != 1 or results.get("schema_version") != 1:
        raise ValueError("Unsupported experiment schema")
    if results.get("protocol_sha256") != digest(protocol):
        raise ValueError("Results do not match the frozen protocol")
    if protocol.get("status") != "frozen":
        raise ValueError("The protocol is a draft; freeze it before collecting comparison results")
    for field in ("target_metric", "hardware", "backend", "dataset_sha256", "baseline_sha256", "candidate_sha256", "license_review"):
        if not isinstance(protocol.get(field), str) or not protocol[field].strip():
            raise ValueError(f"Protocol requires {field}")
    for field in ("dataset_sha256", "baseline_sha256", "candidate_sha256"):
        value = protocol[field]
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
            raise ValueError(f"{field} must identify the exact artifact")
    if protocol["baseline_sha256"] == protocol["candidate_sha256"]:
        raise ValueError("Baseline and candidate must identify different models")
    if results.get("hardware") != protocol["hardware"] or results.get("backend") != protocol["backend"]:
        raise ValueError("Comparison hardware/backend differs from the frozen protocol")
    clip_ids = protocol.get("clip_ids", [])
    clips = results.get("clips", [])
    if not clip_ids or len(clip_ids) != len(set(clip_ids)):
        raise ValueError("Protocol requires unique reviewed clip IDs")
    if len(clips) != len(clip_ids) or {clip.get("clip_id") for clip in clips} != set(clip_ids):
        raise ValueError("Results must include every frozen clip exactly once")
    minimum_units = protocol.get("minimum_independent_units", 10)
    if isinstance(minimum_units, bool) or not isinstance(minimum_units, int) or minimum_units < 10:
        raise ValueError("At least ten independent units are required by this initial engineering protocol")
    channel_names = protocol.get("expression_channels", [])
    axis_names = protocol.get("gaze_axes", [])
    if not channel_names or not axis_names or len(set(channel_names)) != len(channel_names) or len(set(axis_names)) != len(axis_names):
        raise ValueError("Declare unique supported expression channels and gaze axes")

    groups = {}
    baseline_targets, candidate_targets = [], []
    channel_errors = {key: [[], []] for key in channel_names}
    axis_errors = {key: [[], []] for key in axis_names}
    invalid = [0, 0]
    total_samples = 0
    lag_regressions, cost_regressions, frame_budgets = [], [], []
    coverage = []
    for clip in clips:
        clip_id = clip["clip_id"]
        if clip.get("reviewed_labels") is not True or clip.get("blind_final") is not True:
            raise ValueError(f"{clip_id}: reviewed blind-final labels are required")
        unit = clip.get("independent_unit_id")
        if not isinstance(unit, str) or not unit:
            raise ValueError(f"{clip_id}: declare the independent clip/session grouping")
        period = finite(clip.get("measured_frame_period_ms"), "measured frame period", minimum=0.001)
        count = clip.get("sample_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{clip_id}: sample_count must be positive")
        total_samples += count
        pair, costs, lags = [], [], []
        for side, name in enumerate(("baseline", "candidate")):
            values = clip.get(name, {})
            pair.append(finite(values.get("target_error"), f"{clip_id}/{name}/target_error"))
            bad = values.get("invalid_count")
            if isinstance(bad, bool) or not isinstance(bad, int) or not 0 <= bad <= count:
                raise ValueError(f"{clip_id}/{name}: invalid_count is inconsistent")
            invalid[side] += bad
            expression, gaze = values.get("expression_mae", {}), values.get("gaze_axis_mae_deg", {})
            if set(expression) != set(channel_names) or set(gaze) != set(axis_names):
                raise ValueError(f"{clip_id}/{name}: every declared channel/axis needs the same fixed-label coverage")
            for key in channel_names:
                channel_errors[key][side].append(finite(expression[key], key, maximum=1))
            for key in axis_names:
                axis_errors[key][side].append(finite(gaze[key], key))
            lags.append(finite(values.get("response_lag_ms"), "response lag"))
            times = values.get("inference_ms", [])
            if not isinstance(times, list) or len(times) < 100:
                raise ValueError(f"{clip_id}/{name}: at least 100 inference timings are needed")
            costs.append(percentile([finite(t, "inference time", minimum=0.000001) for t in times], .95))
        baseline_targets.append(pair[0])
        candidate_targets.append(pair[1])
        groups.setdefault(unit, []).append(pair)
        lag_regressions.append((lags[1] - lags[0]) / period)
        cost_regressions.append(costs[1] / costs[0])
        frame_budgets.append(costs[1] / period)
        coverage.append({"clip_id": clip_id, "independent_unit_id": unit, "samples": count,
                         "frame_period_ms": period, "baseline_p95_ms": costs[0], "candidate_p95_ms": costs[1]})

    # Resample independent groups, not correlated frames or multiple clips from one session.
    unit_pairs = [(statistics.mean(pair[0] for pair in values), statistics.mean(pair[1] for pair in values))
                  for values in groups.values()]
    unit_deltas = [pair[0] - pair[1] for pair in unit_pairs]
    rng = random.Random(7600)
    draws = [statistics.mean(rng.choices(unit_deltas, k=len(unit_deltas))) for _ in range(10000)]
    interval = [percentile(draws, .025), percentile(draws, .975)]
    baseline = statistics.mean(pair[0] for pair in unit_pairs)
    candidate = statistics.mean(pair[1] for pair in unit_pairs)
    improvement = (baseline - candidate) / baseline if baseline > 0 else None
    expression_regressions = {key: statistics.mean(pair[1]) - statistics.mean(pair[0]) for key, pair in channel_errors.items()}
    gaze_regressions = {key: statistics.mean(pair[1]) - statistics.mean(pair[0]) for key, pair in axis_errors.items()}
    gates = {
        "enough_independent_units": len(groups) >= minimum_units,
        "target_improvement_at_least_10_percent": improvement is not None and improvement >= .10,
        "paired_95_percent_interval_excludes_no_gain": interval[0] > 0,
        "expression_regression_at_most_002": all(v <= .02 for v in expression_regressions.values()),
        "gaze_axis_regression_at_most_half_degree": all(v <= .5 for v in gaze_regressions.values()),
        "invalid_rate_regression_at_most_one_percentage_point": (invalid[1] - invalid[0]) / total_samples <= .01,
        "added_lag_at_most_one_frame": max(lag_regressions) <= 1,
        "p95_cost_at_most_110_percent": max(cost_regressions) <= 1.10,
        "p95_below_measured_frame_period": max(frame_budgets) < 1,
    }
    return {
        "schema_version": 1, "protocol_sha256": digest(protocol), "target_metric": protocol["target_metric"],
        "decision": "eligible_for_review" if all(gates.values()) else "retain_baseline",
        "automatic_promotion": False, "gates": gates, "baseline_target_error": baseline,
        "candidate_target_error": candidate, "relative_improvement": improvement,
        "paired_group_bootstrap_95_percent_interval": interval, "independent_units": len(groups),
        "expression_mae_regression": expression_regressions, "gaze_axis_mae_regression_deg": gaze_regressions,
        "invalid_rate": {"baseline": invalid[0] / total_samples, "candidate": invalid[1] / total_samples},
        "coverage": coverage,
        "limitations": "Engineering gates, not established perceptual thresholds. Supplied labels and independence require human review; parity alone does not establish accuracy.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(json.loads(args.protocol.read_text(encoding="utf-8")), json.loads(args.results.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(report["decision"])


if __name__ == "__main__":
    main()
