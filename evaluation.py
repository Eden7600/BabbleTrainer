"""Fixed-label, routine-balanced evaluation of the exported semantics."""
import hashlib
import json
import numpy as np
import torch
from torch import nn

import config as C
from model_dual import DualTaskTower, MergedDualEye


def _quotas(capacities, limit):
    quotas = [0] * len(capacities)
    remaining = min(limit, sum(capacities))
    while remaining:
        for i, capacity in enumerate(capacities):
            if quotas[i] < capacity and remaining:
                quotas[i] += 1
                remaining -= 1
    return quotas


def balanced_indices(routines, limit=4000, *, blocks=None, positions=None):
    """Equal routine/block budgets, spread across each block's entire retained time span.

    Only existing dataset indices are selected: input history, partition membership,
    exclusions and continuity still come from the dataset, never from adjacent picks.
    """
    routines = list(map(str, routines))
    if limit < 1:
        raise ValueError("Evaluation sample limit must be positive")
    blocks = [0] * len(routines) if blocks is None else list(blocks)
    positions = list(range(len(routines))) if positions is None else list(positions)
    if len(blocks) != len(routines) or len(positions) != len(routines):
        raise ValueError("Evaluation sample identity lengths differ")
    groups = {}
    for i, (routine, block) in enumerate(zip(routines, blocks)):
        groups.setdefault(routine, {}).setdefault(block, []).append(i)
    ordered = [groups[name] for name in sorted(groups)]
    budgets = _quotas([sum(map(len, group.values())) for group in ordered], limit)
    selected = []
    for group, budget in zip(ordered, budgets):
        rows = [sorted(group[key], key=lambda i: (positions[i], i)) for key in sorted(group)]
        for row, count in zip(rows, _quotas(list(map(len, rows)), budget)):
            if count:
                times = np.asarray([positions[i] for i in row], float)
                targets = ([(times[0] + times[-1]) / 2] if count == 1 else
                           np.linspace(times[0], times[-1], count))
                previous = -1
                for j, target in enumerate(targets):
                    # Nearest available timestamp, reserving enough later rows so
                    # a gap cannot duplicate a sample or consume the remaining budget.
                    lo, hi = previous + 1, len(row) - (count - j)
                    after = min(hi, max(lo, int(np.searchsorted(times, target))))
                    before = max(lo, after - 1)
                    offset = before if abs(times[before] - target) < abs(times[after] - target) else after
                    selected.append(row[offset])
                    previous = offset
    return selected


def paired_sample_indices(paired, limit=4000, *, ensure_expression_cohorts=False):
    routines = [str(paired.base.routine[ic]) for _, _, ic, _ in paired.pairs]
    blocks = [(bid, int(paired.base.block[ic])) for bid, _, ic, _ in paired.pairs]
    times = [paired.base.timestamp[ic] for _, _, ic, _ in paired.pairs]
    chosen = balanced_indices(routines, limit, blocks=blocks, positions=times)
    initial = set(chosen)
    required = set()
    if ensure_expression_cohorts:
        # A short reviewed endpoint must survive the sampling budget. Use the same
        # physical-eye order and final masks as the merged evaluator, without images.
        canonical = [ic for _, _, ic, _ in paired.pairs]
        mirror = [im for _, _, _, im in paired.pairs]
        left, right = paired.base.label[canonical], paired.base.label[mirror]
        lm, rm = paired.base.label_mask[canonical], paired.base.label_mask[mirror]
        if C.SWAP_OUTPUT_SIDES:
            left, right, lm, rm = right, left, rm, lm
        targets = np.concatenate([right, left] if C.OUTPUT_SIDE_ORDER[0] == "right" else [left, right], axis=1)
        masks = np.concatenate([rm, lm] if C.OUTPUT_SIDE_ORDER[0] == "right" else [lm, rm], axis=1)
        groups = {}
        for channel in (2, 3, 4, 5, 8, 9, 10, 11):
            for i in np.flatnonzero(masks[:, channel]):
                value = targets[i, channel]
                cohort = "inactive" if value <= .2 else "active" if value >= .8 else "intermediate"
                groups.setdefault((routines[i], channel, cohort), []).append(int(i))
        for rows in groups.values():
            present = [i for i in rows if i in initial]
            if present:
                rows = present
            middle = (min(times[i] for i in rows) + max(times[i] for i in rows)) / 2
            required.add(min(rows, key=lambda i: (abs(times[i] - middle), i)))
        if len(required) > limit:
            raise ValueError("Evaluation budget cannot cover all reviewed expression cohorts")
    added = sorted(required - initial)
    excess = max(0, len(chosen) + len(added) - limit)
    if excess:
        by_block = {}
        for i in chosen:
            by_block.setdefault(blocks[i], []).append(i)
        edges = {i for rows in by_block.values() for i in (rows[0], rows[-1])}
        removable = [i for i in chosen if i not in required and i not in edges]
        removable += [i for i in chosen if i not in required and i in edges]
        removed = set(removable[:excess])
        chosen = [i for i in chosen if i not in removed]
    return chosen + added


def sampling_report(identities, indices, limit, timestamps=None):
    """Identity digest makes differing evaluation populations visible in comparisons."""
    selected = [identities[i] for i in indices]
    encoded = json.dumps(selected, separators=(",", ":")).encode()
    groups = {}
    for identity in identities:
        key, pos = tuple(identity[:-1]), identity[-1]
        item = groups.setdefault(key, dict(eligible=0, selected=0, first_record=pos, last_record=pos))
        item["eligible"] += 1
        item["first_record"] = min(item["first_record"], pos)
        item["last_record"] = max(item["last_record"], pos)
    for identity in selected:
        item = groups[tuple(identity[:-1])]
        item["selected"] += 1
        item["selected_first_record"] = min(item.get("selected_first_record", identity[-1]), identity[-1])
        item["selected_last_record"] = max(item.get("selected_last_record", identity[-1]), identity[-1])
    if timestamps is not None:
        for i in indices:
            item = groups[tuple(identities[i][:-1])]
            time = float(timestamps[i])
            item["selected_first_timestamp_ms"] = min(item.get("selected_first_timestamp_ms", time), time)
            item["selected_last_timestamp_ms"] = max(item.get("selected_last_timestamp_ms", time), time)
    return dict(method="routine_block_time_spread_v1", budget=limit, eligible=len(identities),
                selected=len(indices), identity_sha256=hashlib.sha256(encoded).hexdigest(),
                blocks=[dict(identity=list(key), **value) for key, value in sorted(groups.items())])


def paired_inputs(ds, indices):
    inputs, targets, masks = [], [], []
    for i in indices:
        left, right, ll, rl, lm, rm = ds[i]
        x = torch.empty((8, C.DUAL_INPUT_SIZE, C.DUAL_INPUT_SIZE))
        x[C.EXPORT_NOFLIP_CH] = left
        x[C.EXPORT_FLIP_CH] = torch.flip(right, [-1])
        rl = rl.clone()
        rl[1] = 1 - rl[1]
        if C.SWAP_OUTPUT_SIDES:
            ll, rl, lm, rm = rl, ll, rm, lm
        inputs.append(x)
        targets.append(torch.cat([rl, ll] if C.OUTPUT_SIDE_ORDER[0] == "right" else [ll, rl]))
        masks.append(torch.cat([rm, lm] if C.OUTPUT_SIDE_ORDER[0] == "right" else [lm, rm]))
    if not inputs:
        raise ValueError("No paired validation images; repeat calibration")
    return torch.stack(inputs), torch.stack(targets), torch.stack(masks)


def summarize(pred, target, mask, routines, names):
    pred, target, mask = np.asarray(pred), np.asarray(target), np.asarray(mask, bool)
    if pred.shape != target.shape or mask.shape != target.shape or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("Invalid or non-finite evaluation tensors")
    routines = np.asarray(routines)
    result = {"channels": {}, "routines": {}, "cohorts": {},
              "cohort_definition": {"inactive_max": .2, "active_min": .8,
                                    "aggregation": "equal_cohorts_then_equal_channels; equal_routines_within_cohort"},
              "event_metrics": "unavailable_without_reviewed_events"}
    for name in map(str, sorted(set(routines))):
        rows = routines == name
        result["routines"][name] = {}
        for c, channel in enumerate(names):
            use = rows & mask[:, c]
            err = np.abs(pred[use, c] - target[use, c])
            result["routines"][name][channel] = dict(count=int(use.sum()),
                masked=int(rows.sum() - use.sum()), mae=float(err.mean()) if len(err) else None)
    for c, channel in enumerate(names):
        stats = [r[channel]["mae"] for r in result["routines"].values() if r[channel]["mae"] is not None]
        result["channels"][channel] = dict(mae=float(np.mean(stats)) if stats else None,
                                           count=int(mask[:, c].sum()), masked=int((~mask[:, c]).sum()))
        if channel.endswith(("EyeY", "EyeX")) or channel.startswith("combined"):
            continue
        result["cohorts"][channel] = {}
        for cohort, rows in (("inactive", target[:, c] <= .2), ("active", target[:, c] >= .8),
                             ("intermediate", (target[:, c] > .2) & (target[:, c] < .8))):
            use = rows & mask[:, c]
            by_routine = {}
            for routine in map(str, sorted(set(routines))):
                selected = use & (routines == routine)
                errors = np.abs(pred[selected, c] - target[selected, c])
                if len(errors):
                    by_routine[routine] = dict(count=int(len(errors)), mae=float(errors.mean()))
            result["cohorts"][channel][cohort] = dict(count=int(use.sum()), routines=by_routine,
                mae=float(np.mean([r["mae"] for r in by_routine.values()])) if by_routine else None)
        values = [v["mae"] for v in result["cohorts"][channel].values() if v["mae"] is not None]
        result["channels"][channel]["balanced_mae"] = float(np.mean(values)) if values else None
    return result


def score(report, kind="expression"):
    values = [v.get("balanced_mae", v["mae"]) for name, v in report["channels"].items()
              if v["mae"] is not None and ((name.endswith(("EyeY", "EyeX"))) == (kind == "gaze"))
              and not name.startswith("combined")]
    if not values:
        raise ValueError(f"No valid {kind} evaluation labels")
    return float(np.mean(values))


def hold_statistics(predictions, timestamps_ms, channel_mask, names, expected_count, duration_ms):
    """Population SD on an independently annotated hold; SD is not mean accuracy."""
    predictions = np.asarray(predictions, float)
    timestamps_ms = np.asarray(timestamps_ms, float)
    if predictions.shape != (len(timestamps_ms), len(names)) or duration_ms <= 0 or expected_count < len(predictions):
        raise ValueError("Invalid reviewed hold samples")
    channels = {}
    for c, name in enumerate(names):
        if not channel_mask[c]:
            continue
        values = predictions[:, c]
        finite = np.isfinite(values)
        values = values[finite]
        gaze = name.endswith(("EyeX", "EyeY"))
        channels[name] = dict(standard_deviation=(float(values.std()) * (2 * C.GAZE_RANGE_DEG if gaze else 1)
                                                   if len(values) >= 2 else None),
            units="degrees" if gaze else "normalized_value", count=len(values),
            invalid=int((~finite).sum()), missing=int(expected_count - len(predictions)),
            clipped=int(np.sum((values <= 0) | (values >= 1))))
    return dict(duration_seconds=duration_ms / 1000,
                sampling_rate_hz=(expected_count - 1) / (duration_ms / 1000), channels=channels)


OPEN_LID_EXTRA_TOLERANCE = .03


def regression_guard(baseline, candidate, kind="expression"):
    """Never let inactive examples hide active regression (or vice versa)."""
    guard = .5 / (2 * C.GAZE_RANGE_DEG) if kind == "gaze" else .02
    old_sampling, new_sampling = baseline.get("evaluation_sampling"), candidate.get("evaluation_sampling")
    if old_sampling != new_sampling:
        return False, "Incomparable evaluation sample identities"
    for name, old in baseline["channels"].items():
        if name.startswith("combined") or (name.endswith(("EyeY", "EyeX"))) != (kind == "gaze"):
            continue
        new = candidate["channels"].get(name)
        if new is None or old["count"] != new["count"] or old["masked"] != new["masked"]:
            return False, f"Incomparable label coverage: {name}"
        if kind == "expression" and ("cohorts" in baseline or "cohorts" in candidate):
            old_cohorts = baseline.get("cohorts", {}).get(name)
            new_cohorts = candidate.get("cohorts", {}).get(name)
            if old_cohorts is None or new_cohorts is None or old_cohorts.keys() != new_cohorts.keys():
                return False, f"Incomparable activity cohorts: {name}"
            if old["count"] and any(old_cohorts[endpoint]["count"] == 0 for endpoint in ("inactive", "active")):
                return False, f"Incomplete validation endpoints: {name}; retain baseline or review both endpoints"
            for cohort, before in old_cohorts.items():
                after = new_cohorts[cohort]
                coverage = lambda value: {r: v["count"] for r, v in value["routines"].items()}
                if before["count"] != after["count"] or coverage(before) != coverage(after):
                    return False, f"Incomparable cohort coverage: {name}/{cohort}"
                # Open-lid labels are confirmed where the pretrained model already reads the eye as open, so
                # the baseline is near-perfect on that cohort by construction; allow a little more drift there.
                limit = guard + (OPEN_LID_EXTRA_TOLERANCE if name.endswith("EyeLid") and cohort == "inactive" else 0)
                if before["mae"] is not None and (after["mae"] is None or after["mae"] > before["mae"] + limit):
                    return False, f"Activity-cohort regression guard exceeded: {name}/{cohort}"
        elif old["mae"] is not None and (new["mae"] is None or new["mae"] > old["mae"] + guard):
            return False, f"Per-channel regression guard exceeded: {name}"
    return True, "Common coverage and regression guardrails passed"


def choose_candidate(baseline, candidate, kind="expression"):
    passed, reason = regression_guard(baseline, candidate, kind)
    if not passed:
        return False, reason
    if score(candidate, kind) >= score(baseline, kind):
        return False, "Baseline wins: no aggregate improvement (ties retain baseline)"
    return True, "Candidate improves the balanced score and passes regression guardrails"


class _NeutralGaze(nn.Module):
    def forward(self, x):
        return x.new_full((x.shape[0], 2), .5)


@torch.no_grad()
def evaluate_expression(net, consensus, paired, device, limit=4000):
    merged = MergedDualEye(DualTaskTower(net, _NeutralGaze()), consensus).to(device).eval()
    report = evaluate_merged(merged, paired, device, limit)
    return report


@torch.no_grad()
def evaluate_merged(merged, paired, device="cpu", limit=4000):
    routines = np.asarray([paired.base.routine[ic] for _, _, ic, _ in paired.pairs])
    indices = paired_sample_indices(paired, limit, ensure_expression_cohorts=True)
    scored_count = len(indices)
    sampling = sampling_report([(str(routines[i]), int(bid), int(paired.base.block[ic]), int(pos))
                                for i, (bid, pos, ic, _) in enumerate(paired.pairs)], indices, limit,
                               [paired.base.timestamp[ic] for _, _, ic, _ in paired.pairs])
    # Reviewed intervals must be evaluated completely; a sampling budget cannot hide missing data.
    relevant_holds = []
    for hold in paired.base.reviewed_holds:
        matches = [i for i, (bid, pos, _, _) in enumerate(paired.pairs)
                   if bid == hold["bin_id"] and hold["start_record"] <= pos < hold["end_record_exclusive"]]
        if matches:
            relevant_holds.append(hold)
            indices.extend(i for i in matches if i not in indices)
    preds, targets, masks = [], [], []
    for start in range(0, len(indices), 64):
        x, y, m = paired_inputs(paired, indices[start:start + 64])
        pred = merged(x.to(device)).float().cpu().numpy()
        preds.append(pred[:, :12]); targets.append(y.numpy()); masks.append(m.numpy())
    if not preds:
        raise ValueError("Validation contains no usable paired records")
    predictions, targets, masks = np.concatenate(preds), np.concatenate(targets), np.concatenate(masks)
    report = summarize(predictions[:scored_count], targets[:scored_count], masks[:scored_count],
                       routines[indices[:scored_count]], C.BLENDSHAPE_NAMES)
    report["evaluation_sampling"] = sampling
    sampling["expression_cohorts_required"] = True
    report["gaze_axis_mae_degrees"] = {k: (None if v["mae"] is None else v["mae"] * 2 * C.GAZE_RANGE_DEG)
                                        for k, v in report["channels"].items() if k.endswith(("EyeX", "EyeY"))}
    report["clipped_boundary_values"] = int(np.sum((predictions <= 0) | (predictions >= 1)))
    holds = []
    for hold in relevant_holds:
        rows = [j for j, i in enumerate(indices) if paired.pairs[i][0] == hold["bin_id"]
                and hold["start_record"] <= paired.pairs[i][1] < hold["end_record_exclusive"]]
        start = int(paired.base.bin_rec_start[hold["bin_id"]])
        first = paired.base.rec_timestamp[start + hold["start_record"]]
        last = paired.base.rec_timestamp[start + hold["end_record_exclusive"] - 1]
        if last > first:
            times = [paired.base.rec_timestamp[start + paired.pairs[indices[j]][1]] for j in rows]
            stats = hold_statistics(predictions[rows], times, hold["channel_mask"], C.BLENDSHAPE_NAMES,
                                    hold["end_record_exclusive"] - hold["start_record"], last - first)
            holds.append(dict(start_record=hold["start_record"], end_record_exclusive=hold["end_record_exclusive"], **stats))
    report["steady_hold_jitter"] = dict(status="measured_reviewed_holds" if holds else "unavailable_without_reviewed_holds",
                                        metric="population_standard_deviation", intervals=holds,
                                        holds_omitted_by_user_exclusions=paired.base.omitted_reviewed_holds)
    report["fusion"] = dict(channels=merged.fuse_channels, source=merged.fuse_source,
                             consensus=merged.consensus is not None)
    return report


@torch.no_grad()
def evaluate_gaze_report(net, ds, device, limit=4000):
    indices = balanced_indices(ds.routine, limit,
        blocks=list(zip(ds.bin_id.tolist(), ds.block.tolist(), ds.side.tolist())), positions=ds.timestamp)
    predictions, targets, masks = [], [], []
    for start in range(0, len(indices), 128):
        batch = [ds[i] for i in indices[start:start + 128]]
        predictions.append(net(torch.stack([b[0] for b in batch]).to(device)).cpu().numpy())
        targets.append(torch.stack([b[1] for b in batch]).numpy()[:, :2])
        masks.append(torch.stack([b[3] for b in batch]).numpy()[:, :2])
    if not predictions:
        raise ValueError("No gaze validation inputs")
    predictions, targets, masks = np.concatenate(predictions), np.concatenate(targets), np.concatenate(masks)
    out = np.zeros((len(indices), 4), np.float32)
    truth = np.zeros_like(out)
    valid = np.zeros_like(out, bool)
    for row, i in enumerate(indices):
        right = ds.side[i] == C.TRAIN_MIRROR_BIN_SIDE
        if right:
            predictions[row, 1] = 1 - predictions[row, 1]
            targets[row, 1] = 1 - targets[row, 1]
        if C.SWAP_OUTPUT_SIDES:
            right = not right
        offset = 0 if right else 2
        out[row, offset:offset + 2] = predictions[row]
        truth[row, offset:offset + 2] = targets[row]
        valid[row, offset:offset + 2] = masks[row]
    report = summarize(out, truth, valid, ds.routine[indices],
                       ["rightEyeY", "rightEyeX", "leftEyeY", "leftEyeX"])
    report["evaluation_sampling"] = sampling_report([(str(ds.routine[i]), int(ds.bin_id[i]),
        int(ds.block[i]), int(ds.side[i]), int(ds.pos[i])) for i in range(len(ds))], indices, limit, ds.timestamp)
    return report
