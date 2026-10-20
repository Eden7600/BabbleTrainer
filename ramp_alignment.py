"""Labels derived from the pretrained expression model.

Ramp blocks label each frame from the schedule the overlay shows, but people do not move in step with it:
they start late, then anticipate the steady rhythm and lead it (measured: -175 to -450 ms). The offset is
estimated once per capture as the shift that best correlates the pretrained model's output with the
schedule, then every ramp label is re-read at (t - lag) within its own block and eye.

Prompted open-eye blocks carry no lid label because people still blink during them. confirm_open
supplies the "open" lid endpoint where the pretrained model sees the eye clearly open.
"""
import numpy as np

# Per-eye label column each ramp routine drives: [Y, X, lid closure, widen, squint, brow].
RAMP_CHANNEL = {"lidramp": 2, "widenramp": 3, "squintramp": 4, "browramp": 5}
LAGS_MS = np.arange(-800, 801, 25)   # negative: the face leads the cue


def _groups(ds, rows):
    keys = np.stack([ds.bin_id[rows], ds.block[rows], ds.side[rows]], axis=1)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    for g in range(inverse.max() + 1 if len(rows) else 0):
        member = rows[inverse.ravel() == g]
        yield member[np.argsort(ds.timestamp[member], kind="stable")]


def shifted(ds, rows, lag_ms):
    """Ramp labels re-read at t - lag. Rows are indices into ds; returns an array aligned with rows."""
    out = ds.label[rows].copy()
    where = {r: i for i, r in enumerate(rows)}
    for member in _groups(ds, rows):
        c = RAMP_CHANNEL[str(ds.routine[member[0]])]
        t = ds.timestamp[member].astype(np.float64)
        out[[where[r] for r in member], c] = np.interp(t - lag_ms, t, ds.label[member, c])
    return out


def estimate_lag(ds, rows, predictions, lags_ms=LAGS_MS):
    """Lag (ms) maximizing the sample-weighted per-block correlation; predictions align with rows (N x 6)."""
    where = {r: i for i, r in enumerate(rows)}
    best, best_score = 0, -np.inf
    for lag in lags_ms:
        labels = shifted(ds, rows, lag)
        score = weight = 0.0
        for member in _groups(ds, rows):
            c = RAMP_CHANNEL[str(ds.routine[member[0]])]
            i = [where[r] for r in member]
            y, p = labels[i, c], predictions[i, c]
            if y.std() > 1e-6 and p.std() > 1e-6:
                score += len(i) * np.corrcoef(y, p)[0, 1]
                weight += len(i)
        if weight and score / weight > best_score:
            best, best_score = int(lag), score / weight
    return best, (best_score if np.isfinite(best_score) else None)


def ramp_rows(ds):
    return np.flatnonzero(np.isin(ds.routine.astype(str), list(RAMP_CHANNEL)))


def _predict(probe, rows, predict):
    """(N, 6) pretrained outputs for probe rows; columns 2: hold the four expression channels."""
    preds = np.zeros((len(rows), 6), np.float32)
    for start in range(0, len(rows), 128):
        batch = rows[start:start + 128]
        preds[start:start + len(batch), 2:] = predict(np.stack([probe[int(i)][0].numpy() for i in batch]))
    return preds


def _isotonic(y):
    """Closest non-decreasing sequence to y (pool adjacent violators)."""
    values, counts = [], []
    for v in y:
        values.append(float(v)); counts.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            n = counts[-2] + counts[-1]
            values[-2] = (values[-2] * counts[-2] + values[-1] * counts[-1]) / n
            counts[-2] = n
            values.pop(); counts.pop()
    return np.repeat(values, counts)


def shape(schedule, response, min_range=.2, min_corr=.6):
    """Ramp labels following how the face actually moved: the response is rescaled between its levels in
    the schedule's true 0 and 1 holds, then made monotonic within each rise and fall. Returns None when the
    response is too flat or too unlike the schedule to trust (the lag-shifted schedule is kept)."""
    low, high = schedule < .02, schedule > .98
    if low.sum() < 3 or high.sum() < 3:
        return None
    lo, hi = np.median(response[low]), np.median(response[high])
    if hi - lo < min_range or np.corrcoef(schedule, response)[0, 1] < min_corr:
        return None
    q = np.clip((response - lo) / (hi - lo), 0, 1)
    # Cut at the middle of every hold so each segment holds exactly one rise or fall.
    hold = np.where(low, 0, np.where(high, 1, -1))
    cuts, i = [0], 0
    while i < len(hold):
        j = i
        while j + 1 < len(hold) and hold[j + 1] == hold[i]:
            j += 1
        if hold[i] >= 0:
            cuts.append((i + j) // 2)
        i = j + 1
    cuts.append(len(q))
    out = q.copy()
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a < 2:
            continue
        rising = schedule[b - 1] >= schedule[a]
        out[a:b] = _isotonic(q[a:b]) if rising else -_isotonic(-q[a:b])
    return out


def align(datasets, probes, predict, lags_ms=LAGS_MS, shape_labels=True):
    """Estimate the lag on probes[0], apply it to every dataset in place, then (shape_labels) replace each
    ramp curve with the base model's monotonic response where it is trustworthy. probes are un-augmented
    datasets covering every ramp window (train and validation)."""
    rows = ramp_rows(probes[0])
    if not len(rows):
        return None
    lag, corr = estimate_lag(probes[0], rows, _predict(probes[0], rows, predict), lags_ms)
    for ds in datasets:
        r = ramp_rows(ds)
        if len(r):
            ds.label[r] = shifted(ds, r, lag)
    report = {"lag_ms": lag, "correlation": None if corr is None else round(float(corr), 4),
              "probe_windows": int(len(rows)), "shaped_blocks": 0, "schedule_blocks": 0}
    if not shape_labels:
        return report
    response = {}
    for probe in probes:
        r = ramp_rows(probe)
        for i, pred in zip(r, _predict(probe, r, predict)):
            response[(int(probe.bin_id[i]), int(probe.pos[i]), int(probe.side[i]))] = pred
    shaped = {}
    for ds in datasets:
        for member in _groups(ds, ramp_rows(ds)):
            keys = [(int(ds.bin_id[m]), int(ds.pos[m]), int(ds.side[m])) for m in member]
            if not all(k in response for k in keys):
                continue
            c = RAMP_CHANNEL[str(ds.routine[member[0]])]
            group = (keys[0][0], int(ds.block[member[0]]), keys[0][2])
            if group not in shaped:
                shaped[group] = shape(ds.label[member, c], np.array([response[k][c] for k in keys]))
            if shaped[group] is not None and len(shaped[group]) == len(member):
                ds.label[member, c] = shaped[group]
    report["shaped_blocks"] = sum(v is not None for v in shaped.values())
    report["schedule_blocks"] = sum(v is None for v in shaped.values())
    return report


OPEN_ROUTINES = ("open", "gaze")


def confirm_open(datasets, probes, predict, max_closure=.15):
    """Mark lids open (closure 0) in prompted open-eye blocks where the pretrained model agrees, on every
    dataset sharing these windows. probes must be un-augmented; returns the number of confirmed windows."""
    confirmed = set()
    for probe in probes:
        rows = np.flatnonzero(np.isin(probe.routine.astype(str), OPEN_ROUTINES))
        for start in range(0, len(rows), 128):
            batch = rows[start:start + 128]
            closure = predict(np.stack([probe[int(i)][0].numpy() for i in batch]))[:, 0]
            confirmed |= {(int(probe.bin_id[i]), int(probe.pos[i]), int(probe.side[i]))
                          for i, c in zip(batch, closure) if c < max_closure}
    for ds in datasets:
        rows = np.flatnonzero(np.isin(ds.routine.astype(str), OPEN_ROUTINES))
        hit = [r for r in rows if (int(ds.bin_id[r]), int(ds.pos[r]), int(ds.side[r])) in confirmed]
        ds.label_mask[hit, 2] = True
        ds.label[hit, 2] = 0
    return len(confirmed)
