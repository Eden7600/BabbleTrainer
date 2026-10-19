"""Labels derived from the pretrained expression model.

Ramp blocks label each frame from the schedule the overlay shows, but people follow it late. The lag is
estimated once per capture as the shift that best correlates the pretrained model's output with the
schedule, then every ramp label is re-read at (t - lag) within its own block and eye.

Prompted open-eye blocks carry no lid label because people still blink during them. confirm_open
supplies the "open" lid endpoint where the pretrained model sees the eye clearly open.
"""
import numpy as np

# Per-eye label column each ramp routine drives: [Y, X, lid closure, widen, squint, brow].
RAMP_CHANNEL = {"lidramp": 2, "widenramp": 3, "squintramp": 4, "browramp": 5}
LAGS_MS = np.arange(0, 801, 25)


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


def align(datasets, probe_ds, predict, lags_ms=LAGS_MS):
    """Estimate lag on probe_ds with predict(inputs)->(N, 4 expr) and apply it to every dataset in place."""
    rows = ramp_rows(probe_ds)
    if not len(rows):
        return None
    preds = np.zeros((len(rows), 6), np.float32)
    for start in range(0, len(rows), 128):
        batch = rows[start:start + 128]
        preds[start:start + len(batch), 2:] = predict(np.stack([probe_ds[int(i)][0].numpy() for i in batch]))
    lag, corr = estimate_lag(probe_ds, rows, preds, lags_ms)
    for ds in datasets:
        r = ramp_rows(ds)
        if len(r):
            ds.label[r] = shifted(ds, r, lag)
    return {"lag_ms": lag, "correlation": None if corr is None else round(float(corr), 4), "probe_windows": int(len(rows))}


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
