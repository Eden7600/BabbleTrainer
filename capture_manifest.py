"""Strict sidecar validation without changing the existing 100-byte capture header."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_RECORDS = 1_000_000
ROLES = {"train": 0, "validation": 1, "test": 2}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(bin_path):
    path = Path(str(bin_path) + ".manifest.json")
    if not path.exists():
        return None
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("Capture sidecar exceeds the size limit")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported capture schema")
    binary = data.get("binary", {})
    n = binary.get("record_count")
    if type(n) is not int or not 0 < n <= MAX_RECORDS:
        raise ValueError("Invalid capture record count")
    if binary.get("byte_length") != Path(bin_path).stat().st_size or binary.get("sha256", "").lower() != sha256(bin_path):
        raise ValueError("Capture sidecar does not match binary hash/length")
    if not data.get("session_id") or not data.get("run_id"):
        raise ValueError("Capture session and run identities are required")
    role = np.full(n, -1, np.int8)
    block = np.full(n, -1, np.int32)
    segment = np.full(n, -1, np.int32)
    masks = np.zeros((n, 12), dtype=bool)
    routines = []
    ids = set()
    prompt_wink_lid_masks_removed = 0
    for i, item in enumerate(data.get("blocks", [])):
        a, b = _range(item, n)
        if (block[a:b] >= 0).any():
            raise ValueError("Overlapping capture blocks")
        identity = item.get("block_id")
        if not identity or identity in ids:
            raise ValueError("Duplicate or missing block identity")
        ids.add(identity)
        if item.get("role") not in ROLES or not item.get("routine_id"):
            raise ValueError("Unknown partition role or missing routine")
        mask = item.get("label_validity")
        _mask(mask)
        mask = list(mask)
        if item.get("annotation_provenance") == "prompt" and item["routine_id"] in ("winkleft", "winkright"):
            # Older captures enabled the fellow/open eye. Intersect with the corrected mask, never add truth.
            channel = 2 if item["routine_id"] == "winkleft" else 8
            if mask[channel]:
                prompt_wink_lid_masks_removed += b - a
            mask[channel] = False
        role[a:b] = ROLES[item["role"]]
        block[a:b] = i
        masks[a:b] = mask
        routines.append(item["routine_id"])
    for i, item in enumerate(data.get("segments", [])):
        a, b = _range(item, n)
        if (segment[a:b] >= 0).any() or len(set(block[a:b])) != 1:
            raise ValueError("Overlapping segment or segment crosses blocks")
        segment[a:b] = i
    if (block < 0).any() or (segment < 0).any():
        raise ValueError("Orphan capture records without a block/continuity segment")
    overridden = np.zeros(n, bool)
    value_overrides = None
    for item in data.get("label_overrides", []):
        a, b = _range(item, n)
        if overridden[a:b].any():
            raise ValueError("Overlapping label overrides")
        overridden[a:b] = True
        _mask(item.get("label_validity"))
        values = item.get("values")
        if values is not None:
            if (item.get("annotation_provenance") != "reviewed" or not isinstance(values, list) or len(values) != 12
                    or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values)):
                raise ValueError("Numeric overrides require reviewed normalized finite values for all 12 channels")
            if value_overrides is None:
                value_overrides = np.full((n, 12), np.nan, np.float32)
            value_overrides[a:b] = np.where(item["label_validity"], values, np.nan)
        # Overrides can remove unsupported supervision; adding labels requires reviewed provenance.
        mask = np.asarray(item["label_validity"], bool)
        if (mask & ~masks[a:b]).any() and item.get("annotation_provenance") != "reviewed":
            raise ValueError("Unreviewed override cannot add verified supervision")
        masks[a:b] = mask
    timing = data.get("timing", {})
    image = timing.get("image_receive_ms", [])
    pose = timing.get("pose_receive_ms", [])
    if len(image) != n or len(pose) != n or not timing.get("clock_domain") or not timing.get("source"):
        raise ValueError("Missing or inconsistent capture timing provenance")
    if any(type(t) not in (int, float) or not math.isfinite(t) for t in image):
        raise ValueError("Invalid image timestamp")
    if any(t is not None and (type(t) not in (int, float) or not math.isfinite(t)) for t in pose):
        raise ValueError("Invalid pose timestamp")
    image = np.asarray(image, np.float64)
    pose = np.asarray([np.nan if t is None else t for t in pose], np.float64)
    for i in np.unique(segment):
        values = image[segment == i]
        if (values < 0).any() or (np.diff(values) <= 0).any() or (np.diff(values) > 200).any():
            raise ValueError("Timestamp discontinuity needs a new continuity segment")
    in_hold = np.zeros(n, bool)
    for hold in data.get("reviewed_holds", []):
        a, b = _range(hold, n)
        _mask(hold.get("channel_mask"))
        if (hold.get("annotation_provenance") != "reviewed" or not any(hold["channel_mask"])
                or len(set(segment[a:b])) != 1 or in_hold[a:b].any()):
            raise ValueError("A reviewed hold must have reviewed provenance and one continuity segment")
        in_hold[a:b] = True
    fresh_pose = np.isfinite(pose) & (image >= pose) & (image - pose <= 200)
    masks[:, [0, 1, 6, 7]] &= fresh_pose[:, None]
    result = dict(raw=data, role=role, block=block, segment=segment, masks=masks,
                  routines=routines, timestamps=image, record_count=n, value_overrides=value_overrides,
                  excluded=np.zeros(n, bool),
                  prompt_wink_lid_masks_removed=prompt_wink_lid_masks_removed)
    return result


def _range(item, n):
    a, b = item.get("start_record"), item.get("end_record_exclusive")
    if type(a) is not int or type(b) is not int or not 0 <= a < b <= n:
        raise ValueError("Invalid half-open capture range")
    return a, b


def _mask(mask):
    if not isinstance(mask, list) or len(mask) != 12 or any(type(v) is not bool for v in mask):
        raise ValueError("Supervision mask must contain exactly 12 booleans")


def validate_coverage(roles, routines, masks, reviewed=False):
    """Require repetitions, not merely some frames from the end of a capture."""
    coverage = {}
    if not np.any(roles == 0) or not np.any(roles == 1):
        raise ValueError("No usable training or validation frames remain; undo exclusions or repeat calibration")
    for routine in sorted(set(routines)):
        counts = {name: int(np.sum((roles == role) & (routines == routine)))
                  for name, role in ROLES.items()}
        if not reviewed and (counts["train"] == 0 or counts["validation"] == 0):
            raise ValueError(f"Repeat {routine}: usable training and independent validation blocks are required")
        coverage[routine] = counts
    for channel in range(masks.shape[1]):
        if not reviewed and np.any(masks[roles == 0, channel]) and not np.any(masks[roles == 1, channel]):
            raise ValueError(f"Missing validation label coverage for channel {channel}")
    return coverage
