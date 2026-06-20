import struct

import cv2
import numpy as np

FRAME_FMT = "=ffffffffffffffffqqqiii"
FRAME_SIZE = struct.calcsize(FRAME_FMT)  # 100

FIELDS = [
    "routine_pitch", "routine_yaw", "routine_distance", "routine_convergence",
    "fov_adjust_distance", "left_eye_pitch", "left_eye_yaw", "right_eye_pitch",
    "right_eye_yaw", "left_lid", "right_lid", "brow_raise", "brow_angry",
    "widen", "squint", "dilate",
    "timestamp", "video_ts_left", "video_ts_right",
    "routine_state", "jpeg_len_left", "jpeg_len_right",
]

FLAG_GOOD_DATA = 1 << 30
FLAG_GAZE_DATA = 1 << 0
FLAG_EXPR_UNLABELED = 1 << 1
SAFE_FRAME_STATE = 67108864
MAX_JPEG = 10 * 1024 * 1024


def iter_frames(path, good_only=True):
    """Yield per-frame dicts (header fields + raw 'jpeg_left'/'jpeg_right' bytes).

    Stops cleanly on truncated/invalid records. With good_only, skips frames that
    don't have FLAG_GOOD_DATA set.
    """
    with open(path, "rb") as f:
        while True:
            hdr = f.read(FRAME_SIZE)
            if len(hdr) < FRAME_SIZE:
                return
            d = dict(zip(FIELDS, struct.unpack(FRAME_FMT, hdr)))
            jl, jr = d["jpeg_len_left"], d["jpeg_len_right"]
            if jl < 0 or jr < 0 or jl > MAX_JPEG or jr > MAX_JPEG:
                return
            lj = f.read(jl)
            rj = f.read(jr)
            if len(lj) != jl or len(rj) != jr:
                return
            if good_only and not (int(d["routine_state"]) & FLAG_GOOD_DATA):
                continue
            d["jpeg_left"] = lj
            d["jpeg_right"] = rj
            yield d


def decode_eye(jpeg_bytes):
    return cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)


def frame_targets(d, channels):
    def val(ch, lid):
        if ch == "EyesClosed":
            return 1.0 - lid
        if ch == "BrowLowerer":
            return d["brow_angry"]
        if ch == "EyeWide":
            return d["widen"]
        if ch == "LidTightener":
            return d["squint"]
        raise KeyError(f"no bin mapping for channel '{ch}'")

    left = np.array([val(c, d["left_lid"]) for c in channels], dtype=np.float32)
    right = np.array([val(c, d["right_lid"]) for c in channels], dtype=np.float32)
    return left, right


def is_gaze(d):
    return bool(int(d["routine_state"]) & FLAG_GAZE_DATA)


def is_expr_unlabeled(d):
    """True for the free-expression gaze section (look-at-dot + random faces). Gaze-labeled but the
    expression is unlabeled -> feeds the decoupled unlabeled-expression temporal/Mean-Teacher stream."""
    return bool(int(d["routine_state"]) & FLAG_EXPR_UNLABELED)


# Per-eye gaze is assumed in [-45, +45] deg.
GAZE_RANGE_DEG = 45.0


def gaze_norm(angle_deg, rng_deg=GAZE_RANGE_DEG):
    """Normalize a per-eye gaze angle (deg) to [0,1]: (deg + rng)/(2*rng), clamped.
    rng=45 -> (deg+45)/90"""
    v = (float(angle_deg) + rng_deg) / (2.0 * rng_deg)
    return float(min(1.0, max(0.0, v)))


def frame_targets_dual(d, mirror_side=1, gaze_range=GAZE_RANGE_DEG):
    widen = float(d["widen"]); squint = float(d["squint"]); brow = float(d["brow_angry"])

    pitch_l = gaze_norm(d["left_eye_pitch"], gaze_range)
    yaw_l = gaze_norm(d["left_eye_yaw"], gaze_range)
    pitch_r = gaze_norm(d["right_eye_pitch"], gaze_range)
    yaw_r = gaze_norm(d["right_eye_yaw"], gaze_range)
    if mirror_side == 1:
        yaw_r = 1.0 - yaw_r
    elif mirror_side == 0:
        yaw_l = 1.0 - yaw_l

    left = np.array([pitch_l, yaw_l, 1.0 - float(d["left_lid"]), widen, squint, brow], dtype=np.float32)
    right = np.array([pitch_r, yaw_r, 1.0 - float(d["right_lid"]), widen, squint, brow], dtype=np.float32)
    return left, right, (1.0 if is_gaze(d) else 0.0)


def is_blank(img, thresh=2.0):
    return img is None or float(img.std()) < thresh


def iter_eye_samples(path, channels, good_only=True, drop_blank=True, mirror_right=True):
    for d in iter_frames(path, good_only=good_only):
        lt, rt = frame_targets(d, channels)
        L = decode_eye(d["jpeg_left"])
        if not (drop_blank and is_blank(L)):
            yield L, lt, "L"
        R = decode_eye(d["jpeg_right"])
        if not (drop_blank and is_blank(R)):
            yield (cv2.flip(R, 1) if mirror_right else R), rt, "R"
