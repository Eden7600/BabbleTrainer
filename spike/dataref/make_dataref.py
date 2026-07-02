"""Generate parity reference data so a Rust reimplementation of BabbleTrainer's .bin
data pipeline can be byte/pixel-checked. CPU only; no GPU is touched.

Run with the DirectML venv python:
  BabbleTrainer/env_directml/Scripts/python.exe make_dataref.py

Deterministic and idempotent: same inputs -> identical outputs every run.
"""

import json
import struct
import sys
from itertools import islice
from pathlib import Path

# --- Paths -------------------------------------------------------------------
HERE = Path(__file__).resolve().parent                      # .../BabbleBurn/spike/dataref
REPO = HERE.parents[2]                                        # .../Babal-training
BT = REPO / "BabbleTrainer"
BIN_PATH = REPO / "ModelData" / "user_cal.bin"

sys.path.insert(0, str(BT))
import cv2                                                   # noqa: E402
import numpy as np                                           # noqa: E402
import bin_reader as BR                                      # noqa: E402
import config as C                                           # noqa: E402

N = 16
S = int(C.DUAL_INPUT_SIZE)
MIRROR = int(C.TRAIN_MIRROR_BIN_SIDE)


def process_eye_verbose(jpeg, side, raw_shapes=None):
    """Mirror bin_temporal_dataset._process_eye exactly, but also report the raw
    decoded shape and whether the INTER_AREA resize fires. Returns (eye_or_None, resized_bool)."""
    eye = BR.decode_eye(jpeg)
    if BR.is_blank(eye):
        if raw_shapes is not None and eye is not None:
            raw_shapes.append(tuple(eye.shape[:2]))
        return None, False
    raw_shape = tuple(eye.shape[:2])
    if raw_shapes is not None:
        raw_shapes.append(raw_shape)
    resized = False
    if eye.shape[:2] != (S, S):
        eye = cv2.resize(eye, (S, S), interpolation=cv2.INTER_AREA)
        resized = True
    eye = cv2.equalizeHist(eye)
    if side == MIRROR:
        eye = cv2.flip(eye, 1)
    return eye, resized


def main():
    HERE.mkdir(parents=True, exist_ok=True)

    frames = list(islice(BR.iter_frames(str(BIN_PATH), good_only=True), N))
    if len(frames) < N:
        raise RuntimeError(f"only {len(frames)} GOOD_DATA frames found, need {N}")

    left_labels = np.zeros((N, 6), dtype=np.float32)
    right_labels = np.zeros((N, 6), dtype=np.float32)
    gaze_valid = np.zeros((N,), dtype=np.float32)
    routine_state = np.zeros((N,), dtype=np.int64)
    flags = np.zeros((N, 3), dtype=np.int32)          # good, gaze, expr_unlabeled
    frames_side0 = np.zeros((N, S, S), dtype=np.uint8)
    frames_side1 = np.zeros((N, S, S), dtype=np.uint8)
    blank0 = np.zeros((N,), dtype=bool)
    blank1 = np.zeros((N,), dtype=bool)

    raw_shapes = []          # raw decoded eye dims across all processed eyes
    resize_fired = False

    for i, d in enumerate(frames):
        lt, rt, gv = BR.frame_targets_dual(d, mirror_side=MIRROR)
        left_labels[i] = lt
        right_labels[i] = rt
        gaze_valid[i] = gv
        st = int(d["routine_state"])
        routine_state[i] = st
        flags[i, 0] = 1 if (st & BR.FLAG_GOOD_DATA) else 0
        flags[i, 1] = 1 if BR.is_gaze(d) else 0
        flags[i, 2] = 1 if BR.is_expr_unlabeled(d) else 0

        eye0, r0 = process_eye_verbose(d["jpeg_left"], 0, raw_shapes)
        eye1, r1 = process_eye_verbose(d["jpeg_right"], 1, raw_shapes)
        resize_fired = resize_fired or r0 or r1
        if eye0 is None:
            blank0[i] = True
        else:
            frames_side0[i] = eye0
        if eye1 is None:
            blank1[i] = True
        else:
            frames_side1[i] = eye1

    # --- refs.npz ---
    refs_path = HERE / "refs.npz"
    np.savez(refs_path,
             left_labels=left_labels,
             right_labels=right_labels,
             gaze_valid=gaze_valid,
             routine_state=routine_state,
             flags=flags,
             frames_side0=frames_side0,
             frames_side1=frames_side1,
             blank0=blank0,
             blank1=blank1)

    # --- first_frame.json : raw header dump of the first good frame + byte offset ---
    # Re-scan to find the byte offset where the first good frame's header starts, and
    # grab the raw 100 header bytes there.
    first_off = None
    first_hdr_bytes = None
    with open(BIN_PATH, "rb") as f:
        while True:
            off = f.tell()
            hdr = f.read(BR.FRAME_SIZE)
            if len(hdr) < BR.FRAME_SIZE:
                raise RuntimeError("EOF before finding a good frame")
            rec = struct.unpack(BR.FRAME_FMT, hdr)
            d = dict(zip(BR.FIELDS, rec))
            jl, jr = d["jpeg_len_left"], d["jpeg_len_right"]
            if jl < 0 or jr < 0 or jl > BR.MAX_JPEG or jr > BR.MAX_JPEG:
                raise RuntimeError("bad jpeg len before finding a good frame")
            lj = f.read(jl)
            rj = f.read(jr)
            if len(lj) != jl or len(rj) != jr:
                raise RuntimeError("truncated before finding a good frame")
            if int(d["routine_state"]) & BR.FLAG_GOOD_DATA:
                first_off = off
                first_hdr_bytes = hdr
                break

    d0 = dict(zip(BR.FIELDS, struct.unpack(BR.FRAME_FMT, first_hdr_bytes)))
    # 16 float fields (routine_pitch..dilate) in order.
    float_fields = BR.FIELDS[:16]
    first_frame = {
        "byte_offset": int(first_off),
        "header_bytes_hex": first_hdr_bytes.hex(),
        "float_fields": {k: float(d0[k]) for k in float_fields},
        # explicit named callouts requested for the Rust struct check
        "left_eye_pitch": float(d0["left_eye_pitch"]),
        "left_eye_yaw": float(d0["left_eye_yaw"]),
        "right_eye_pitch": float(d0["right_eye_pitch"]),
        "right_eye_yaw": float(d0["right_eye_yaw"]),
        "left_lid": float(d0["left_lid"]),
        "right_lid": float(d0["right_lid"]),
        "widen": float(d0["widen"]),
        "squint": float(d0["squint"]),
        "brow_angry": float(d0["brow_angry"]),
        "routine_state": int(d0["routine_state"]),
        "jpeg_len_left": int(d0["jpeg_len_left"]),
        "jpeg_len_right": int(d0["jpeg_len_right"]),
    }
    ff_path = HERE / "first_frame.json"
    ff_path.write_text(json.dumps(first_frame, indent=2, sort_keys=True))

    # --- spec.json ---
    spec = {
        "FRAME_SIZE": int(BR.FRAME_SIZE),
        "FRAME_FMT": BR.FRAME_FMT,
        "FIELDS": list(BR.FIELDS),
        "FLAG_GOOD_DATA": int(BR.FLAG_GOOD_DATA),
        "FLAG_GAZE_DATA": int(BR.FLAG_GAZE_DATA),
        "FLAG_EXPR_UNLABELED": int(BR.FLAG_EXPR_UNLABELED),
        "MAX_JPEG": int(BR.MAX_JPEG),
        "GAZE_RANGE_DEG": float(BR.GAZE_RANGE_DEG),
        "IS_BLANK_THRESH": 2.0,
        "TRAIN_MIRROR_BIN_SIDE": int(C.TRAIN_MIRROR_BIN_SIDE),
        "DUAL_INPUT_SIZE": int(C.DUAL_INPUT_SIZE),
    }
    spec_path = HERE / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True))

    # ---------------------------------------------------------------- report --
    n_gaze = int(flags[:, 1].sum())
    n_eu = int(flags[:, 2].sum())
    n_good = int(flags[:, 0].sum())
    uniq_raw = sorted(set(raw_shapes))

    print("=== make_dataref report ===")
    print(f"bin: {BIN_PATH}")
    print(f"frames: {N}  good={n_good}  gaze_valid={n_gaze}  expr_unlabeled={n_eu}")
    print(f"routine_state (hex) per frame: {[hex(int(x)) for x in routine_state]}")
    print(f"flags [good,gaze,expr_unlabeled] per frame:\n{flags.tolist()}")
    print(f"blank0={blank0.tolist()}")
    print(f"blank1={blank1.tolist()}")
    print("--- first good frame ---")
    print(f"byte_offset = {first_off}")
    print(f"left_label  = {left_labels[0].tolist()}")
    print(f"right_label = {right_labels[0].tolist()}")
    print(f"gaze_valid  = {float(gaze_valid[0])}")
    print("--- raw decoded eye dims ---")
    print(f"first 6 raw decoded shapes: {raw_shapes[:6]}")
    print(f"unique raw decoded shapes across all {len(raw_shapes)} eyes: {uniq_raw}")
    print(f"resize (INTER_AREA) fired anywhere: {resize_fired}")
    print(f"already 128x128 on disk: {uniq_raw == [(S, S)]}")
    print("--- outputs ---")
    for p in (refs_path, ff_path, spec_path):
        print(f"{p.name}: {p.stat().st_size} bytes  ({p})")


if __name__ == "__main__":
    main()
