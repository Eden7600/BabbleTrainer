# Emit eval_model.py's exact frame-split val mask so the Rust trainer can hold out the same frames.
import os
import sys

os.environ.setdefault("BACKBONE", "mobilenetv4_conv_small_050.e3000_r224_in1k")
BT = r"C:\Users\dan2wik\Documents\GitHub\Babal-training\BabbleTrainer"
sys.path.insert(0, BT)

import numpy as np
import config as C
import bin_temporal_dataset as TD

BIN = r"C:\Users\dan2wik\Documents\GitHub\Babal-training\ModelData\user_cal.bin"
IDX = r"C:\Users\dan2wik\AppData\Local\Temp\claude\c--Users-dan2wik-Documents-GitHub-Babal-training\7aa64667-b8c0-417a-9077-8b0ede92db19\scratchpad\eval_index.npz"
HERE = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(IDX):
    TD.build_index([BIN], index_path=IDX)
idx = np.load(IDX)
n = len(idx["bin_id"])
rng = np.random.default_rng(C.SEED)  # matches BinTemporalDataset frame split
perm = rng.permutation(n)
keep_val = np.zeros(n, dtype=bool)
keep_val[perm[: max(1, int(n * 0.1))]] = True

np.save(os.path.join(HERE, "val_mask.npy"), keep_val.astype(np.uint8))
# per-sample (record pos, side) for Rust alignment verification
np.save(os.path.join(HERE, "val_recside.npy"), np.stack([idx["pos"], idx["side"].astype(np.int32)], 1))
print(f"n={n} val={int(keep_val.sum())} train={int((~keep_val).sum())}")
print(f"wrote val_mask.npy + val_recside.npy")
