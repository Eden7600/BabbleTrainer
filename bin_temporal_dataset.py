import json
import random
import struct
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config as C
import bin_reader as BR

USERDATA = C.ROOT / "userdata"
DECRYPTED = USERDATA / "decrypted"
INDEX_PATH = C.DUAL_INDEX_PATH
BINS_JSON = USERDATA / "bin_temporal_index_files.json"


# --- Geometric augmentation helpers (geom params sampled separately so temporal pairs share them) ---
def sample_geom(strong: bool, rng: random.Random) -> dict:
    if strong:
        return dict(
            angle=rng.uniform(-10, 10),
            tx=rng.uniform(-0.08, 0.08),
            ty=rng.uniform(-0.08, 0.08),
            scale=rng.uniform(0.85, 1.15),
        )
    return dict(
        angle=rng.uniform(-3, 3),
        tx=rng.uniform(-0.03, 0.03),
        ty=rng.uniform(-0.03, 0.03),
        scale=rng.uniform(0.97, 1.03),
    )


def apply_geom(eye: np.ndarray, g: dict) -> np.ndarray:
    h, w = eye.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), g["angle"], g["scale"])
    m[0, 2] += g["tx"] * w
    m[1, 2] += g["ty"] * h
    return cv2.warpAffine(eye, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def user_of(path) -> str:
    """User identity = the hash prefix before the first underscore (as in bin_dataset.py)."""
    return Path(path).name.split("_")[0]


def _bins_json_for(index_path):
    """The companion file listing the bin paths for a given index .npz."""
    p = Path(index_path)
    return p.with_name(p.stem + "_files.json")



def build_index(bin_paths=None, index_path=INDEX_PATH, limit=None):
    if bin_paths is None:
        bin_paths = sorted(str(p) for p in DECRYPTED.glob("*.bin"))
    else:
        bin_paths = [str(p) for p in bin_paths]
    if limit:
        bin_paths = bin_paths[:limit]
    index_path = Path(index_path)
    print(f"Scanning {len(bin_paths)} bins for temporal dual-eye samples...")

    bin_ids, positions, sides, labels, gazes, expr_unlabeleds = [], [], [], [], [], []
    rec_offsets = []       
    bin_rec_start = [0]    

    for bid, path in enumerate(bin_paths):
        n_before = len(bin_ids)
        pos = 0
        with open(path, "rb") as f:
            while True:
                off = f.tell()
                hdr = f.read(BR.FRAME_SIZE)
                if len(hdr) < BR.FRAME_SIZE:
                    break
                d = dict(zip(BR.FIELDS, struct.unpack(BR.FRAME_FMT, hdr)))
                jl, jr = d["jpeg_len_left"], d["jpeg_len_right"]
                if jl < 0 or jr < 0 or jl > BR.MAX_JPEG or jr > BR.MAX_JPEG:
                    break
                lj = f.read(jl)
                rj = f.read(jr)
                if len(lj) != jl or len(rj) != jr:
                    break
                rec_offsets.append(off)
                if int(d["routine_state"]) & BR.FLAG_GOOD_DATA:
                    lt, rt, gv = BR.frame_targets_dual(d, mirror_side=C.TRAIN_MIRROR_BIN_SIDE)
                    # Freeform gaze section (look at the dot with random faces) 
                    # Used for gaze + the unlabeled expr stream
                    eu = 1.0 if BR.is_expr_unlabeled(d) else 0.0
                    if not BR.is_blank(BR.decode_eye(lj)):
                        bin_ids.append(bid); positions.append(pos); sides.append(0)
                        labels.append(lt); gazes.append(gv); expr_unlabeleds.append(eu)
                    if not BR.is_blank(BR.decode_eye(rj)):
                        bin_ids.append(bid); positions.append(pos); sides.append(1)
                        labels.append(rt); gazes.append(gv); expr_unlabeleds.append(eu)
                pos += 1
        bin_rec_start.append(len(rec_offsets))
        print(f"  [{bid + 1}/{len(bin_paths)}] {Path(path).name[:24]}... +{len(bin_ids) - n_before} eyes "
              f"({pos} records, total {len(bin_ids)})", flush=True)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(index_path,
             bin_id=np.array(bin_ids, dtype=np.int32),
             pos=np.array(positions, dtype=np.int32),
             side=np.array(sides, dtype=np.int8),
             label=np.clip(np.array(labels, dtype=np.float32), 0.0, 1.0),
             gaze_valid=np.array(gazes, dtype=np.float32),
             expr_unlabeled=np.array(expr_unlabeleds, dtype=np.float32),
             rec_offset=np.array(rec_offsets, dtype=np.int64),
             bin_rec_start=np.array(bin_rec_start, dtype=np.int64))
    _bins_json_for(index_path).write_text(json.dumps(bin_paths))
    users = sorted({user_of(p) for p in bin_paths})
    ng = int(np.sum(gazes)) if gazes else 0
    print(f"\nIndex: {len(bin_ids)} eye-samples ({ng} gaze-valid) across {len(bin_paths)} bins, "
          f"{len(users)} users\n  -> {index_path}")



def _aug_stack(frames, rng):
    g = sample_geom(strong=True, rng=rng)                 # one transform for all frames
    bright = rng.uniform(-0.30, 0.30)
    contrast = rng.uniform(0.7, 1.3)
    gamma = rng.uniform(0.6, 1.6)
    sigma = rng.uniform(0.0, 1.5)
    out = []
    for fr in frames:
        x = apply_geom(fr, g).astype(np.float32) / 255.0
        x = np.clip(x * contrast + bright, 0, 1) ** gamma
        if sigma > 0.3:
            x = cv2.GaussianBlur(x, (0, 0), sigma)
        out.append(np.clip(x * 255.0, 0, 255).astype(np.uint8))
    return out


def _frames_to_stack(frames):
    """uint8 frame list -> [T,S,S] float32 in [0,1]."""
    return np.stack([fr.astype(np.float32) / 255.0 for fr in frames], axis=0)


def _mixup_frames(fa, fb, lam, rng):
    ga = sample_geom(strong=True, rng=rng)
    gb = sample_geom(strong=True, rng=rng)
    bright = rng.uniform(-0.30, 0.30); contrast = rng.uniform(0.7, 1.3)
    gamma = rng.uniform(0.6, 1.6); sigma = rng.uniform(0.0, 1.5)
    out = []
    for a, b in zip(fa, fb):
        a = apply_geom(a, ga).astype(np.float32) / 255.0
        b = apply_geom(b, gb).astype(np.float32) / 255.0
        x = (1.0 - lam) * a + lam * b                         # linear pixel blend
        x = np.clip(x * contrast + bright, 0, 1) ** gamma     # photometric on the blend
        if sigma > 0.3:
            x = cv2.GaussianBlur(x, (0, 0), sigma)
        out.append(np.clip(x * 255.0, 0, 255).astype(np.uint8))
    return _frames_to_stack(out)


def _aug_weak(frames, rng, geom):
    bright = rng.uniform(-0.10, 0.10)
    out = []
    for fr in frames:
        x = apply_geom(fr, geom).astype(np.float32) / 255.0
        x = np.clip(x + bright, 0, 1)
        out.append(np.clip(x * 255.0, 0, 255).astype(np.uint8))
    return out


def _occlude_lids(frames, rng, max_frac=None):
    max_frac = C.GAZE_OCCLUDE_MAX_FRAC if max_frac is None else max_frac
    S = frames[0].shape[0]
    ht = int(rng.uniform(0.0, max_frac) * S)               # upper lid drop
    hb = int(rng.uniform(0.0, max_frac) * S)               # lower lid raise
    ft = int(rng.uniform(0, 1) * 255); fb = int(rng.uniform(0, 1) * 255)   # lid value
    out = []
    for fr in frames:
        f = fr.copy()
        if ht > 0:
            f[:ht, :] = ft
        if hb > 0:
            f[S - hb:, :] = fb
        out.append(f)
    return out


def _aug_stack_gaze(frames, rng):
    do_spatial = rng.random() < 0.3
    do_intensity = rng.random() < 0.4
    do_blur = rng.random() < 0.3
    if do_spatial:
        g = dict(angle=rng.uniform(-12, 12), tx=rng.uniform(-0.172, 0.172),
                 ty=rng.uniform(-0.172, 0.172), scale=rng.uniform(0.9, 1.1))
    if do_intensity:
        bright = rng.uniform(-0.2, 0.2)
        contrast = rng.uniform(0.4, 1.6)
    if do_blur:
        ksize = rng.choice([3, 5, 7, 9, 11, 13, 15])
    out = []
    for fr in frames:
        x = apply_geom(fr, g) if do_spatial else fr
        x = x.astype(np.float32) / 255.0
        if do_intensity:
            x = np.clip(x * contrast + bright, 0, 1)
        if do_blur:
            x = cv2.GaussianBlur(x, (ksize, ksize), 0)
        out.append(np.clip(x * 255.0, 0, 255).astype(np.uint8))
    if rng.random() < C.GAZE_OCCLUDE_PROB:
        out = _occlude_lids(out, rng)              
    return out



class BinTemporalDataset(Dataset):
    def __init__(self, split: str, index_path=INDEX_PATH, split_mode: str = "user",
                 val_frac: float = 0.1, augment: bool = None, aug_mode: str = "expr",
                 mixup: bool = False, exclude_gaze: bool = False,
                 exclude_expr_unlabeled: bool = False):
        index_path = Path(index_path)
        if not index_path.exists():
            raise RuntimeError(f"{index_path} missing — build it first with build_index(bin_paths, index_path)")
        idx = np.load(index_path)
        self.bin_paths = json.loads(_bins_json_for(index_path).read_text())
        self.augment = (split == "train") if augment is None else augment
        rng = np.random.default_rng(C.SEED)

        if split_mode == "user":
            # deterministic user-level split (cross-user val; for the full corpus)
            users = sorted({user_of(p) for p in self.bin_paths})
            perm = rng.permutation(len(users))
            n_val = max(1, int(len(users) * val_frac))
            val_users = {users[i] for i in perm[:n_val]}
            bin_user = [user_of(p) for p in self.bin_paths]
            keep_val = np.array([bin_user[b] in val_users for b in idx["bin_id"]])
            desc = f"{len(val_users) if split == 'val' else len(users) - len(val_users)} {split} users"
        elif split_mode == "frame":
            # per-frame holdout — used when tuning on a single bin (1 user, so a user-split
            # would leave one side empty). Val is an optimistic same-session sanity metric.
            n = len(idx["bin_id"])
            perm = rng.permutation(n)
            keep_val = np.zeros(n, dtype=bool)
            keep_val[perm[:max(1, int(n * val_frac))]] = True
            desc = f"frame-split {split}"
        elif split_mode == "frame_tail":
            # CONTIGUOUS holdout: val = the last val_frac of the capture (index order = capture order).
            # Lets a held-out continuous segment be measured for frame-to-frame jitter (the random
            # 'frame' split scatters val frames, so consecutive-frame deltas can't be formed there).
            n = len(idx["bin_id"])
            n_val = max(1, int(n * val_frac))
            keep_val = np.zeros(n, dtype=bool)
            keep_val[-n_val:] = True
            desc = f"frame-tail {split}"
        else:
            raise ValueError(f"split_mode must be 'user', 'frame', or 'frame_tail', got {split_mode!r}")
        mask = keep_val if split == "val" else ~keep_val
        if exclude_gaze:
            # With expressions gaze recordings, those frames have squint frames but a forced expr=0 label
            # The expr-section frames carry the real expr labels.
            mask = mask & (idx["gaze_valid"] < 0.5)
        if exclude_expr_unlabeled:
            # FLAG_EXPR_UNLABELED: random faces with a fake neutral label while KEEPING the neutral gaze frames for grounding.
            # Narrower than exclude_gaze (which drops all gaze-valid frames, losing neutral grounding).
            eu = idx["expr_unlabeled"] if "expr_unlabeled" in idx.files else np.zeros(len(idx["bin_id"]), np.float32)
            mask = mask & (eu < 0.5)

        self.bin_id = idx["bin_id"][mask]
        self.pos = idx["pos"][mask]
        self.side = idx["side"][mask]
        self.label = idx["label"][mask]
        self.gaze_valid = idx["gaze_valid"][mask]
        eu_all = idx["expr_unlabeled"] if "expr_unlabeled" in idx.files else np.zeros(len(idx["bin_id"]), np.float32)
        self.expr_unlabeled = eu_all[mask]           # free expression gaze section 
        self.rec_offset = idx["rec_offset"]          # shared across all samples 
        self.bin_rec_start = idx["bin_rec_start"]
        self._fh = {}                                # per-worker lazy file handles
        if aug_mode not in ("expr", "gaze"):
            raise ValueError(f"aug_mode must be 'expr' or 'gaze', got {aug_mode!r}")
        self.aug_mode = aug_mode                     # 'expr' -> our strong aug; 'gaze' -> BabbleTrainer's
        # intensity-mixup pools: the bin labels are binary, so synthesize mid range by blending a NEUTRAL stack and a full EXPRESSION stack 
        self.mixup = mixup and self.augment
        if self.mixup:
            em = self.label[:, C.EXPR_SLICE[0]:C.EXPR_SLICE[1]].max(axis=1)
            self.neutral_idx = np.where(em < C.EXPR_MIXUP_NEUTRAL_THRESH)[0]
            self.expr_idx = np.where(em > C.EXPR_MIXUP_EXPR_THRESH)[0]
            if len(self.neutral_idx) == 0 or len(self.expr_idx) == 0:
                self.mixup = False     # need both poles to interpolate
        print(f"BinTemporalDataset[{split}]: {len(self.label)} eye-samples, "
              f"{int(self.gaze_valid.sum())} gaze-valid ({desc}, aug={aug_mode}"
              f"{f', mixup(n={len(self.neutral_idx)},e={len(self.expr_idx)})' if self.mixup else ''})")

    def __len__(self):
        return len(self.label)

    def __getstate__(self):
        # Don't pickle open file handles when DataLoader ships the dataset to workers
        # (Windows spawn) — workers reopen them lazily in _handle().
        s = self.__dict__.copy()
        s["_fh"] = {}
        return s

    def _handle(self, bid):
        fh = self._fh.get(bid)
        if fh is None:
            fh = open(self.bin_paths[bid], "rb")
            self._fh[bid] = fh
        return fh

    def _load_frame(self, bid, rpos, side):
        """uint8 [S,S] gray frame (histeq'd, right mirrored) or None if out-of-range /
        unreadable / blank (caller zero-pads)."""
        if rpos < 0:
            return None
        start = int(self.bin_rec_start[bid])
        nrec = int(self.bin_rec_start[bid + 1]) - start
        if rpos >= nrec:
            return None
        off = int(self.rec_offset[start + rpos])
        f = self._handle(bid)
        f.seek(off)
        hdr = f.read(BR.FRAME_SIZE)
        if len(hdr) < BR.FRAME_SIZE:
            return None
        d = struct.unpack(BR.FRAME_FMT, hdr)
        jl, jr = d[-2], d[-1]
        if side == 0:
            jpeg = f.read(jl)
        else:
            f.seek(jl, 1)             # skip the left jpeg
            jpeg = f.read(jr)
        eye = BR.decode_eye(jpeg)
        if BR.is_blank(eye):
            return None
        S = C.DUAL_INPUT_SIZE
        if eye.shape[:2] != (S, S):
            eye = cv2.resize(eye, (S, S), interpolation=cv2.INTER_AREA)
        eye = cv2.equalizeHist(eye)
        if side == C.TRAIN_MIRROR_BIN_SIDE:
            eye = cv2.flip(eye, 1)    # mirror this eye into canonical space
        return eye

    def _load_frames(self, bid, pos, side):
        """The TEMPORAL_FRAMES uint8 frames at (bid, pos, side): current + 3 predecessors, zero-padded."""
        S = C.DUAL_INPUT_SIZE
        zeros = np.zeros((S, S), dtype=np.uint8)
        frames = [self._load_frame(bid, pos - k, side) for k in range(C.TEMPORAL_FRAMES)]
        return [fr if fr is not None else zeros for fr in frames]

    @staticmethod
    def _to_stack(frames):
        """uint8 frame list -> [T,S,S] float32 in [0,1]."""
        return np.stack([fr.astype(np.float32) / 255.0 for fr in frames], axis=0)

    def _stack(self, bid, pos, side, aug_rng):
        """Build the [T,S,S] float32 stack at (bid, pos, side). aug_rng=None -> no augmentation; else
        the aug fn is selected by self.aug_mode ('expr' = our strong aug, 'gaze' = BabbleTrainer's)."""
        frames = self._load_frames(bid, pos, side)
        if aug_rng is not None:
            frames = (_aug_stack_gaze if self.aug_mode == "gaze" else _aug_stack)(frames, aug_rng)
        return self._to_stack(frames)

    def __getitem__(self, i):
        bid = int(self.bin_id[i]); pos = int(self.pos[i]); side = int(self.side[i])

        if self.mixup:
            r = random.random()
            ia = ib = None
            if r < C.EXPR_MIXUP_NEUTRAL_PROB:                # neutral <-> full-expression (the linearity driver)
                ia = int(np.random.choice(self.neutral_idx)); ib = int(np.random.choice(self.expr_idx))
            elif r < C.EXPR_MIXUP_NEUTRAL_PROB + C.EXPR_MIXUP_CROSS_PROB:   # general cross blend
                ia = random.randrange(len(self.label)); ib = random.randrange(len(self.label))
            if ia is not None:
                lam = random.random()
                fa = self._load_frames(int(self.bin_id[ia]), int(self.pos[ia]), int(self.side[ia]))
                fb = self._load_frames(int(self.bin_id[ib]), int(self.pos[ib]), int(self.side[ib]))
                cur = _mixup_frames(fa, fb, lam, random.Random())
                lbl = ((1.0 - lam) * self.label[ia] + lam * self.label[ib]).astype(np.float32)
                return (torch.from_numpy(cur), torch.from_numpy(lbl),
                        torch.tensor(0.0, dtype=torch.float32))
            # else: fall through to a plain sample

        label = torch.from_numpy(self.label[i].copy())
        gv = torch.tensor(float(self.gaze_valid[i]), dtype=torch.float32)

        cur = self._stack(bid, pos, side, random.Random() if self.augment else None)
        return torch.from_numpy(cur), label, gv

    def sampler_weights(self, active_boost: float = 4.0, gaze_boost: float = 4.0,
                        neutral_gaze_weight: float = None):
        """Oversample active-expression frames AND gaze-valid frames so both the sparse
        gaze head and the mostly-neutral expression channels get enough signal."""
        a, b = C.EXPR_SLICE
        expr = self.label[:, a:b].max(axis=1).astype(np.float64)
        w = 1.0 + active_boost * expr + gaze_boost * self.gaze_valid.astype(np.float64)
        ngw = C.EXPR_NEUTRAL_GAZE_WEIGHT if neutral_gaze_weight is None else neutral_gaze_weight
        if ngw != 1.0:
            neutral_gaze = ((self.gaze_valid > 0.5) & (expr < C.GAZE_EXPR_ACTIVE_THRESH)
                            & (self.expr_unlabeled < 0.5))   # gaze-valid, low expr label, not gazeexpr
            w[neutral_gaze] *= ngw
        return w

    def gaze_sampler_weights(self, closed_ratio: float = None, lid_thresh: float = None,
                             expr_ratio: float = None):
        """gaze batcher STRATIFIED 3 ways so the gaze net is not swamped by the long neutral gaze pass. 
        Within each bucket weights are normalized by count, so each bucket hits its target proportion regardless of size"""
        closed_ratio = C.GAZE_CLOSED_RATIO if closed_ratio is None else closed_ratio
        expr_ratio = C.GAZE_EXPR_RATIO if expr_ratio is None else expr_ratio
        lid_thresh = C.GAZE_CLOSED_LID_THRESH if lid_thresh is None else lid_thresh
        gaze = self.gaze_valid > 0.5
        closed = (self.label[:, C.EXPR_SLICE[0]] > lid_thresh) & (~gaze)   # EyeLid (=1-lid) high, not gaze
        a, b = C.EXPR_SLICE
        active = self.label[:, a:b].max(axis=1) > C.GAZE_EXPR_ACTIVE_THRESH
        expr_gaze = gaze & (active | (self.expr_unlabeled > 0.5))          # held squint/widen/brow + gazeexpr
        neut_gaze = gaze & (~expr_gaze)
        w = np.zeros(len(self.label), dtype=np.float64)
        gaze_budget = 1.0 - closed_ratio
        # split the gaze budget; if one gaze sub-bucket is empty the other takes the whole gaze budget
        e_share = expr_ratio * gaze_budget if expr_gaze.any() else 0.0
        n_share = (1.0 - expr_ratio) * gaze_budget if neut_gaze.any() else 0.0
        if e_share + n_share > 0:                                          # renormalize to the full gaze budget
            scale = gaze_budget / (e_share + n_share)
            e_share *= scale; n_share *= scale
        if closed.any():
            w[closed] = closed_ratio / closed.sum()
        if expr_gaze.any():
            w[expr_gaze] = e_share / expr_gaze.sum()
        if neut_gaze.any():
            w[neut_gaze] = n_share / neut_gaze.sum()
        if w.sum() == 0:
            w[:] = 1.0
        return w


class BinPairedEyeDataset(Dataset):
    """Paired-eye stream for Mechanism C (evenness loss) + Mechanism B (consensus head): yields BOTH eyes
    of the SAME real frame -> (left_stack, right_stack, left_label[6], right_label[6]). 'left' is the
    un-flipped canonical eye (bin side != TRAIN_MIRROR_BIN_SIDE -> leftEye* at export); 'right' is the
    mirrored eye (bin side == TRAIN_MIRROR_BIN_SIDE -> rightEye*). This (un-flipped, mirrored) order
    matches MergedDualEye so the consensus head sees (feat_left, feat_right) the same way at train + export.

    Real frames only (NO mixup): the L/R pairing requires both physical eyes of one capture instant, which
    a mixup blend of two random frames would destroy. Both eyes are predicted through the ONE shared tower,
    so pred_left/pred_right are directly comparable for the evenness loss."""

    def __init__(self, split: str, index_path=INDEX_PATH, split_mode: str = "user", val_frac: float = 0.1,
                 augment: bool = None, exclude_expr_unlabeled: bool = False):
        base = BinTemporalDataset(split, index_path=index_path, split_mode=split_mode, val_frac=val_frac,
                                  augment=augment, aug_mode="expr", mixup=False,
                                  exclude_expr_unlabeled=exclude_expr_unlabeled)
        self.base = base
        self.mirror_side = int(C.TRAIN_MIRROR_BIN_SIDE)   # -> rightEye* (feat_right)
        self.canon_side = 1 - self.mirror_side            # un-flipped -> leftEye* (feat_left)
        # Pair the kept samples by (bin_id, pos): keep frames where BOTH eyes survived (non-blank, in-split).
        by_frame = {}
        for k in range(len(base.label)):
            by_frame.setdefault((int(base.bin_id[k]), int(base.pos[k])), {})[int(base.side[k])] = k
        self.pairs = [(b, p, sd[self.canon_side], sd[self.mirror_side])
                      for (b, p), sd in by_frame.items()
                      if self.canon_side in sd and self.mirror_side in sd]
        a, bb = C.EXPR_SLICE
        self._expr_max = (np.array([max(base.label[ic, a:bb].max(), base.label[im, a:bb].max())
                                    for _, _, ic, im in self.pairs], dtype=np.float64)
                          if self.pairs else np.zeros(0, dtype=np.float64))
        print(f"BinPairedEyeDataset[{split}]: {len(self.pairs)} L/R eye-pairs "
              f"(from {len(base.label)} eye-samples)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        b, p, ic, im = self.pairs[i]
        aug = self.base.augment
        # Independent augmentation per eye (the two eyes are different images); the evenness loss is a
        # gentle agree-gated pull, so this doubles as light L/R aug-consistency.
        left = self.base._stack(b, p, self.canon_side, random.Random() if aug else None)
        right = self.base._stack(b, p, self.mirror_side, random.Random() if aug else None)
        ll = torch.from_numpy(self.base.label[ic].copy())
        rl = torch.from_numpy(self.base.label[im].copy())
        return torch.from_numpy(left), torch.from_numpy(right), ll, rl

    def sampler_weights(self, active_boost: float = 4.0):
        return 1.0 + active_boost * self._expr_max


def _unlabeled_frames_from_bin(bin_path):
    """Flat [2*N, S, S] uint8 (frame i -> [2i]=left, [2i+1]=right-mirrored) from the FLAG_EXPR_UNLABELED"""
    S = C.DUAL_INPUT_SIZE
    out = []
    n_rec = 0
    with open(bin_path, "rb") as f:
        while True:
            hdr = f.read(BR.FRAME_SIZE)
            if len(hdr) < BR.FRAME_SIZE:
                break
            d = dict(zip(BR.FIELDS, struct.unpack(BR.FRAME_FMT, hdr)))
            jl, jr = d["jpeg_len_left"], d["jpeg_len_right"]
            if jl < 0 or jr < 0 or jl > BR.MAX_JPEG or jr > BR.MAX_JPEG:
                break
            lj = f.read(jl); rj = f.read(jr)
            if len(lj) != jl or len(rj) != jr:
                break
            st = int(d["routine_state"])
            if not (st & BR.FLAG_GOOD_DATA) or not (st & BR.FLAG_EXPR_UNLABELED):
                continue
            n_rec += 1
            for side, jpeg in ((0, lj), (1, rj)):
                eye = BR.decode_eye(jpeg)
                if BR.is_blank(eye):
                    eye = np.zeros((S, S), dtype=np.uint8)
                else:
                    if eye.shape[:2] != (S, S):
                        eye = cv2.resize(eye, (S, S), interpolation=cv2.INTER_AREA)
                    eye = cv2.equalizeHist(eye)
                    if side == C.TRAIN_MIRROR_BIN_SIDE:
                        eye = cv2.flip(eye, 1)
                out.append(eye)
    if not out:
        raise RuntimeError(f"no FLAG_EXPR_UNLABELED (free-expression gaze) frames in {bin_path} — was the "
                           f"'gazeexpr' calibration section recorded?")
    print(f"  bin unlabeled-expr section: {n_rec} records -> {len(out)} eye-frames from {Path(bin_path).name}")
    return np.stack(out, axis=0)


class FootageUnlabeledDataset(Dataset):
    """Continuous UNLABELED eye-video for the decoupled expression anti-jitter losses. Yields three
    views — (cur_strong, cur_weak, nbr_weak, pair_ok) — but NO label: the consistency/temporal
    losses run here, SEPARATE from the supervised labeled batch (so they never fight supervision).

    npz layout (from preprocess_footage): jpegs[2*N] (frame i -> [2i]=sideA, [2i+1]=sideB), size=S.
    A temporal stack at (frame i, side) = frames [i, i-1, i-2, i-3] of that side (zero-padded); the
    temporal partner is the previous timestep's stack (frame i-1)."""

    def __init__(self, npz_path=None, augment: bool = True, frames=None):
        if frames is not None:
            # calibration bin's free-expression section, small enough to hold in memory 
            self.frames = frames
            src = "in-memory bin section"
        else:
            npz_path = Path(npz_path)
            # Decode the JPEGs ONCE into a raw uint8 cache and mmap it for no access decode
            raw = npz_path.with_name(npz_path.stem + "_raw.npy")
            if not raw.exists():
                d = np.load(npz_path, allow_pickle=True)
                jpegs = d["jpegs"]; S = int(d["size"])
                arr = np.zeros((len(jpegs), S, S), dtype=np.uint8)
                for i, buf in enumerate(jpegs):
                    arr[i] = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)
                np.save(raw, arr)
            self.frames = np.load(raw, mmap_mode="r")   # [2*N, S, S] uint8, shared across workers
            src = f"{raw.name}, mmap"
        self.S = int(self.frames.shape[1])
        self.n = len(self.frames) // 2              # frames per eye-side
        self.augment = augment
        print(f"FootageUnlabeledDataset: {self.n} frames x 2 sides ({src})")

    @classmethod
    def from_bin(cls, bin_path, augment: bool = True):
        """Unlabeled-expression stream from a calibration bin's FREE-expression gaze section"""
        return cls(augment=augment, frames=_unlabeled_frames_from_bin(Path(bin_path)))

    def __len__(self):
        return self.n * 2

    def _frame(self, fi, side):
        if fi < 0 or fi >= self.n:
            return None
        return np.asarray(self.frames[fi * 2 + side])   # mmap slice to ndarray

    def _frames(self, fi, side):
        z = np.zeros((self.S, self.S), dtype=np.uint8)
        out = []
        for k in range(C.TEMPORAL_FRAMES):
            f = self._frame(fi - k, side)
            out.append(f if f is not None else z)
        return out

    def __getitem__(self, idx):
        fi, side = idx // 2, idx % 2
        cur_frames = self._frames(fi, side)
        pair_ok = 1.0 if fi >= 1 else 0.0
        if self.augment:
            cur_strong = _frames_to_stack(_aug_stack(cur_frames, random.Random()))
            g_weak = sample_geom(strong=False, rng=random.Random())
            cur_weak = _frames_to_stack(_aug_weak(cur_frames, random.Random(), g_weak))
            nbr_weak = (_frames_to_stack(_aug_weak(self._frames(fi - 1, side), random.Random(), g_weak))
                        if pair_ok else cur_weak.copy())
        else:
            cur_strong = cur_weak = _frames_to_stack(cur_frames)
            nbr_weak = cur_weak.copy()
        return (torch.from_numpy(cur_strong), torch.from_numpy(cur_weak), torch.from_numpy(nbr_weak),
                torch.tensor(pair_ok, dtype=torch.float32))
