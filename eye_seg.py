"""Eye-region segmentation (background, sclera, iris, pupil) at Babble's 128x128 input, and the lid/pupil
geometry derived from it. Research tool: trained on OpenEDS, which is research-only (never ship these weights).

  python eye_seg.py train E:/datasets/openEDS --out runs/seg
"""
import argparse
import json
import random
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import config as C
import device as Dev

SIZE = C.DUAL_INPUT_SIZE
CLASSES = 4  # 0 background, 1 sclera, 2 iris, 3 pupil (OpenEDS convention)
MIN_IRIS_PX = 12


# ---------------------------------------------------------------- geometry
def geometry(mask):
    """(openness, pupil_ratio, iris_width_px) from one class mask. Both ratios use iris width, which the
    lids rarely hide, so they are independent of crop scale. NaN where the iris is not visible."""
    iris = mask >= 2
    cols = np.flatnonzero(iris.any(axis=0))
    width = float(cols[-1] - cols[0] + 1) if len(cols) else 0.0
    if iris.sum() < 8 or width < MIN_IRIS_PX:  # a sliver of iris gives meaningless ratios
        return (0.0 if (mask > 0).sum() < 8 else np.nan), np.nan, width
    cx = int(np.argmax(iris.sum(axis=0)))  # a median column can fall in a gap between iris fragments
    rows = np.flatnonzero((mask[:, max(0, cx - 1):cx + 2] > 0).any(axis=1))
    openness = float(rows[-1] - rows[0] + 1) / width
    pupil = mask == 3
    pupil_ratio = float(pupil.any(axis=0).sum()) / width if pupil.sum() >= 3 else np.nan
    return openness, pupil_ratio, width


# ---------------------------------------------------------------- data
def _babble_view(img, mask, box, rng, train):
    """Warp an OpenEDS frame so the eye sits like a Beyond-style crop: small, high, seen from below."""
    x1, x2, y1, y2 = box
    if x2 - x1 >= img.shape[1] - 1:  # no eye box (blink); fall back to the frame centre
        x1, x2, y1, y2 = 160, 480, 120, 280
    w = max(x2 - x1, 40)
    if train:
        eye_w = rng.uniform(.4, .95) * SIZE
        cx, cy = rng.uniform(.3, .7) * SIZE, rng.uniform(.2, .6) * SIZE
        squash, angle, flip = rng.uniform(.55, 1.05), rng.uniform(-15, 15), rng.random() < .5
    else:
        eye_w, cx, cy, squash, angle, flip = .7 * SIZE, .5 * SIZE, .4 * SIZE, .8, 0, False
    s = eye_w / w
    a = np.deg2rad(angle)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.diag([s, s * squash])
    src_c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    m = np.hstack([rot, (np.array([cx, cy]) - rot @ src_c)[:, None]])
    img = cv2.warpAffine(img, m, (SIZE, SIZE), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(mask, m, (SIZE, SIZE), flags=cv2.INTER_NEAREST, borderValue=0)
    if flip:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if train:
        f = img.astype(np.float32) / 255
        f = np.clip((f ** rng.uniform(.6, 1.6)) * rng.uniform(.7, 1.3), 0, 1)
        if rng.random() < .5:  # the Beyond LED streak along one edge
            x = int(rng.integers(0, 10)) if rng.random() < .7 else SIZE - int(rng.integers(1, 10))
            f[:, max(0, x - 3):x + 3] = np.clip(f[:, max(0, x - 3):x + 3] + rng.uniform(.5, 1), 0, 1)
        sigma = rng.uniform(0, 1.2)
        if sigma > .2:
            f = cv2.GaussianBlur(f, (0, 0), sigma)
        f = np.clip(f + rng.normal(0, rng.uniform(0, .03), f.shape), 0, 1)
        img = (f * 255).astype(np.uint8)
    return cv2.equalizeHist(np.ascontiguousarray(img)), np.ascontiguousarray(mask)


class OpenEDSSeg(Dataset):
    def __init__(self, root, split, val_frac=.1):
        root = Path(root)
        base, boxes = root / "openEDS" / "openEDS", root / "bbox" / "bbox"
        self.items, self.train = [], split == "train"
        for seq in sorted(p for p in base.iterdir() if p.is_dir()):
            # Sequences, not frames, are split so validation never shares a recording with training.
            if (zlib.crc32(seq.name.encode()) % 1000 < val_frac * 1000) == self.train:
                continue
            box_file = boxes / f"{seq.name}.txt"
            lines = box_file.read_text().split("\n") if box_file.exists() else []
            for png in sorted(seq.glob("*.png"), key=lambda p: int(p.stem)):
                if not png.with_suffix(".npy").exists():
                    continue
                i = int(png.stem)
                box = tuple(int(v) for v in lines[i].split()) if i < len(lines) and lines[i].strip() else (0, 640, 0, 400)
                self.items.append((png, box))
        print(f"OpenEDSSeg[{split}]: {len(self.items)} frames", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        png, box = self.items[i]
        img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        mask = np.load(png.with_suffix(".npy"), allow_pickle=False).astype(np.uint8)
        rng = np.random.default_rng(None if self.train else i)
        img, mask = _babble_view(img, mask, box, rng, self.train)
        return torch.from_numpy(img).float().div(255)[None], torch.from_numpy(mask).long()


# ---------------------------------------------------------------- model
def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class EyeSegNet(nn.Module):
    """Small U-Net, about 0.5M parameters, single-channel 128x128 in, CLASSES logits out."""

    def __init__(self, ch=(16, 32, 64, 128)):
        super().__init__()
        self.down = nn.ModuleList()
        prev = 1
        for c in ch:
            self.down.append(_block(prev, c))
            prev = c
        self.up = nn.ModuleList(_block(ch[i] + ch[i + 1], ch[i]) for i in reversed(range(len(ch) - 1)))
        self.head = nn.Conv2d(ch[0], CLASSES, 1)
        self.presence = nn.Linear(ch[-1], 1)  # logit: is an eye region (open or closed) in front of the camera
        self.state = nn.Linear(ch[-1], 3)     # lid openness, pupil/iris ratio, pupil-visible logit

    def forward(self, x, with_presence=False):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        presence = self.presence(x.mean((2, 3)))[:, 0]
        for block, skip in zip(self.up, reversed(skips[:-1])):
            x = block(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False), skip], 1))
        return (self.head(x), presence) if with_presence else self.head(x)


    def _pooled(self, x):
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
        return x.mean((2, 3))

    def presence_logit(self, x):
        """Encoder-only path for the presence head (no decoder), as deployed in the eye model."""
        return self.presence(self._pooled(x))[:, 0]

    def eye_state(self, x):
        """Encoder-only (presence logit, openness, pupil ratio, pupil-visible logit), each shaped [B]."""
        f = self._pooled(x)
        s = self.state(f)
        return self.presence(f)[:, 0], s[:, 0], torch.sigmoid(s[:, 1]), s[:, 2]


def load_seg(path):
    net = EyeSegNet()
    missing, _ = net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"], strict=False)
    if any(not k.startswith(("presence.", "state.")) for k in missing):
        raise ValueError(f"segmentation checkpoint is missing {missing}")
    net.has_presence = not any(k.startswith("presence.") for k in missing)
    net.has_state = not any(k.startswith("state.") for k in missing)
    return net


def dice_loss(logits, target):
    p = logits.softmax(1)
    # Comparison instead of F.one_hot, which DirectML cannot scatter.
    t = (target[:, None] == torch.arange(CLASSES, device=target.device)[None, :, None, None]).float()
    inter = (p * t).sum((0, 2, 3))
    return 1 - ((2 * inter + 1) / (p.sum((0, 2, 3)) + t.sum((0, 2, 3)) + 1))[1:].mean()


@torch.no_grad()
def evaluate(net, ds, device, limit=3000):
    net.eval()
    idx = np.linspace(0, len(ds) - 1, min(limit, len(ds))).astype(int)
    inter, union = np.zeros(CLASSES), np.zeros(CLASSES)
    geo_err = []
    for start in range(0, len(idx), 128):
        x, y = zip(*(ds[int(i)] for i in idx[start:start + 128]))
        pred = net(torch.stack(x).to(device)).argmax(1).cpu().numpy()
        y = torch.stack(y).numpy()
        for c in range(CLASSES):
            inter[c] += ((pred == c) & (y == c)).sum()
            union[c] += ((pred == c) | (y == c)).sum()
        for p, t in zip(pred, y):
            geo_err.append(np.subtract(geometry(p)[:2], geometry(t)[:2]))
    net.train()
    err = np.abs(np.array(geo_err))
    return {"iou": [round(float(v), 4) for v in inter / np.maximum(union, 1)],
            "openness_mae": round(float(np.nanmean(err[:, 0])), 4),
            "pupil_ratio_mae": round(float(np.nanmean(err[:, 1])), 4)}


def train(args):
    torch.manual_seed(C.SEED); np.random.seed(C.SEED); random.seed(C.SEED)
    device = Dev.get_device()
    tr, va = OpenEDSSeg(args.root, "train"), OpenEDSSeg(args.root, "val")
    loader = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                        persistent_workers=args.workers > 0)
    net = EyeSegNet().to(device)
    print(f"EyeSegNet: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=.05)
    weight = torch.tensor([.5, 1, 1, 2], device=device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    step, best = 0, -1
    while step < args.steps:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            loss = F.cross_entropy(logits, y, weight=weight) + dice_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            step += 1
            if step % 1000 == 0 or step == args.steps:
                report = evaluate(net, va, device)
                score = float(np.mean(report["iou"][1:]))
                line = f"[{step}/{args.steps}] loss={loss.item():.4f} {report}"
                print(line, flush=True)
                with open(out / "train.log", "a", encoding="utf-8") as log:
                    log.write(line + "\n")
                if score > best:
                    best = score
                    torch.save({"model": net.state_dict(), "step": step, "validation": report}, out / "seg_best.pt")
            if step >= args.steps:
                break


# ---------------------------------------------------------------- presence (headset worn?)
def _photometric(img, rng):
    f = img.astype(np.float32) / 255
    f = np.clip((f ** rng.uniform(.7, 1.4)) * rng.uniform(.7, 1.3), 0, 1)
    sigma = rng.uniform(0, 1.2)
    if sigma > .2:
        f = cv2.GaussianBlur(f, (0, 0), sigma)
    f = np.clip(f + rng.normal(0, rng.uniform(0, .03), f.shape), 0, 1)
    img = (f * 255).astype(np.uint8)
    return img[:, ::-1] if rng.random() < .5 else img


def _synthetic_absent(rng):
    """Dark, noisy views with the headset's LED streak and stray bright objects, but no eye."""
    f = np.full((SIZE, SIZE), rng.uniform(0, .35), np.float32)
    f += cv2.GaussianBlur(rng.normal(0, rng.uniform(.02, .15), (SIZE, SIZE)).astype(np.float32), (0, 0), rng.uniform(1, 12))
    for _ in range(int(rng.integers(0, 3))):
        c = tuple(int(v) for v in rng.integers(0, SIZE, 2))
        cv2.ellipse(f, c, tuple(int(v) for v in rng.integers(3, 30, 2)), float(rng.uniform(0, 180)), 0, 360, float(rng.uniform(.5, 1)), -1)
    x = int(rng.integers(0, 12))
    f[:, max(0, x - 3):x + 3] += rng.uniform(.3, 1)
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


NO_STATE = (np.nan, np.nan, 0.0)   # headset off: nothing to measure, and no pupil is visible


def state_targets(mask):
    """(openness, pupil ratio, pupil visible) from a class mask; NaN where the geometry is unknown."""
    openness, pupil, _ = geometry(mask)
    return float(openness), float(pupil), float(np.isfinite(pupil))


def _state(targets):
    t = torch.tensor(targets, dtype=torch.float32)
    return torch.nan_to_num(t), torch.isfinite(t)


class EyeFrames(Dataset):
    """Unsegmented 128x128 eye crops with a presence label (mask is ignored when has_mask is 0) and optional
    per-frame state targets."""

    def __init__(self, images, present, train, absent_mask=False, state=None):
        self.images, self.present, self.train, self.absent_mask = images, float(present), train, absent_mask
        self.state = state

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.train else i)
        img = _photometric(self.images[i], rng) if self.train else self.images[i]
        img = cv2.equalizeHist(np.ascontiguousarray(img))
        # A frame with no eye has an all-background mask, so absent frames also supervise segmentation.
        targets = tuple(self.state[i]) if self.state is not None else (NO_STATE if not self.present else (np.nan,) * 3)
        return (torch.from_numpy(img).float().div(255)[None], torch.zeros(SIZE, SIZE, dtype=torch.long),
                torch.tensor(self.present), torch.tensor(float(self.absent_mask)), *_state(targets))


class SyntheticAbsent(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = cv2.equalizeHist(_synthetic_absent(np.random.default_rng()))
        return (torch.from_numpy(img).float().div(255)[None], torch.zeros(SIZE, SIZE, dtype=torch.long),
                torch.tensor(0.), torch.tensor(1.), *_state(NO_STATE))


class WithPresence(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, torch.tensor(1.), torch.tensor(1.), *_state(state_targets(y.numpy()))


def capture_eyes(bin_path, every=3):
    """Both eyes of every `every`-th record, split by the manifest's train/validation passes."""
    import bin_reader as B
    import capture_manifest as CM
    role = CM.read_manifest(bin_path)["role"]
    parts = {0: [], 1: []}
    for k, d in enumerate(B.iter_frames(bin_path, good_only=False)):
        if k % every == 0 and k < len(role) and role[k] in parts:
            parts[int(role[k])] += [B.decode_eye(d["jpeg_left"]), B.decode_eye(d["jpeg_right"])]
    return parts[0], parts[1]


def clip_eyes(paths, val_frac=.25):
    """Clip frames split by time: the last quarter of every clip is held out."""
    train, val = [], []
    for path in paths:
        eyes = np.load(path)["eyes"]
        cut = int(len(eyes) * (1 - val_frac))
        train += [e for pair in eyes[:cut] for e in pair]
        val += [e for pair in eyes[cut:] for e in pair]
    return train, val


@torch.no_grad()
def pseudo_state(net, images, device, batch=256):
    """State targets for unlabeled frames from the segmentation net's own masks, kept only where they are
    clear: the iris is plainly visible, or the eye is plainly closed (no eye pixels at all)."""
    net.eval()
    out = []
    for start in range(0, len(images), batch):
        x = torch.stack([torch.from_numpy(cv2.equalizeHist(np.ascontiguousarray(im))).float().div(255)[None]
                         for im in images[start:start + batch]]).to(device)
        for mask in net(x).argmax(1).cpu().numpy():
            openness, pupil, width = geometry(mask)
            clear = width >= MIN_IRIS_PX or (mask > 0).sum() < 8
            out.append((openness if clear else np.nan, pupil if clear else np.nan, float(np.isfinite(pupil))))
    return np.array(out, np.float32)


@torch.no_grad()
def state_scores(net, ds, device, limit=4000):
    """Presence probability plus state predictions and targets for a dataset."""
    net.eval()
    idx = np.linspace(0, len(ds) - 1, min(limit, len(ds))).astype(int)
    rows = []
    for start in range(0, len(idx), 256):
        items = [ds[int(i)] for i in idx[start:start + 256]]
        x = torch.stack([it[0] for it in items]).to(device)
        presence, openness, pupil, visible = net.eye_state(x)
        pred = torch.stack([torch.sigmoid(presence), openness, pupil, torch.sigmoid(visible)], 1).cpu().numpy()
        target = torch.stack([it[4] for it in items]).numpy()
        ok = torch.stack([it[5] for it in items]).numpy()
        rows.append((pred, target, ok))
    net.train()
    pred, target, ok = (np.concatenate(r) for r in zip(*rows))
    report = {"presence": round(float(pred[:, 0].mean()), 3)}
    for k, name in ((0, "openness_mae"), (1, "pupil_ratio_mae")):
        if ok[:, k].any():
            report[name] = round(float(np.abs(pred[ok[:, k], k + 1] - target[ok[:, k], k]).mean()), 4)
    if ok[:, 2].any():
        report["pupil_visible_acc"] = round(float(((pred[ok[:, 2], 3] > .5) == (target[ok[:, 2], 2] > .5)).mean()), 4)
    return report


def train_state(args):
    """Fine-tune the segmentation net's headset-worn (presence) and eye-state heads."""
    from torch.utils.data import ConcatDataset, WeightedRandomSampler
    torch.manual_seed(C.SEED); np.random.seed(C.SEED); random.seed(C.SEED)
    device = Dev.get_device()
    clips = sorted(Path(args.clips).glob("*.npz"))
    absent_tr, absent_va = clip_eyes([c for c in clips if c.name.startswith("absent")])
    present_tr, present_va = clip_eyes([c for c in clips if c.name.startswith("present")])
    worn_tr, worn_va = [], []
    for capture in args.captures:
        tr, va = capture_eyes(capture)
        worn_tr += tr; worn_va += va
    print(f"absent clips {len(absent_tr)}/{len(absent_va)}  present clips {len(present_tr)}/{len(present_va)}  "
          f"captures {len(worn_tr)}/{len(worn_va)} eye frames (train/val)", flush=True)
    if not absent_tr:
        raise SystemExit("No absent_*.npz clips found; record headset-off clips with eye_viewer.py camera (n).")
    net = load_seg(args.init).to(device)
    labeller = load_seg(args.init).to(device)   # frozen copy labels the unsegmented frames once
    state = {name: pseudo_state(labeller, images, device) for name, images in
             (("worn_tr", worn_tr), ("worn_va", worn_va), ("present_tr", present_tr), ("present_va", present_va))}
    del labeller
    groups = [WithPresence(OpenEDSSeg(args.root, "train")), EyeFrames(worn_tr, 1, True, state=state["worn_tr"]),
              EyeFrames(present_tr, 1, True, state=state["present_tr"]),
              EyeFrames(absent_tr, 0, True, absent_mask=True), SyntheticAbsent(20000)]
    share = [.35, .25, .1, .2, .1]   # sampling share per group, independent of group size
    weights = np.concatenate([np.full(len(g), s / max(len(g), 1)) for g, s in zip(groups, share)])
    loader = DataLoader(ConcatDataset(groups), batch_size=args.batch, num_workers=args.workers, drop_last=True,
                        sampler=WeightedRandomSampler(torch.as_tensor(weights), args.steps * args.batch, replacement=True))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=.1)
    weight = torch.tensor([.5, 1, 1, 2], device=device)
    val_eds = OpenEDSSeg(args.root, "val")
    val_sets = {"openeds": WithPresence(val_eds), "captures (worn)": EyeFrames(worn_va, 1, False, state=state["worn_va"]),
                "present clip": EyeFrames(present_va, 1, False, state=state["present_va"]),
                "absent clip": EyeFrames(absent_va, 0, False)}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for step, (x, y, present, has_mask, target, ok) in enumerate(loader, 1):
        x, y, present, has_mask = x.to(device), y.to(device), present.to(device), has_mask.to(device).bool()
        target, ok = target.to(device), ok.to(device)
        logits, p_logit = net(x, with_presence=True)
        _, openness, pupil, visible = net.eye_state(x)
        loss = F.binary_cross_entropy_with_logits(p_logit, present)
        if has_mask.any():
            loss = loss + F.cross_entropy(logits[has_mask], y[has_mask], weight=weight) + dice_loss(logits[has_mask], y[has_mask])
        if ok[:, 0].any():
            loss = loss + F.smooth_l1_loss(openness[ok[:, 0]], target[ok[:, 0], 0], beta=.05)
        if ok[:, 1].any():
            loss = loss + 4 * F.l1_loss(pupil[ok[:, 1]], target[ok[:, 1], 1])   # ratios span ~0.2-0.7
        if ok[:, 2].any():
            loss = loss + F.binary_cross_entropy_with_logits(visible[ok[:, 2]], target[ok[:, 2], 2])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == args.steps:
            report = {"step": step, "segmentation": evaluate(net, val_eds, device, limit=1000)}
            report.update({name: state_scores(net, ds, device) for name, ds in val_sets.items()})
            line = json.dumps(report)
            print(line, flush=True)
            with open(out / "train.log", "a", encoding="utf-8") as log:
                log.write(line + "\n")
    torch.save({"model": net.state_dict(), "step": args.steps, "validation": report}, out / "seg_state.pt")
    print(f"saved {out / 'seg_state.pt'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("root")
    t.add_argument("--out", default="runs/seg")
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--workers", type=int, default=8)
    pr = sub.add_parser("state", help="fine-tune the headset-worn (presence) and eye-state heads")
    pr.add_argument("root", help="OpenEDS root (keeps segmentation from drifting)")
    pr.add_argument("--init", default="runs/seg/seg_presence.pt")
    pr.add_argument("--clips", default="runs/presence/clips")
    pr.add_argument("--captures", nargs="*", default=[str(p) for p in sorted(
        (Path.home() / "AppData/Roaming/ProjectBabble/ModelData/captures").glob("*/user_cal.bin"))
        if Path(str(p) + ".manifest.json").exists()])
    pr.add_argument("--out", default="runs/seg")
    pr.add_argument("--steps", type=int, default=4000)
    pr.add_argument("--batch", type=int, default=64)
    pr.add_argument("--lr", type=float, default=1e-3)
    pr.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "state":
        train_state(args)


if __name__ == "__main__":
    main()
