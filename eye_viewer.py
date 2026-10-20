"""Eye model visualizer: scrub a capture with its labels, the deployed model's outputs and segmentation, or
watch Baballonia's live OSC output.

  python eye_viewer.py replay CAPTURE/user_cal.bin [--model eye.onnx] [--seg runs/seg/seg_best.pt]
  python eye_viewer.py camera [--index N]     # live headset cameras; close Baballonia first
  python eye_viewer.py live [--port 9001]     # point a Baballonia OSC target at this port

Replay keys: space play/pause, a/d step, A/D jump a block, w/s change plotted channel, q quit.
"""
import argparse
import json
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

import bin_reader as B
import capture_manifest as CM

CHANNELS = ["EyeY", "EyeX", "EyeLid", "EyeWiden", "EyeSquint", "EyeBrow"]
SCALE = 2
PANEL_W = 6 * 128 * SCALE // 2 * 2  # three tiles per eye, two eyes
WHITE, GREY, GREEN, ORANGE, CYAN = (255, 255, 255), (130, 130, 130), (90, 220, 90), (40, 160, 255), (230, 200, 60)
SEG_COLORS = np.array([[0, 0, 0], [200, 120, 60], [60, 200, 60], [60, 60, 255]], np.uint8)  # bg, sclera, iris, pupil


def text(img, s, x, y, color=WHITE, scale=.5):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def bar(img, x, y, w, value, color):
    cv2.rectangle(img, (x, y), (x + w, y + 8), (60, 60, 60), -1)
    if value is not None and np.isfinite(value):
        cv2.rectangle(img, (x, y), (x + int(w * float(np.clip(value, 0, 1))), y + 8), color, -1)


def plot(img, x, y, w, h, series, t_now, window_s=8.0):
    """series: list of (times_s, values, color, label). Values are drawn on a 0..1 scale."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.line(img, (x + w // 2, y), (x + w // 2, y + h), (80, 80, 80), 1)
    for k, (ts, vs, color, label) in enumerate(series):
        ts, vs = np.asarray(ts, float), np.asarray(vs, float)
        keep = (np.abs(ts - t_now) <= window_s / 2) & np.isfinite(vs)
        if keep.sum() > 1:
            px = (x + (ts[keep] - t_now + window_s / 2) / window_s * w).astype(np.int32)
            py = (y + h - np.clip(vs[keep], 0, 1) * h).astype(np.int32)
            cv2.polylines(img, [np.stack([px, py], 1)], False, color, 1, cv2.LINE_AA)
        text(img, label, x + 6 + 130 * k, y + 14, color, .45)


# ---------------------------------------------------------------- models
class EyeModel:
    """The deployed ONNX, fed exactly like Baballonia's ImageCollector (equalized, 4 newest frames, L/R interleaved)."""

    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.names = json.loads(meta["blendshape_names"]) if "blendshape_names" in meta else None
        self.input = self.sess.get_inputs()[0].name

    def run_all(self, frames, timestamps):
        """Predictions for every frame. History restarts after a >200 ms gap, like the app."""
        eq = [(cv2.equalizeHist(l), cv2.equalizeHist(r)) for l, r in frames]
        stacks, history = [], deque(maxlen=4)
        for i, pair in enumerate(eq):
            if i and timestamps[i] - timestamps[i - 1] > 200:
                history.clear()
            history.appendleft(pair)
            # shortcut: pads missing history with the oldest frame; the app outputs nothing for its first 4 frames
            hist = list(history) + [history[-1]] * (4 - len(history))
            stacks.append(np.concatenate([np.stack(p) for p in hist]).astype(np.float32) / 255)
        out = []
        for start in range(0, len(stacks), 256):
            out.append(self.sess.run(None, {self.input: np.stack(stacks[start:start + 256])})[0])
        return np.concatenate(out)


class SegModel:
    def __init__(self, path):
        import torch
        from eye_seg import load_seg
        self.torch = torch
        self.net = load_seg(path).eval()

    def __call__(self, eye):
        return self.run(eye)[0]

    def run(self, eye):
        """(class mask, presence probability or NaN for checkpoints without a presence head)."""
        x = self.torch.from_numpy(cv2.equalizeHist(eye)).float().div(255)[None, None]
        with self.torch.no_grad():
            logits, presence = self.net(x, with_presence=True)
        p = float(self.torch.sigmoid(presence)[0]) if self.net.has_presence else np.nan
        return logits.argmax(1)[0].numpy().astype(np.uint8), p

    def state(self, eye):
        """(openness, pupil ratio, pupil-visible probability) from the eye-state head, or None without one."""
        if not getattr(self.net, "has_state", False):
            return None
        x = self.torch.from_numpy(cv2.equalizeHist(eye)).float().div(255)[None, None]
        with self.torch.no_grad():
            _, openness, pupil, visible = self.net.eye_state(x)
        return float(openness[0]), float(pupil[0]), float(self.torch.sigmoid(visible)[0])


def default_seg():
    root = Path(__file__).parent / "runs"
    for candidate in (root / "state/seg_state.pt", root / "seg/seg_presence.pt", root / "seg/seg_best.pt"):
        if candidate.exists():
            return str(candidate)
    return str(root / "seg/seg_best.pt")


class PresenceGate:
    """Preview of the app-side gate: not worn after both eyes stay low for `off_s`, worn again after `on_s` high."""

    def __init__(self, low=.3, high=.7, off_s=.5, on_s=.2):
        self.low, self.high, self.off_s, self.on_s = low, high, off_s, on_s
        self.worn, self.since = True, None

    def update(self, presences, now):
        p = max(presences)
        flip = (p < self.low) if self.worn else (p > self.high)
        if not flip:
            self.since = None
        elif self.since is None:
            self.since = now
        elif now - self.since >= (self.off_s if self.worn else self.on_s):
            self.worn, self.since = not self.worn, None
        return self.worn


SETTINGS = Path.home() / "AppData/Roaming/ProjectBabble/ApplicationData/LocalSettings.json"


def app_settings():
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def default_model():
    """The eye model Baballonia is configured to use, else the newest one on disk."""
    configured = app_settings().get("EyeHome_EyeModel")
    if configured and Path(configured).is_file():
        return Path(configured)
    models = sorted((Path.home() / "AppData/Roaming/ProjectBabble/Models").glob("*.onnx"), key=lambda p: p.stat().st_mtime)
    return models[-1] if models else None


# ---------------------------------------------------------------- replay
def load_capture(path):
    manifest = CM.read_manifest(path)
    frames, labels, valid, times, raw = [], [], [], [], []
    for d in B.iter_frames(path, good_only=False):
        frames.append((B.decode_eye(d["jpeg_left"]), B.decode_eye(d["jpeg_right"])))
        left, right, _ = B.frame_targets_dual(d, mirror_side=None)
        labels.append(np.concatenate([left, right]))  # binary left, binary right
        times.append(float(d["timestamp"]))
        raw.append(d)
    n = len(frames)
    labels = np.array(labels)
    if manifest:
        # Manifest masks are semantic [right, left]; binary-left is semantic-right.
        valid = manifest["masks"][:n] & ~manifest["excluded"][:n, None]
        times = manifest["timestamps"][:n]
        routine = np.array([manifest["routines"][b] if b >= 0 else "" for b in manifest["block"][:n]])
        role = manifest["role"][:n]
        block = manifest["block"][:n]
    else:
        valid = np.ones((n, 12), bool)
        routine, role, block = np.array([""] * n), np.full(n, -1), np.zeros(n, int)
    return frames, labels, valid, np.asarray(times, float), routine, role, block


def model_columns(names):
    """Index of each [binary-left 6, binary-right 6] label column in the model output."""
    if names is None:
        return None
    lookup = {n.lstrip("/").lower(): i for i, n in enumerate(names)}
    cols = []
    for eye in ("right", "left"):  # binary-left is semantic-right
        for ch in CHANNELS:
            cols.append(lookup.get(f"{eye}{ch}".lower()))
    return cols


def replay(args):
    frames, labels, valid, times, routine, role, block = load_capture(args.bin)
    n = len(frames)
    print(f"{n} frames, {len(set(block.tolist()))} blocks", flush=True)
    model = EyeModel(args.model) if args.model else None
    preds = cols = None
    if model:
        print(f"Running {Path(args.model).name} on all frames...", flush=True)
        preds, cols = model.run_all(frames, times), model_columns(model.names)
    seg = SegModel(args.seg) if args.seg and Path(args.seg).exists() else None
    t_s = (times - times[0]) / 1000
    roles = {0: "train", 1: "validation", 2: "test"}

    def draw(pos, channel):
        canvas = np.zeros((128 * SCALE + 316, 6 * 128 * SCALE, 3), np.uint8)
        geo = []
        for e in range(2):
            eye = frames[pos][e]
            tiles = [eye, cv2.equalizeHist(eye)]
            if seg:
                mask = seg(eye)
                from eye_seg import geometry
                geo.append(geometry(mask))
                over = cv2.addWeighted(cv2.cvtColor(tiles[1], cv2.COLOR_GRAY2BGR), .6, SEG_COLORS[mask], .4, 0)
            else:
                over = np.zeros((128, 128, 3), np.uint8)
                text(over, "no --seg", 30, 64, GREY, .4)
            tiles = [cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) for t in tiles] + [over]
            for k, tile in enumerate(tiles):
                x = (e * 3 + k) * 128 * SCALE
                canvas[:128 * SCALE, x:x + 128 * SCALE] = cv2.resize(tile, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
        top = 128 * SCALE
        text(canvas, "binary-left (semantic right): raw | model input | segmentation", 8, top + 18, GREY, .45)
        text(canvas, "binary-right (semantic left)", 3 * 128 * SCALE + 8, top + 18, GREY, .45)
        text(canvas, f"#{pos}/{n - 1}  t={t_s[pos]:.2f}s  block={block[pos]} {routine[pos]} ({roles.get(int(role[pos]), '-')})",
             8, top + 42)
        for e in range(2):
            x0 = 8 + e * 3 * 128 * SCALE
            mask_offset = 0 if e == 0 else 6  # semantic order: right first
            for c, name in enumerate(CHANNELS):
                y = top + 64 + c * 18
                col = e * 6 + c
                ok = bool(valid[pos, mask_offset + c])
                text(canvas, ("> " if c == channel else "  ") + name, x0, y + 8, WHITE if ok else GREY, .45)
                bar(canvas, x0 + 110, y, 150, labels[pos, col] if ok else None, GREEN)
                if preds is not None and cols[col] is not None:
                    bar(canvas, x0 + 270, y, 150, preds[pos, cols[col]], ORANGE)
            text(canvas, "label", x0 + 110, top + 58, GREEN, .4)
            text(canvas, "model", x0 + 270, top + 58, ORANGE, .4)
            if geo:
                o, p, w = geo[e]
                text(canvas, f"seg openness {o:.2f}   pupil/iris {p:.2f}   iris {w:.0f}px", x0, top + 180, CYAN, .45)
        e_col = [channel, 6 + channel]
        series = []
        for e, col in enumerate(e_col):
            lab = np.where(valid[:, (0 if e == 0 else 6) + channel], labels[:, col], np.nan)
            series.append((t_s, lab, GREEN if e == 0 else (60, 140, 60), f"label {'bL' if e == 0 else 'bR'}"))
            if preds is not None and cols[col] is not None:
                series.append((t_s, preds[:, cols[col]], ORANGE if e == 0 else (30, 100, 180), f"model {'bL' if e == 0 else 'bR'}"))
        plot(canvas, 8, top + 190, canvas.shape[1] - 16, 100, series, t_s[pos])
        text(canvas, f"plot: {CHANNELS[channel]} (w/s)   space play  a/d step  A/D block  q quit", 8, canvas.shape[0] - 6, GREY, .4)
        return canvas

    if args.snapshot:
        cv2.imwrite(args.snapshot, draw(min(args.frame, n - 1), 2))
        return
    pos, playing, channel = 0, False, 2
    win = "Babble eye viewer"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("frame", win, 0, max(1, n - 1), lambda v: None)
    while True:
        pos = cv2.getTrackbarPos("frame", win) if not playing else pos
        cv2.imshow(win, draw(pos, channel))
        key = cv2.waitKey(15 if playing else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        step = {ord("d"): 1, ord("a"): -1}.get(key, 0)
        if key == ord(" "):
            playing = not playing
        elif key in (ord("w"), ord("s")):
            channel = (channel + (1 if key == ord("s") else -1)) % len(CHANNELS)
        elif key in (ord("D"), ord("A")):
            same = block == block[pos]
            target = np.flatnonzero(~same & (np.arange(n) > pos)) if key == ord("D") else np.flatnonzero(~same & (np.arange(n) < pos))
            if len(target):
                pos = int(target[0] if key == ord("D") else np.flatnonzero(block == block[target[-1]])[0])
        if playing:
            step = 1
        pos = int(np.clip(pos + step, 0, n - 1))
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:  # closed with the window's X
            break
        cv2.setTrackbarPos("frame", win, pos)
    cv2.destroyAllWindows()


class BlinkShaper:
    """Mirror of Baballonia's BlinkShaper.cs (keep the two identical) for previewing blinks live."""

    def __init__(self, drop=.2, window=.15, max_blink=.4, recover=.08, refractory=.6,
                 close=.06, hold=.12, release=.15, smoothing=.15):
        self.drop, self.window, self.max_blink, self.recover, self.refractory = drop, window, max_blink, recover, refractory
        self.close, self.hold, self.release, self.smoothing = close, hold, release, smoothing
        self.recent, self.candidate = deque(), None
        self.start = self.last = -np.inf
        self.left = self.right = np.nan
        self.blinking = False

    def _envelope(self, since):
        smooth = lambda x: x * x * (3 - 2 * x)
        if since < 0: return 1.0
        if since < self.close: return 1 - smooth(since / self.close)
        since -= self.close
        if since < self.hold: return 0.0
        since -= self.hold
        return smooth(since / self.release) if since < self.release else 1.0

    def update(self, raw_left, raw_right, t):
        if not (np.isfinite(raw_left) and np.isfinite(raw_right)):
            return raw_left, raw_right
        elapsed = t - self.last
        if elapsed <= 0 or elapsed > .5:
            self.left, self.right = raw_left, raw_right
            self.recent.clear(); self.candidate = None
        else:
            a = 1 - np.exp(-elapsed / self.smoothing)
            self.left += a * (raw_left - self.left); self.right += a * (raw_right - self.right)
        self.last = t
        self.recent.append((t, raw_left, raw_right))
        while self.recent[0][0] < t - self.window:
            self.recent.popleft()
        peak_l, peak_r = max(r[1] for r in self.recent), max(r[2] for r in self.recent)
        if self.candidate is not None:
            c0, cl, cr = self.candidate
            if t - c0 > self.max_blink:
                self.candidate = None
            elif raw_left >= cl - self.recover and raw_right >= cr - self.recover:
                self.start, self.candidate = t, None
        else:
            dl, dr = peak_l - raw_left, peak_r - raw_right
            if max(dl, dr) >= self.drop and min(dl, dr) >= self.drop / 2 and t - self.start > self.refractory:
                self.candidate = (t, peak_l, peak_r)
        env = self._envelope(t - self.start)
        self.blinking = env < 1
        return min(self.left, env), min(self.right, env)


# ---------------------------------------------------------------- live camera
def transform_eye(half, cs, size=128):
    """Baballonia's ImageTransformer: ROI crop, grayscale, linear gain, then rotate/flip warp or resize."""
    h, w = half.shape[:2]
    roi = cs.get("Roi") or {}
    x, y, rw, rh = (int(roi.get(k, -1)) for k in ("X", "Y", "Width", "Height"))
    # The app uses the whole half for invalid ROIs and for ROIs exactly as wide or tall as the half.
    if not (x < 0 or y < 0 or rw <= 0 or rh <= 0 or x + rw > w or y + rh > h or rw == w or rh == h):
        half = half[y:y + rh, x:x + rw]
    if half.ndim == 3:
        half = half[:, :, 0].copy() if cs.get("UseRedChannel") else cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
    gain = float(cs.get("Gamma", 1) or 1)
    if gain != 1:
        half = np.clip(half.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    rot, hf, vf = float(cs.get("RotationRadians", 0) or 0), cs.get("UseHorizontalFlip"), cs.get("UseVerticalFlip")
    if rot != 0 or hf or vf:
        c, s = np.cos(rot), np.sin(rot)
        k = 1 / (abs(c) + abs(s))
        hs, vs = (-k if hf else k), (-k if vf else k)
        m = np.array([[size / half.shape[1] * c * hs, size / half.shape[0] * s * hs, (size - (size * c + size * s) * hs) / 2],
                      [-size / half.shape[1] * s * vs, size / half.shape[0] * c * vs, (size + (size * s - size * c) * vs) / 2]])
        return cv2.warpAffine(half, m, (size, size))
    return cv2.resize(half, (size, size))


def find_split_camera(max_index=8):
    """Index of the first DirectShow camera with a 2:1 side-by-side frame (the Beyond's both-eye stream)."""
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok:
            h, w = frame.shape[:2]
            print(f"  camera {i}: {w}x{h}", flush=True)
            if abs(w / h - 2) < .05:
                return i
    return None


def camera(args):
    settings = app_settings()
    index = args.index if args.index is not None else find_split_camera()
    if index is None:
        raise SystemExit("No 2:1 side-by-side eye camera found. Close Baballonia (it holds the camera) or pass --index.")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Camera {index} could not be opened. Close Baballonia, which holds it exclusively.")
    model = EyeModel(args.model) if args.model else None
    cols = model_columns(model.names) if model else None
    seg = SegModel(args.seg) if args.seg and Path(args.seg).exists() else None
    from eye_seg import geometry
    print(f"camera {index}  model {Path(args.model).name if args.model else '-'}  seg {'on' if seg else 'off'}", flush=True)
    history, trace, gate = deque(maxlen=4), deque(maxlen=600), PresenceGate()
    shaper, blinks = BlinkShaper(drop=args.blink_threshold), 0
    lid_cols = (model.names.index("leftEyeLid"), model.names.index("rightEyeLid")) if model and model.names else None
    t0, last, fps = time.monotonic(), time.monotonic(), 0.0
    win, channel = "Babble live camera", 2
    recording, clip, record_dir = None, [], Path(args.record_dir)

    def save_clip():
        if clip:
            record_dir.mkdir(parents=True, exist_ok=True)
            path = record_dir / f"{recording}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
            # eyes are in model order (binary-left, binary-right), 128x128 uint8 after the app's crop.
            np.savez_compressed(path, eyes=np.array([c[1] for c in clip], np.uint8),
                                t=np.array([c[0] for c in clip]), label=recording)
            print(f"saved {len(clip)} frames -> {path}", flush=True)
        clip.clear()
    while True:
        ok, frame = cap.read()
        if not ok:
            if cv2.waitKey(100) & 0xFF in (ord("q"), 27):
                break
            continue
        now = time.monotonic()
        fps = .9 * fps + .1 / max(now - last, 1e-3)
        last = now
        h, w = frame.shape[:2]
        halves = (frame[:, :w // 2], frame[:, w // 2:])
        left_half = transform_eye(halves[0], settings.get("LeftCamera") or {})
        right_half = transform_eye(halves[1], settings.get("RightCamera") or {})
        eyes = (right_half, left_half)  # model order: binary-left (right half) first, like ImageCollector's swap
        history.appendleft(tuple(cv2.equalizeHist(e) for e in eyes))
        pred = None
        if model and len(history) == 4:
            x = np.concatenate([np.stack(p) for p in history]).astype(np.float32)[None] / 255
            pred = model.sess.run(None, {model.input: x})[0][0]
        segs = [seg.run(e) if seg else (None, np.nan) for e in eyes]
        geo = [geometry(m) if m is not None else (np.nan, np.nan, 0) for m, _ in segs]
        presence = [p for _, p in segs]
        worn = gate.update(presence, now) if all(np.isfinite(presence)) else None
        shaped = None
        if pred is not None and lid_cols:
            was = shaper.blinking
            # Openness as the app sees it after ProcessExpressions (1 = open), semantic left then right.
            shaped = shaper.update(1 - pred[lid_cols[0]], 1 - pred[lid_cols[1]], now)
            blinks += shaper.blinking and not was
        trace.append((now - t0, pred, geo, shaped))
        if recording:
            clip.append((now - t0, np.stack(eyes)))

        # Full frame with each eye's crop box, so lashes/fit problems are visible in context.
        view = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        for k, name in enumerate(("LeftCamera", "RightCamera")):
            roi = (settings.get(name) or {}).get("Roi") or {}
            rx, ry, rw, rh = (int(roi.get(q, -1)) for q in ("X", "Y", "Width", "Height"))
            ox = k * (w // 2)
            if rw > 0 and rh > 0 and rx >= 0 and ry >= 0:
                cv2.rectangle(view, (ox + rx, ry), (ox + rx + rw, ry + rh), (40, 160, 255), 2)
        scale = 256 / h
        view = cv2.resize(view, (int(w * scale), 256))
        canvas = np.zeros((256 + 128 * SCALE + 262, max(view.shape[1], 6 * 128 * SCALE), 3), np.uint8)
        canvas[:256, :view.shape[1]] = view
        text(canvas, f"camera {index} {w}x{h}  {fps:.0f} fps   orange boxes: crops from Baballonia settings", 8, 248, WHITE, .45)
        text(canvas, f"blinks: {blinks}" + ("   BLINK" if shaper.blinking else ""), canvas.shape[1] - 560, 70,
             (80, 80, 255) if shaper.blinking else WHITE, .6)
        if worn is not None:
            state, color = ("WORN", GREEN) if worn else ("NOT WORN - outputs neutral", (60, 60, 255))
            text(canvas, f"presence bL {presence[0]:.2f}  bR {presence[1]:.2f}   gate: {state}", canvas.shape[1] - 560, 24, color, .6)
            for e in range(2):
                bar(canvas, canvas.shape[1] - 560 + e * 160, 34, 140, presence[e], color)
        top = 256
        for e in range(2):
            mask = segs[e][0]
            eq = cv2.equalizeHist(eyes[e])
            tiles = [eyes[e], eq]
            over = cv2.addWeighted(cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR), .6, SEG_COLORS[mask], .4, 0) if mask is not None \
                else np.zeros((128, 128, 3), np.uint8)
            for k, tile in enumerate([cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) for t in tiles] + [over]):
                x0 = (e * 3 + k) * 128 * SCALE
                canvas[top:top + 128 * SCALE, x0:x0 + 128 * SCALE] = cv2.resize(tile, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
        y0 = top + 128 * SCALE
        for e in range(2):
            x0 = 8 + e * 3 * 128 * SCALE
            text(canvas, ("binary-left / right half (semantic right)", "binary-right / left half (semantic left)")[e], x0, y0 + 16, GREY, .45)
            for c, name in enumerate(CHANNELS):
                yy = y0 + 28 + c * 16
                text(canvas, ("> " if c == channel else "  ") + name, x0, yy + 8, WHITE, .42)
                col = cols[e * 6 + c] if cols else None
                bar(canvas, x0 + 110, yy, 200, pred[col] if pred is not None and col is not None else None, ORANGE)
            o, p, iw = geo[e]
            text(canvas, f"open {o:.2f}  pupil/iris {p:.2f}  iris {iw:.0f}px", x0 + 330, y0 + 44, CYAN, .45)
            st = seg.state(eyes[e]) if seg else None
            if st:   # what the exported model reports: its own openness and pupil, gated by pupil visibility
                text(canvas, f"model: open {st[0]:.2f}  pupil {st[1]:.2f}" + ("" if st[2] > .5 else " (hidden)"),
                     x0 + 330, y0 + 76, (120, 200, 255), .45)
            bar(canvas, x0 + 330, y0 + 56, 150, o / 1.2 if np.isfinite(o) else None, CYAN)
        ts = [s[0] for s in trace]
        series = []
        for e, (cm, cg) in enumerate(((ORANGE, CYAN), ((30, 100, 180), (150, 130, 40)))):
            col = cols[e * 6 + channel] if cols else None
            if col is not None:
                series.append((ts, [s[1][col] if s[1] is not None else np.nan for s in trace], cm, f"model {'bL' if e == 0 else 'bR'}"))
            if channel == 2:  # compare lid closure with segmentation (1 - openness, scaled)
                series.append((ts, [1 - min(s[2][e][0] / .8, 1) if np.isfinite(s[2][e][0]) else np.nan for s in trace], cg,
                               f"1-seg open {'bL' if e == 0 else 'bR'}"))
        if channel == 2 and lid_cols:  # what the game receives after blink shaping (closure, semantic left eye)
            series.append((ts, [1 - s[3][0] if s[3] is not None else np.nan for s in trace], (80, 80, 255), "shaped (sent)"))
        plot(canvas, 8, canvas.shape[0] - 120, canvas.shape[1] - 16, 100, series, ts[-1] - 4, window_s=8)
        text(canvas, f"plot: {CHANNELS[channel]} (w/s)   record: n = no eyes / headset off, p = worn, again to stop   q quit",
             8, canvas.shape[0] - 6, GREY, .4)
        if recording:
            cv2.circle(canvas, (canvas.shape[1] - 20, 20), 8, (0, 0, 255), -1)
            text(canvas, f"REC {recording} {len(clip)}", canvas.shape[1] - 200, 26, (80, 80, 255), .55)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (ord("w"), ord("s")):
            channel = (channel + (1 if key == ord("s") else -1)) % len(CHANNELS)
        elif key in (ord("n"), ord("p")):
            label = "absent" if key == ord("n") else "present"
            save_clip()
            recording = None if recording == label else label
    save_clip()
    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------- live OSC
def live(args):
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import ThreadingOSCUDPServer
    history = defaultdict(lambda: deque(maxlen=4000))
    lock = threading.Lock()
    t0 = time.monotonic()

    def handle(address, *values):
        floats = [float(v) for v in values if isinstance(v, (int, float))]
        if floats:
            with lock:
                history[address].append((time.monotonic() - t0, floats))

    dispatcher = Dispatcher()
    dispatcher.set_default_handler(handle)
    server = ThreadingOSCUDPServer((args.host, args.port), dispatcher)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Listening for OSC on {args.host}:{args.port}. Add this as an OSC target in Baballonia.", flush=True)
    win = "Babble live OSC"
    selected = 0
    while True:
        now = time.monotonic() - t0
        with lock:
            items = sorted((k, list(v)) for k, v in history.items())
        rows = max(1, len(items))
        canvas = np.zeros((90 + 18 * rows + 140, 1180, 3), np.uint8)
        text(canvas, f"{len(items)} addresses on port {args.port}   w/s select   q quit", 8, 20, GREY, .45)
        for i, (address, samples) in enumerate(items):
            y = 40 + i * 18
            last_t, last = samples[-1]
            stale = now - last_t > 1
            text(canvas, ("> " if i == selected else "  ") + address, 8, y + 8, GREY if stale else WHITE, .42)
            for k, v in enumerate(last[:4]):
                # Gaze-like values arrive in -1..1 or degrees; show them centred on the bar.
                shown = v if 0 <= v <= 1 else (v / 90 + .5 if abs(v) > 1 else v / 2 + .5)
                bar(canvas, 420 + k * 185, y, 110, shown, ORANGE)
                text(canvas, f"{v:+.3f}", 536 + k * 185, y + 8, WHITE, .38)
        if items:
            selected %= len(items)
            address, samples = items[selected]
            ts = [s[0] for s in samples]
            series = [(ts, [s[1][k] if 0 <= s[1][k] <= 1 else s[1][k] / 2 + .5 for s in samples], c, f"{address}[{k}]")
                      for k, c in zip(range(min(4, len(samples[-1][1]))), (ORANGE, GREEN, CYAN, WHITE))]
            plot(canvas, 8, canvas.shape[0] - 130, canvas.shape[1] - 16, 120, series, now - 5, window_s=10)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            selected += 1
        elif key == ord("w"):
            selected = max(0, selected - 1)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    server.shutdown()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("bin")
    r.add_argument("--model", default=default_model())
    r.add_argument("--seg", default=default_seg())
    r.add_argument("--snapshot", help="write one rendered frame to this PNG and exit")
    r.add_argument("--frame", type=int, default=0)
    cam = sub.add_parser("camera", help="live headset eye cameras through the model (close Baballonia first)")
    cam.add_argument("--index", type=int, help="DirectShow camera index; default finds the 2:1 side-by-side eye camera")
    cam.add_argument("--model", default=default_model())
    cam.add_argument("--seg", default=default_seg())
    cam.add_argument("--blink-threshold", type=float, default=.2, help="same meaning as the app's Blink threshold")
    cam.add_argument("--record-dir", default=str(Path(__file__).parent / "runs/presence/clips"))
    lv = sub.add_parser("live")
    lv.add_argument("--host", default="127.0.0.1")
    lv.add_argument("--port", type=int, default=9001)
    args = ap.parse_args()
    {"replay": replay, "camera": camera, "live": live}[args.cmd](args)


if __name__ == "__main__":
    main()
