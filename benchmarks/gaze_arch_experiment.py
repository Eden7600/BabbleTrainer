"""Development comparison of gaze-net heads on recorded sessions. Not a promotion result (see README).

Each variant fine-tunes from the shipped gaze checkpoint on one session's train blocks, selects on that
session's validation blocks, and is scored on the other session's validation blocks (a headset refit).

  python benchmarks/gaze_arch_experiment.py CAPTURE_DIR CAPTURE_DIR --init dist/.../gaze_model_best.pt
"""
import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import config as C
import device as Dev
import bin_temporal_dataset as TD
import evaluation as Eval
import train_gaze
from model_dual import MicroChad, GAZE_OUT


VARIANTS = {"maxpool": lambda: MicroChad(out_count=GAZE_OUT), "spatial": lambda: MicroChad(out_count=GAZE_OUT, spatial=True)}


def warm_start(net, path):
    src = torch.load(path, map_location="cpu", weights_only=True)
    src = src.get("gaze", src)
    own = net.state_dict()
    # A head whose shape changed starts fresh; every conv layer keeps the shipped weights.
    net.load_state_dict({k: v for k, v in src.items() if own[k].shape == v.shape}, strict=False)


def session(capture_dir, tmp):
    capture_dir = Path(capture_dir)
    bin_path = capture_dir / "user_cal.bin"
    index = Path(tmp) / f"{capture_dir.name}.npz"
    TD.build_index([bin_path], index)
    kw = dict(index_path=index, split_mode="block", val_frac=.1)
    return dict(name=capture_dir.name[:8], train=TD.BinTemporalDataset("train", aug_mode="gaze", **kw),
                val=TD.BinTemporalDataset("val", augment=False, **kw))


def gaze_mae(net, ds, device):
    report = Eval.evaluate_gaze_report(net.eval(), ds, device)
    axes = {k: round(v.get("balanced_mae", v["mae"]) * 2 * C.GAZE_RANGE_DEG, 3)
            for k, v in report["channels"].items() if v["mae"] is not None}
    return round(Eval.score(report, "gaze") * 2 * C.GAZE_RANGE_DEG, 3), axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs=2)
    ap.add_argument("--init", required=True)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=C.GAZE_STEPS)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--output", default="gaze-arch-results.json")
    args = ap.parse_args()
    device = Dev.get_device()
    results = []
    with tempfile.TemporaryDirectory(prefix="gaze-arch-") as tmp:
        sessions = [session(c, tmp) for c in args.captures]
        with torch.no_grad():
            shipped = Dev.model_to(VARIANTS["maxpool"](), device)
            warm_start(shipped, args.init)
            for s in sessions:
                results.append(dict(variant="shipped", seed=None, test=s["name"], mae=gaze_mae(shipped, s["val"], device)))
        for fit, test in ((sessions[0], sessions[1]), (sessions[1], sessions[0])):
            for variant in args.variants:
                for seed in range(args.seeds):
                    torch.manual_seed(C.SEED + seed); np.random.seed(C.SEED + seed); random.seed(C.SEED + seed)
                    train_gaze.build_gaze_net, train_gaze.warm_start_gaze = VARIANTS[variant], warm_start
                    ckpt_dir = Path(tmp) / f"{fit['name']}-{variant}-{seed}"
                    train_gaze.fit_gaze(fit["train"], fit["val"], steps=args.steps, batch=args.batch,
                                        init=args.init, ckpt_dir=ckpt_dir, log="plain")
                    net = VARIANTS[variant]()
                    net.load_state_dict(torch.load(ckpt_dir / "model_best.pt", map_location="cpu", weights_only=True)["gaze"])
                    net = Dev.model_to(net, device)
                    with torch.no_grad():
                        row = dict(variant=variant, seed=seed, fit=fit["name"],
                                   same_session=gaze_mae(net, fit["val"], device),
                                   cross_session=gaze_mae(net, test["val"], device))
                    print(json.dumps(row), flush=True)
                    results.append(row)
                    Path(args.output).write_text(json.dumps(results, indent=2))
        for s in sessions:
            s["train"].close(); s["val"].close()
    for variant in args.variants:
        for key in ("same_session", "cross_session"):
            v = [r[key][0] for r in results if r["variant"] == variant]
            print(f"{variant:8s} {key:14s} mean {np.mean(v):.2f} deg  sd {np.std(v):.2f}  n={len(v)}")


if __name__ == "__main__":
    main()
