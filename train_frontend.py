import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("BACKBONE", "mobilenetv4_conv_small_050.e3000_r224_in1k")

import config as C
import device as Dev
import bin_temporal_dataset as TD
from train_dual import fit_expr_stages
from train_gaze import fit_gaze
from export_dual_onnx import export


def app_dir() -> Path:
    """Directory the app lives in: next to the .exe when frozen (PyInstaller), else next to this script."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def _first_existing(candidates):
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None


def find_expr_pretrain():
    """Locate the EXPRESSION warm-start checkpoint (4-ch temporal pretrain) shipped next to the exe
    (the dev tree resolves it next to this script). None -> train the expr net from scratch."""
    return _first_existing([app_dir() / "model_best.pt"])


def find_footage():
    """Locate the in-domain UNLABELED footage npz for the per-user expr anti-jitter stream (Mean-Teacher
    + temporal). Recorded on the same setup as the user_cal bins, so it is in-domain for the deployment.
    None -> the per-user tune runs clean (no AJ)."""
    return _first_existing([
        app_dir() / "footage_unlabeled.npz",            # next to the exe (bundled)
        getattr(sys, "_MEIPASS", None) and Path(sys._MEIPASS) / "footage_unlabeled.npz",
        C.FOOTAGE_UNLABELED_PATH,                        # dev tree
    ])


def find_gaze_pretrain():
    """Locate the GAZE (MicroChad) warm-start checkpoint shipped next to the exe (the dev tree resolves
    it next to this script). None -> train gaze from scratch (BabbleTrainer also trains it per-capture)."""
    return _first_existing([app_dir() / "gaze_model_best.pt"])

STEPS = C.EXPR_STEPS  
GAZE_STEPS = C.GAZE_STEPS    
BATCH = 64
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WARMUP = 200
VAL_EVERY = 1000
VAL_FRAC = 0.1


def selftest() -> int:
    """Report the training device that would be used (CUDA / DirectML / CPU) and run a tiny GPU op,
    then exit. Lets you confirm the build picks up the GPU before committing to a long training run."""
    import torch
    import device as Dev
    dev = Dev.get_device()
    print(f"device          : {dev}", flush=True)
    print(f"torch           : {torch.__version__}  (cuda runtime {torch.version.cuda})", flush=True)
    print(f"cuda_available  : {torch.cuda.is_available()}", flush=True)
    if getattr(dev, "type", None) == "cuda":
        print(f"gpu             : {torch.cuda.get_device_name(dev.index or 0)}", flush=True)
        print(f"bf16_supported  : {torch.cuda.is_bf16_supported()}", flush=True)
        x = torch.randn(2048, 2048, device=dev)
        with Dev.autocast_ctx(dev):
            r = float((x @ x).sum())        # forces a real kernel + device sync
        print(f"gpu_matmul_ok   : {r == r}  (autocast dtype {Dev._amp_dtype(dev)})", flush=True)
    print("selftest OK", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Train the dual-eye model on one .bin and export an ONNX.")
    ap.add_argument("bin", nargs="?", help="input capture .bin")
    ap.add_argument("onnx", nargs="?", help="output model .onnx")
    ap.add_argument("--steps", type=int, default=STEPS,
                    help=f"expression-loop steps (default {STEPS})")
    ap.add_argument("--gaze-steps", type=int, default=GAZE_STEPS,
                    help=f"gaze-loop steps (default {GAZE_STEPS}; BabbleTrainer's per-capture budget)")
    ap.add_argument("--selftest", action="store_true",
                    help="report the selected training device (CUDA/DirectML/CPU) and exit")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.bin or not args.onnx:
        ap.error("the following arguments are required: bin, onnx")
    bin_path, out_onnx = args.bin, args.onnx

    Dev.seed_everything(C.SEED)
    tmp = tempfile.mkdtemp(prefix="qpro-train-")
    index = os.path.join(tmp, "index.npz")
    try:
        TD.build_index(bin_paths=[bin_path], index_path=index)
        train_expr = TD.BinTemporalDataset("train", index_path=index, split_mode="frame",
                                           val_frac=VAL_FRAC, aug_mode="expr", mixup=True,
                                           exclude_expr_unlabeled=True)
        train_gaze = TD.BinTemporalDataset("train", index_path=index, split_mode="frame",
                                           val_frac=VAL_FRAC, aug_mode="gaze")
        val_expr = TD.BinTemporalDataset("val", index_path=index, split_mode="frame", val_frac=VAL_FRAC,
                                         augment=False, exclude_expr_unlabeled=True)
        val_gaze = TD.BinTemporalDataset("val", index_path=index, split_mode="frame", val_frac=VAL_FRAC,
                                         augment=False)
        train_paired = TD.BinPairedEyeDataset("train", index_path=index, split_mode="frame",
                                              val_frac=VAL_FRAC, exclude_expr_unlabeled=True)

        expr_init = find_expr_pretrain()
        gaze_init = find_gaze_pretrain()
        if expr_init:
            print(f"warm-start expr: {expr_init}", flush=True)
        if gaze_init:
            print(f"warm-start gaze: {gaze_init}", flush=True)

        expr_unlabeled = None
        try:
            expr_unlabeled = TD.FootageUnlabeledDataset.from_bin(bin_path)
            print("expr anti-jitter: bin free-expression gaze section", flush=True)
        except Exception as e:
            footage = find_footage()
            if footage:
                expr_unlabeled = TD.FootageUnlabeledDataset(footage)
                print(f"expr anti-jitter footage (fallback npz): {footage}", flush=True)
            else:
                print(f"expr anti-jitter: none ({e}) -> clean tune", flush=True)

        _sa, _sb = C.EXPR_STAGE_A_STEPS, C.EXPR_STAGE_B_STEPS
        sa = max(1, round(args.steps * _sa / (_sa + _sb)))
        sb = max(1, args.steps - sa)
        total = sa + sb + args.gaze_steps
        expr_ckpt, _ = fit_expr_stages(
            train_expr, val_expr, batch=BATCH, val_every=VAL_EVERY,
            pretrained=(expr_init is None), init=expr_init,
            unlabeled_ds=expr_unlabeled, paired_ds=train_paired,
            deconfuse=True, even=True, consensus=C.DUAL_CONSENSUS_HEAD,
            lambda_cons=C.EXPR_TUNE_LAMBDA_CONS, stage_a_steps=sa, stage_b_steps=sb,
            ckpt_dir=Path(tmp) / "expr", log="frontend",
            progress_offset=0, progress_total=total, emit_header=True, emit_completion=False)
        gaze_ckpt, _ = fit_gaze(train_gaze, val_gaze, steps=args.gaze_steps, init=gaze_init,
                                ckpt_dir=Path(tmp) / "gaze", log="frontend",
                                progress_offset=args.steps, progress_total=total,
                                emit_header=False, emit_completion=True)

        if os.environ.get("KEEP_CKPT"):
            base = os.path.splitext(out_onnx)[0]
            shutil.copyfile(expr_ckpt, base + ".expr.pt")
            shutil.copyfile(gaze_ckpt, base + ".gaze.pt")
            print(f"kept ckpts: {base}.expr.pt + {base}.gaze.pt (re-export with different fuse settings)",
                  flush=True)

        export(str(gaze_ckpt), str(expr_ckpt), out_onnx, check=False)
        print("Model exported to ONNX: " + out_onnx, flush=True)
        print("\nTraining completed successfully!\n", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support() 
    main()
