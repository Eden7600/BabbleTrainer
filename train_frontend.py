import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

os.environ.setdefault("BACKBONE", "mobilenetv4_conv_small_050.e3000_r224_in1k")
import training_progress as Progress

CAPABILITIES = {"schema_version": 1, "protocol_version": 1, "capture_schema_version": 1,
                "progress_json": True, "artifact_manifest": "training-artifact.json",
                "independent_blocks": True, "explicit_legacy": True,
                "default_device": "auto", "auto_device_priority": ["cuda", "directml", "cpu"],
                "review_schema_version": 1, "temporal_exclusions": False}


def app_dir():
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent


def expression_coverage(train, validation, paired_validation=None):
    """Require both endpoints in each eye/partition and actual merged validation."""
    import numpy as np
    import config as C
    coverage, preserve = {}, []
    sources = [("train", train, None), ("validation", validation, None)]
    if paired_validation is not None:
        paired_rows = np.zeros(len(paired_validation.base.label), bool)
        for _, _, canonical, mirror in paired_validation.pairs:
            paired_rows[canonical] = paired_rows[mirror] = True
        sources.append(("validation_paired", paired_validation.base, paired_rows))
    for channel in range(2, 6):
        complete, partitions = True, {}
        for name, dataset, paired_rows in sources:
            sides = {}
            for side in (0, 1):
                valid = (dataset.side == side) & dataset.label_mask[:, channel]
                if paired_rows is not None:
                    valid &= paired_rows
                values = dataset.label[valid, channel]
                low, high = int(np.sum(values <= .2)), int(np.sum(values >= .8))
                sides[str(side)] = dict(low_endpoint=low, high_endpoint=high, valid=int(len(values)))
                complete &= low > 0 and high > 0
            partitions[name] = sides
        coverage[C.DUAL_PER_EYE[channel]] = partitions
        if not complete:
            preserve.append(channel - 2)
    return coverage, preserve


def missing_expression_endpoints(coverage):
    import config as C
    missing = []
    for channel, partitions in coverage.items():
        for partition, sides in partitions.items():
            for side, counts in sides.items():
                endpoints = [name for name in ("low_endpoint", "high_endpoint") if counts[name] == 0]
                if endpoints:
                    eye = "right" if int(side) == C.TRAIN_MIRROR_BIN_SIDE else "left"
                    if C.SWAP_OUTPUT_SIDES:
                        eye = "left" if eye == "right" else "right"
                    missing.append(dict(channel=channel, partition=partition, eye=eye, missing=endpoints))
    return missing


def main():
    ap = argparse.ArgumentParser(description="Train eye tracking and export a candidate for desktop verification.")
    ap.add_argument("bin", nargs="?")
    ap.add_argument("onnx", nargs="?")
    ap.add_argument("--capabilities", action="store_true")
    ap.add_argument("--progress-json", action="store_true")
    ap.add_argument("--run-id")
    ap.add_argument("--allow-legacy", action="store_true", help="Explicitly accept non-independent legacy assessment")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--gaze-steps", type=int, default=3200)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    if args.capabilities:
        print(json.dumps(CAPABILITIES))
        return
    if not args.bin or not args.onnx:
        ap.error("capture .bin and output .onnx are required")
    if args.steps < 2 or args.gaze_steps < 1 or args.batch_size < 1 or (args.workers is not None and args.workers < 0):
        ap.error("steps must be >=2, gaze steps/batch >=1, workers >=0")
    run_id = args.run_id or str(uuid.uuid4())
    Progress.configure(run_id if args.progress_json else None)
    try:
        Progress.emit("preparing")
        train(args, run_id)
    except Exception as exc:
        Progress.fail(str(exc))
        raise


def train(args, run_id):
    import config as C
    import device as Dev
    import numpy as np
    import torch
    import bin_temporal_dataset as TD
    from capture_manifest import sha256, read_manifest
    from train_dual import fit_expr_stages
    from train_gaze import fit_gaze
    from export_dual_onnx import export, load_merged_tower
    from model_dual import MergedDualEye, build_expr_net, warm_start_from, load_presence
    import ramp_alignment as RA
    from training_artifacts import write_artifacts, write_json
    from evaluation import evaluate_merged, score

    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    import random
    random.seed(C.SEED)
    bin_path, output = Path(args.bin).resolve(), Path(args.onnx).resolve()
    if not bin_path.is_file():
        raise FileNotFoundError(bin_path)
    if output.exists() or (output.parent / "training-artifact.json").exists():
        raise ValueError("Use a fresh output directory; existing model/artifact will not be overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = read_manifest(bin_path)
    if capture is None and not args.allow_legacy:
        raise ValueError("Recording has no provenance sidecar. Recapture or explicitly choose --allow-legacy limited assessment.")
    split = "block" if capture else "frame"
    datasets = []
    with tempfile.TemporaryDirectory(prefix="babble-train-") as tmp:
        index = Path(tmp) / "index.npz"
        TD.build_index([bin_path], index)
        def dataset(part, **kw):
            ds = TD.BinTemporalDataset(part, index_path=index, split_mode=split, val_frac=.1, **kw)
            datasets.append(ds)
            return ds
        train_expr = dataset("train", aug_mode="expr", mixup=True, exclude_expr_unlabeled=True)
        train_gaze = dataset("train", aug_mode="gaze")
        val_expr = dataset("val", augment=False, exclude_expr_unlabeled=True)
        val_gaze = dataset("val", augment=False)
        train_paired = TD.BinPairedEyeDataset("train", index_path=index, split_mode=split,
                                             exclude_expr_unlabeled=True)
        val_paired = TD.BinPairedEyeDataset("val", index_path=index, split_mode=split, augment=False)
        datasets += [train_paired.base, val_paired.base]
        if not len(train_expr) or not len(val_expr) or not len(val_paired):
            raise ValueError("Insufficient usable expression or paired validation recordings")
        ancestry = TD.assert_independent(train_gaze, val_gaze) if capture else None
        # Labels derived from the pretrained model, before endpoint coverage is judged: open lids in
        # prompted-open blocks (formerly confirmed by hand) and the user's reaction lag on ramp blocks.
        auto_labels = {}
        if (app_dir() / "model_best.pt").is_file():
            device = Dev.get_device()
            probe_net = Dev.model_to(build_expr_net(pretrained=False), device).eval()
            warm_start_from(probe_net, str(app_dir() / "model_best.pt"))
            predict = lambda x: probe_net(torch.from_numpy(x).to(device)).cpu().numpy()
            probe = dataset("train", augment=False, exclude_expr_unlabeled=True)
            with torch.no_grad():
                auto_labels["open_lid_windows"] = RA.confirm_open(datasets, [probe, val_expr], predict)
                if len(RA.ramp_rows(train_expr)):
                    auto_labels["ramp"] = RA.align(datasets, probe, predict)
            del probe_net
            message = f"Confirmed {auto_labels['open_lid_windows']} open-eye windows"
            if "ramp" in auto_labels:
                message += f"; ramp labels shifted by {auto_labels['ramp']['lag_ms']} ms reaction lag"
            Progress.emit("preparing", message=message + ".")
        coverage, preserve_channels = expression_coverage(train_expr, val_expr, val_paired)
        missing_endpoints = missing_expression_endpoints(coverage)
        if not train_gaze.label_mask[:, :2].any() or not val_gaze.label_mask[:, :2].any():
            raise ValueError("No usable gaze labels remain in training or validation; repeat the gaze recording")
        if len(preserve_channels) == 4:
            raise ValueError("No expression channel has both endpoint labels; repeat the calibration")
        for ds in datasets:
            ds.label_mask[:, [2 + c for c in preserve_channels]] = False
            if ds.mixup:
                active = (ds.label[:, 2:] * ds.label_mask[:, 2:]).max(axis=1)
                usable = ds.label_mask[:, 2:].any(axis=1)
                ds.neutral_idx = np.flatnonzero((active < C.EXPR_MIXUP_NEUTRAL_THRESH) & usable)
                ds.expr_idx = np.flatnonzero((active > C.EXPR_MIXUP_EXPR_THRESH) & usable)
                ds.mixup = len(ds.neutral_idx) > 0 and len(ds.expr_idx) > 0
        C.EXPR_FUSE_CHANNELS = [c for c in C.EXPR_FUSE_CHANNELS if c not in preserve_channels]
        consensus_supported = not any(c in preserve_channels for group in C.CONSENSUS_GROUPS for c in group)
        # Mixup and paired samplers must see the same final supervision masks.
        if preserve_channels:
            warning = ("The expression encoder is frozen for ALL expression channels because verified low/high "
                       "endpoints are missing for " + ", ".join(C.DUAL_PER_EYE[c + 2] for c in preserve_channels) +
                       ". Supported classifier heads can still adapt; unsupported outputs remain at their pretrained baseline. "
                       "For lids, keep your eyes relaxed and open during the open-eye step and fully closed during the blink step.")
            print(warning, flush=True)
            Progress.emit("preparing", message=warning)
        unlabeled = TD.PartitionedUnlabeledDataset(train_gaze)
        if not len(unlabeled):
            unlabeled = None
        expr_init, gaze_init = app_dir() / "model_best.pt", app_dir() / "gaze_model_best.pt"
        if not expr_init.is_file() or not gaze_init.is_file():
            raise ValueError("Matched expression and gaze pretrained checkpoints are required beside the trainer")
        sa = max(1, args.steps // 2)
        try:
            Progress.emit("expression", 0, args.steps)
            expr_ckpt, expr_report = fit_expr_stages(train_expr, val_expr,
                batch=args.batch_size, workers=args.workers, val_every=1000,
                pretrained=False, init=str(expr_init), unlabeled_ds=unlabeled,
                paired_ds=train_paired, validation_paired=val_paired,
                preserve_channels=preserve_channels,
                deconfuse=True, even=True, consensus=C.DUAL_CONSENSUS_HEAD and consensus_supported,
                lambda_cons=C.EXPR_TUNE_LAMBDA_CONS, stage_a_steps=sa, stage_b_steps=args.steps - sa,
                ckpt_dir=Path(tmp) / "expr", log="frontend", progress_total=args.steps,
                emit_completion=False)
            Progress.emit("gaze", 0, args.gaze_steps)
            gaze_ckpt, gaze_report = fit_gaze(train_gaze, val_gaze, steps=args.gaze_steps,
                batch=args.batch_size, workers=args.workers, init=str(gaze_init),
                ckpt_dir=Path(tmp) / "gaze", log="frontend", emit_completion=False)
            Progress.emit("exporting")
            # Optional, user-independent headset-worn head; it is never fine-tuned per user.
            presence_path = app_dir() / "presence_model.pt"
            presence = load_presence(presence_path) if presence_path.is_file() else None
            export(str(gaze_ckpt), str(expr_ckpt), str(output), check=False, presence=presence)
            tower, _, consensus = load_merged_tower(gaze_ckpt, expr_ckpt)
            merged = MergedDualEye(tower, consensus, presence).eval()
            report = dict(schema_version=1, run_id=run_id, capture_sha256=sha256(bin_path),
                assessment="independent_selection_blocks" if capture else "limited_non_independent_legacy",
                fresh_headset_check="not_performed", ancestry=ancestry, expression=expr_report, gaze=gaze_report,
                deployed=evaluate_merged(merged, val_paired),
                checkpoints={"expression": sha256(expr_ckpt), "gaze": sha256(gaze_ckpt)},
                pretrained={"expression": sha256(expr_init), "gaze": sha256(gaze_init),
                            "presence": sha256(presence_path) if presence is not None else None},
                seed=C.SEED, event_metrics="unavailable_without_reviewed_events",
                training_backend={"privateuseone": "directml"}.get(Dev.get_device().type, Dev.get_device().type))
            report["expression_label_coverage"] = coverage
            report["auto_labels"] = auto_labels
            report["missing_expression_endpoints"] = missing_endpoints
            report["prompt_wink_lid_masks_removed"] = capture["prompt_wink_lid_masks_removed"] if capture else 0
            report["preserved_baseline_channels"] = [C.DUAL_PER_EYE[c + 2] for c in preserve_channels]
            report["expression_adaptation"] = "head_only_fixed_encoder" if preserve_channels else "full_fine_tune"
            report["expression_adaptation_reason"] = (
                "Missing verified low/high endpoints in one or more eye/partition groups. "
                "The fixed encoder preserves unsupported baseline outputs exactly; supported classifier rows can adapt. "
                "Review valid open and closed examples in both partitions to enable full fine-tuning."
                if preserve_channels else "All expression channels have verified low/high endpoint coverage in both eyes and partitions.")
            report["expression_selection_metric"] = "equal_activity_cohort_routine_mae_v1"
            report["summary"] = {"validation_expression_mae": score(report["deployed"]),
                                  "validation_gaze_mae_deg": score(report["deployed"], "gaze") * 2 * C.GAZE_RANGE_DEG}
            comparisons = []
            for stage_name, checkpoint_dir in (("stage-a", "stageA"), ("stage-b", "stageB")):
                child = output.parent / stage_name
                child.mkdir()
                checkpoint = Path(tmp) / "expr" / checkpoint_dir / "model_best.pt"
                export(str(gaze_ckpt), str(checkpoint), str(child / "candidate.onnx"), check=False, presence=presence)
                ctower, _, cconsensus = load_merged_tower(gaze_ckpt, checkpoint)
                cmodel = MergedDualEye(ctower, cconsensus, presence).eval()
                child_report = dict(schema_version=1, run_id=run_id, candidate_stage=stage_name,
                    assessment=report["assessment"], deployed=evaluate_merged(cmodel, val_paired),
                    fresh_headset_check="not_performed")
                write_artifacts(child / "candidate.onnx", cmodel, val_paired, run_id, child_report)
                comparisons.append(dict(path=f"{stage_name}/training-artifact.json",
                                         sha256=sha256(child / "training-artifact.json")))
            manifest = write_artifacts(output, merged, val_paired, run_id, report)
            manifest["comparison_candidates"] = comparisons
            write_json(output.parent / "training-artifact.json", manifest)
            if os.environ.get("KEEP_CKPT"):
                shutil.copyfile(expr_ckpt, output.with_suffix(".expr.pt"))
                shutil.copyfile(gaze_ckpt, output.with_suffix(".gaze.pt"))
            Progress.emit("checking", metrics=report["summary"], type="artifact_ready", manifest="training-artifact.json")
            print("Training and export completed; desktop model verification is required.", flush=True)
        finally:
            for ds in datasets:
                ds.close()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
