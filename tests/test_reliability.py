import contextlib
import copy
import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BACKBONE", "mobilenetv4_conv_small_050.e3000_r224_in1k")
import cv2
import numpy as np
import torch
import bin_reader as BR
import bin_temporal_dataset as TD
import capture_manifest as CM
import config as C
import evaluation as E
import training_progress as P


def make_capture(folder, frames=12):
    """Synthetic images/labels exercise contracts, never count as headset evidence."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "synthetic.bin"
    blocks, segments, timestamps, poses = [], [], [], []
    count = 0
    with path.open("wb") as stream:
        for repeat, role in enumerate(("train", "validation")):
            for routine in ("gaze", "open", "blink", "squint", "gazeexpr", "widen", "brow"):
                start = count
                for j in range(frames):
                    flags = BR.FLAG_GOOD_DATA
                    if routine in ("gaze", "squint", "gazeexpr"):
                        flags |= BR.FLAG_GAZE_DATA
                    if routine == "gazeexpr":
                        flags |= BR.FLAG_EXPR_UNLABELED
                    values = {key: 0 for key in BR.FIELDS}
                    values.update(routine_state=flags, left_lid=0 if routine == "blink" else 1,
                                  right_lid=0 if routine == "blink" else 1, squint=1 if routine == "squint" else 0,
                                  widen=1 if routine == "widen" else 0, brow_angry=1 if routine == "brow" else 0)
                    image = np.full((128, 128), 80 + (count % 80), np.uint8)
                    cv2.circle(image, (50 + j % 10, 64), 15, 20, -1)
                    jpeg = cv2.imencode(".jpg", image)[1].tobytes()
                    values.update(jpeg_len_left=len(jpeg), jpeg_len_right=len(jpeg), timestamp=count * 16,
                                  video_ts_left=count * 16, video_ts_right=count * 16)
                    stream.write(struct.pack(BR.FRAME_FMT, *(values[k] for k in BR.FIELDS)))
                    stream.write(jpeg); stream.write(jpeg)
                    timestamps.append(count * 16.0)
                    poses.append(count * 16.0 if flags & BR.FLAG_GAZE_DATA else None)
                    count += 1
                mask = [True] * 12
                if routine in ("gaze", "squint"):
                    mask[2] = mask[8] = False
                if routine == "gazeexpr":
                    mask = [i % 6 < 2 for i in range(12)]
                if routine in ("open", "blink"):
                    for i in (0, 1, 6, 7):
                        mask[i] = False
                blocks.append(dict(block_id=f"{routine}-{repeat}", routine_id=routine, repetition_id=repeat,
                    start_record=start, end_record_exclusive=count, role=role,
                    intended_pose=routine, label_validity=mask, annotation_provenance="synthetic"))
                segments.append(dict(segment_id=f"{routine}-{repeat}", start_record=start, end_record_exclusive=count))
    manifest = dict(schema_version=1, session_id="synthetic", run_id="synthetic",
        binary=dict(sha256=CM.sha256(path), byte_length=path.stat().st_size, record_count=count),
        blocks=blocks, segments=segments, timing=dict(image_receive_ms=timestamps, pose_receive_ms=poses,
        clock_domain="synthetic", source="synthetic_fixture"))
    Path(str(path) + ".manifest.json").write_text(json.dumps(manifest))
    return path, manifest


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.path, self.manifest = make_capture(self.folder)
        self.datasets = []

    def tearDown(self):
        for ds in self.datasets:
            ds.close()
        self.tmp.cleanup()

    def dataset(self, split, **kwargs):
        index = self.folder / "index.npz"
        if not index.exists():
            TD.build_index([self.path], index)
        ds = TD.BinTemporalDataset(split, index, split_mode="block", augment=False, **kwargs)
        self.datasets.append(ds)
        return ds

    def test_blocks_no_overlap_with_neighbor_context(self):
        train, val = self.dataset("train"), self.dataset("val")
        self.assertEqual(TD.assert_independent(train, val)["overlap"], 0)
        for ds in (train, val):
            for i in range(len(ds)):
                for _, pos, _ in ds.ancestry(i, neighbor=True):
                    self.assertEqual(ds.rec_segment[pos], ds.rec_segment[int(ds.pos[i])])

    def test_temporal_context_resets_at_block_boundary(self):
        ds = self.dataset("train")
        boundary = next(i for i in range(len(ds)) if ds.pos[i] == 12)
        x = ds[boundary][0]
        self.assertGreater(float(x[0].sum()), 0)
        self.assertEqual(float(x[1:].sum()), 0)

    def test_unlabeled_stream_stays_in_training(self):
        train = self.dataset("train")
        stream = TD.PartitionedUnlabeledDataset(train)
        self.assertEqual(len(stream), 24)
        self.assertEqual(float(stream[0][3]), 0)
        self.assertEqual(float(stream[2][3]), 1)

    def test_mask_survives_mixup_and_closed_gaze_has_zero_sampling(self):
        ds = self.dataset("train")
        ds.augment = True; ds.mixup = True
        ds.neutral_idx = np.array([0]); ds.expr_idx = np.array([72])
        with patch("random.random", return_value=.1):
            _, _, _, mask = ds[0]
        self.assertFalse(mask[0]); self.assertFalse(mask[2])
        closed = ds.label[:, 2] > .5
        self.assertTrue((ds.gaze_sampler_weights()[closed] == 0).all())

    def test_manifest_rejects_hash_overlap_orphan_and_bad_timing(self):
        cases = []
        item = copy.deepcopy(self.manifest); item["binary"]["sha256"] = "0" * 64; cases.append(item)
        item = copy.deepcopy(self.manifest); item["blocks"][1]["start_record"] = 0; cases.append(item)
        item = copy.deepcopy(self.manifest); item["segments"].pop(); cases.append(item)
        item = copy.deepcopy(self.manifest); item["timing"]["image_receive_ms"] = [0]; cases.append(item)
        for item in cases:
            Path(str(self.path) + ".manifest.json").write_text(json.dumps(item))
            with self.assertRaises(ValueError): CM.read_manifest(self.path)

    def test_missing_routine_validation_fails(self):
        next(b for b in self.manifest["blocks"] if b["role"] == "validation" and b["routine_id"] == "gazeexpr")["role"] = "train"
        Path(str(self.path) + ".manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "gazeexpr"):
            self.dataset("train")

    def test_routine_macro_and_mask_counts(self):
        pred = np.array([[.1], [.1], [.9], [.6]])
        target = np.zeros_like(pred)
        report = E.summarize(pred, target, [[True], [True], [True], [False]],
                             ["a", "a", "b", "b"], ["rightEyeSquint"])
        self.assertAlmostEqual(report["channels"]["rightEyeSquint"]["mae"], .5)
        self.assertEqual(report["channels"]["rightEyeSquint"]["masked"], 1)

    def test_candidate_tie_regression_and_improvement(self):
        def report(a, b):
            return {"channels": {"rightEyeSquint": {"mae": a, "count": 1, "masked": 0},
                                 "leftEyeSquint": {"mae": b, "count": 1, "masked": 0}}}
        old = report(.3, .1)
        self.assertFalse(E.choose_candidate(old, old)[0])
        self.assertFalse(E.choose_candidate(old, report(.1, .13))[0])
        self.assertTrue(E.choose_candidate(old, report(.2, .11))[0])

    def test_gaze_baseline_requires_improvement_and_axis_guard(self):
        def report(a, b):
            return {"channels": {"EyeY": {"mae": a, "count": 8, "masked": 0},
                                 "EyeX": {"mae": b, "count": 8, "masked": 0}}}
        old = report(.1, .1)
        self.assertFalse(E.choose_candidate(old, report(.101, .1), "gaze")[0])
        self.assertFalse(E.choose_candidate(old, report(.05, .11), "gaze")[0])
        self.assertTrue(E.choose_candidate(old, report(.08, .1), "gaze")[0])

    def test_semantic_masks_map_to_correct_physical_eye(self):
        for b in self.manifest["blocks"]:
            if b["routine_id"] == "open":
                b["label_validity"][2] = False
        Path(str(self.path) + ".manifest.json").write_text(json.dumps(self.manifest))
        ds = self.dataset("train")
        for i in np.flatnonzero(ds.routine == "open"):
            self.assertEqual(bool(ds.label_mask[i, 2]), ds.side[i] != C.TRAIN_MIRROR_BIN_SIDE)

    def test_metrics_are_safe_weights_only_checkpoint_metadata(self):
        report = E.summarize([[.1]], [[0]], [[True]], np.array(["hold"]), ["rightEyeSquint"])
        checkpoint = self.folder / "checkpoint.pt"
        torch.save({"validation": report}, checkpoint)
        self.assertEqual(torch.load(checkpoint, weights_only=True)["validation"], report)

    def test_truncated_binary_fails_instead_of_using_partial_capture(self):
        Path(str(self.path) + ".manifest.json").unlink()
        self.path.write_bytes(self.path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "Truncated"):
            TD.build_index([self.path], self.folder / "index.npz")

    def test_reviewed_hold_standard_deviation_and_coverage(self):
        result = E.hold_statistics([[.1, .2], [.3, .4], [float("nan"), .6]], [0, 20, 40],
                                   [True, True], ["rightEyeSquint", "leftEyeX"], 4, 60)
        self.assertAlmostEqual(result["channels"]["rightEyeSquint"]["standard_deviation"], .1)
        self.assertEqual(result["channels"]["rightEyeSquint"]["missing"], 1)
        self.assertEqual(result["channels"]["rightEyeSquint"]["invalid"], 1)
        self.assertAlmostEqual(result["channels"]["leftEyeX"]["standard_deviation"], np.std([.2, .4, .6]) * 90)

    def test_invalid_label_cannot_suppress_another_valid_loss_channel(self):
        from train_dual import deconfuse_weight
        target = torch.tensor([[0., 0., 1., 0.]])
        valid = torch.tensor([[1., 1., 0., 1.]])
        self.assertEqual(float(deconfuse_weight(target, valid)[0, 0]), 1.)

    def test_reviewed_values_round_trip_semantic_eye_order(self):
        values = [.15, .2, .4, .5, .6, .7, .25, .8, .3, .4, .5, .6]
        self.manifest["label_overrides"] = [dict(start_record=0, end_record_exclusive=1,
            label_validity=[True] * 12, values=values, annotation_provenance="reviewed")]
        Path(str(self.path) + ".manifest.json").write_text(json.dumps(self.manifest))
        self.dataset("train")
        paired = TD.BinPairedEyeDataset("train", self.folder / "index.npz", split_mode="block", augment=False)
        self.datasets.append(paired.base)
        _, labels, masks = E.paired_inputs(paired, [0])
        np.testing.assert_allclose(labels.numpy()[0], values, atol=1e-6)
        self.assertTrue(masks.all())

    def test_head_only_preserves_lids_and_encoder_in_both_stages(self):
        from model_dual import build_expr_net
        from train_dual import fit_expr_stages
        torch.set_num_threads(2)
        os.environ["DEVICE"] = "cpu"
        original = build_expr_net(pretrained=False).eval()
        checkpoint = self.folder / "initial.pt"
        torch.save({"teacher": original.state_dict(), "backbone": C.BACKBONE}, checkpoint)
        train, val = self.dataset("train"), self.dataset("val")
        paired = TD.BinPairedEyeDataset("val", self.folder / "index.npz", split_mode="block", augment=False)
        self.datasets.append(paired.base)
        for ds in (train, val, paired.base):
            ds.label_mask[:, 2] = False
        inputs = torch.rand(4, 4, 128, 128)
        inputs[0] = 0; inputs[1] = 1; inputs[2, :, :64] = 0
        with torch.no_grad():
            before = original(inputs)[:, 0]
        fit_expr_stages(train, val, batch=2, workers=0, val_every=1,
                        pretrained=False, init=str(checkpoint), ckpt_dir=self.folder / "fit",
                        stage_a_steps=1, stage_b_steps=1, even=False, consensus=False,
                        validation_paired=paired, preserve_channels=[0])
        for stage in ("stageA", "stageB"):
            state = torch.load(self.folder / "fit" / stage / "model_best.pt", weights_only=True)["teacher"]
            after = build_expr_net(pretrained=False).eval()
            after.load_state_dict(state)
            for name, value in original.state_dict().items():
                if not name.startswith("classifier."):
                    self.assertTrue(torch.equal(value, state[name]), name)
            with torch.no_grad():
                torch.testing.assert_close(before, after(inputs)[:, 0], rtol=0, atol=0)

    def test_missing_validation_endpoint_preserves_channel(self):
        from train_frontend import expression_coverage
        train, val = self.dataset("train"), self.dataset("val")
        _, baseline = expression_coverage(train, val)
        self.assertNotIn(2, baseline)
        val.label_mask[(val.side == 0) & (val.label[:, 4] >= .8), 4] = False
        coverage, preserve = expression_coverage(train, val)
        self.assertIn(2, preserve)
        self.assertGreater(coverage["EyeSquint"]["train"]["0"]["high_endpoint"], 0)
        self.assertEqual(coverage["EyeSquint"]["validation"]["0"]["high_endpoint"], 0)

    def test_progress_finite_versioned_and_exponent(self):
        out = io.StringIO()
        P.configure("test")
        with contextlib.redirect_stdout(out):
            P.emit("gaze", 1, 2, {"train_loss": 1e-8})
        data = json.loads(out.getvalue())
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["metrics"]["train_loss"], 1e-8)
        with self.assertRaises(ValueError): P.emit("gaze", metrics={"train_loss": float("nan")})
        P.configure()

    def test_backend_selection_and_explicit_cpu(self):
        import device
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {"torch_directml": None}):
            self.assertEqual(device.get_device().type, "cpu")
            for requested in ("auto", "cpu", ""):
                os.environ["DEVICE"] = requested
                self.assertEqual(device.get_device().type, "cpu")
            os.environ["DEVICE"] = "directml"
            with self.assertRaisesRegex(RuntimeError, "Explicit DirectML"):
                device.get_device()
            os.environ["DEVICE"] = "unsupported"
            with self.assertRaises(ValueError):
                device.get_device()
        from types import SimpleNamespace
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules,
                {"torch_directml": SimpleNamespace(device=lambda: torch.device("privateuseone"))}):
            self.assertEqual(device.get_device().type, "privateuseone")
            os.environ["DEVICE"] = "cpu"
            self.assertEqual(device.get_device().type, "cpu")


if __name__ == "__main__":
    unittest.main()
