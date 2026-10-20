import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BACKBONE", "mobilenetv4_conv_small_050.e3000_r224_in1k")
import evaluation as E
import train_dual as T


def report(active_error, inactive_error, other_error=.1, names=("rightEyeSquint", "leftEyeSquint")):
    target = np.array([[1., 1.], [0., 0.], [0., 0.], [0., 0.]])
    pred = np.array([[1 - active_error, 1 - other_error],
                     [inactive_error, other_error], [inactive_error, other_error],
                     [inactive_error, other_error]])
    return E.summarize(pred, target, np.ones_like(target, bool),
                       ["squint", "open", "gaze", "brow"], list(names))


class ExpressionSelectionTests(unittest.TestCase):
    def test_unpaired_endpoints_do_not_enable_shared_encoder_fine_tuning(self):
        from types import SimpleNamespace
        from train_frontend import expression_coverage, missing_expression_endpoints
        # Each eye has valid open/closed labels on different records; neither
        # contributes lid labels to the only record with two usable images.
        labels = np.zeros((6, 6), np.float32)
        labels[[1, 3], 2:] = 1
        masks = np.ones_like(labels, bool)
        masks[4:, 2] = False
        dataset = SimpleNamespace(label=labels, label_mask=masks, side=np.array([0, 0, 1, 1, 0, 1]))
        paired = SimpleNamespace(base=dataset, pairs=[(0, 2, 5, 4)])
        _, old_preserved = expression_coverage(dataset, dataset)
        self.assertNotIn(0, old_preserved)
        coverage, preserved = expression_coverage(dataset, dataset, paired)
        self.assertIn(0, preserved)
        self.assertEqual(coverage["EyeLid"]["validation_paired"]["0"]["valid"], 0)
        self.assertEqual(coverage["EyeLid"]["validation_paired"]["1"]["valid"], 0)
        missing = missing_expression_endpoints(coverage)
        lid_missing = [m for m in missing if m["channel"] == "EyeLid"]
        self.assertEqual({m["eye"] for m in lid_missing}, {"left", "right"})
        self.assertTrue(all(m["partition"] == "validation_paired" for m in lid_missing))

    def test_active_error_is_not_diluted_by_inactive_routines(self):
        old = report(.1, .3, .3)
        new = report(.18, .01, .1)
        self.assertLess(new["channels"]["rightEyeSquint"]["mae"], old["channels"]["rightEyeSquint"]["mae"])
        self.assertLess(E.score(new), E.score(old))
        accepted, reason = E.choose_candidate(old, new)
        self.assertFalse(accepted)
        self.assertIn("rightEyeSquint/active", reason)
        self.assertAlmostEqual(old["channels"]["rightEyeSquint"]["balanced_mae"], .2)

    def test_inactive_false_activations_are_guarded_too(self):
        self.assertFalse(E.choose_candidate(report(.6, .01), report(.1, .04))[0])
        self.assertTrue(E.choose_candidate(report(.6, .1), report(.2, .11))[0])

    def test_open_lid_cohort_tolerates_small_drift_but_not_large(self):
        lids = ("rightEyeLid", "leftEyeLid")
        # Open-lid labels come from frames the baseline already reads as open, so it starts near zero.
        self.assertTrue(E.choose_candidate(report(.22, .035, names=lids), report(.14, .057, names=lids))[0])
        self.assertFalse(E.choose_candidate(report(.22, .035, names=lids), report(.14, .10, names=lids))[0])

    def test_masked_channels_and_intermediate_cohorts(self):
        out = E.summarize([[.9, .6], [.9, .7]], [[0, .5], [1, .5]],
                          [[False, True], [False, True]], ["a", "b"], ["rightEyeLid", "leftEyeSquint"])
        self.assertIsNone(out["channels"]["rightEyeLid"]["balanced_mae"])
        self.assertEqual(out["cohorts"]["leftEyeSquint"]["intermediate"]["count"], 2)
        self.assertAlmostEqual(E.score(out), .15)

    def test_cohort_coverage_cannot_be_swapped_with_same_total_count(self):
        old, new = report(.4, .1), report(.3, .1)
        new["cohorts"]["rightEyeSquint"]["active"]["routines"] = {"open": {"count": 1, "mae": .3}}
        self.assertIn("coverage", E.choose_candidate(old, new)[1])
        new = copy.deepcopy(old)
        new["evaluation_sampling"] = {"identity_sha256": "different"}
        self.assertIn("sample identities", E.choose_candidate(old, new)[1])

    def test_sample_spans_every_repetition_and_both_eyes(self):
        routines, blocks, positions = [], [], []
        for repeat in (0, 1):
            for side in (0, 1):
                for pos in range(100):
                    routines.append("gaze")
                    blocks.append((0, repeat, side))
                    positions.append(pos)
        indices = E.balanced_indices(routines, 20, blocks=blocks, positions=positions)
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(len(indices), 20)
        for key in set(blocks):
            sampled = [positions[i] for i in indices if blocks[i] == key]
            self.assertEqual(sampled, [0, 25, 50, 74, 99])
        self.assertEqual(indices, E.balanced_indices(routines, 20, blocks=blocks, positions=positions))

    def test_short_blocks_redistribute_budget_and_single_sample_is_middle(self):
        indices = E.balanced_indices(["short"] * 2 + ["long"] * 100, 12)
        self.assertEqual(indices[:2], [2, 13])  # alphabetical routine order: long first
        self.assertIn(101, indices)
        self.assertIn(0, indices)
        self.assertIn(1, indices)
        self.assertEqual(E.balanced_indices(["a"] * 100, 1), [50])

    def test_sampling_report_names_omitted_blocks(self):
        identities = [("a", 0, block, pos) for block in (0, 1) for pos in range(5)]
        result = E.sampling_report(identities, [2], 1)
        self.assertEqual([b["selected"] for b in result["blocks"]], [1, 0])
        self.assertEqual(result["eligible"], 10)

    def test_short_reviewed_endpoint_is_retained_and_missing_endpoint_fails_closed(self):
        from types import SimpleNamespace
        n = 5020
        routines = np.array(["open"] * 5000 + ["blink"] * 20)
        block = np.array([0] * 5000 + [1] * 20)
        times = np.arange(n, dtype=float)
        labels = np.zeros((n * 2, 6), np.float32)
        labels[10000:, 2] = 1
        masks = np.zeros_like(labels, bool)
        masks[[4, 5], 2] = True  # only record 2 was explicitly confirmed open
        masks[10000:, 2] = True
        base = SimpleNamespace(routine=np.repeat(routines, 2), block=np.repeat(block, 2),
            timestamp=np.repeat(times, 2), label=labels, label_mask=masks)
        paired = SimpleNamespace(base=base, pairs=[(0, i, 2 * i + 1, 2 * i) for i in range(n)])
        old = E.paired_sample_indices(paired, 4000)
        self.assertNotIn(2, old)
        fixed = E.paired_sample_indices(paired, 4000, ensure_expression_cohorts=True)
        self.assertIn(2, fixed)
        self.assertEqual(len(set(fixed)), 4000)
        self.assertTrue({0, 4999, 5000, 5019} <= set(fixed))
        truth = np.zeros((n, 1)); truth[5000:] = 1
        valid = np.zeros((n, 1), bool); valid[2] = True; valid[5000:] = True
        before = np.zeros((n, 1)); before[5000:] = .6
        after = np.full((n, 1), .95)
        for samples in (old, fixed):
            baseline = E.summarize(before[samples], truth[samples], valid[samples], routines[samples], ["rightEyeLid"])
            candidate = E.summarize(after[samples], truth[samples], valid[samples], routines[samples], ["rightEyeLid"])
            accepted, reason = E.choose_candidate(baseline, candidate)
            self.assertFalse(accepted)
            self.assertIn("Incomplete validation endpoints" if samples is old else "inactive", reason)
        with self.assertRaisesRegex(ValueError, "budget"):
            E.paired_sample_indices(paired, 1, ensure_expression_cohorts=True)

    def test_irregular_timestamps_use_time_not_dense_prefix_record_counts(self):
        times = list(range(90)) + [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        selected = E.balanced_indices(["gaze"] * len(times), 5, positions=times)
        self.assertEqual([times[i] for i in selected], [0, 300, 500, 800, 1000])
        # Large gaps still yield unique indices and exactly the requested budget.
        times = [0, 1, 2, 1000, 1001, 1002]
        selected = E.balanced_indices(["gaze"] * 6, 5, positions=times)
        self.assertEqual(len(set(selected)), 5)
        self.assertEqual((selected[0], selected[-1]), (0, 5))

    def test_reviewed_holds_do_not_change_selection_population_or_temporal_ancestry(self):
        from test_reliability import make_capture
        import bin_temporal_dataset as TD
        class ConstantMerged(nn.Module):
            fuse_channels = []
            fuse_source = "none"
            consensus = None
            def forward(self, x):
                return x.new_full((len(x), 12), .5)
        with tempfile.TemporaryDirectory() as folder:
            path, _ = make_capture(folder)
            index = Path(folder) / "index.npz"
            TD.build_index([path], index)
            paired = TD.BinPairedEyeDataset("val", index, split_mode="block", augment=False)
            train = TD.BinTemporalDataset("train", index, split_mode="block", augment=False)
            try:
                sampled = E.paired_sample_indices(paired, 14)
                training_ancestry = set().union(*(train.ancestry(i) for i in range(len(train))))
                for i in sampled:
                    for source in paired.pairs[i][2:]:
                        self.assertFalse(paired.base.ancestry(source) & training_ancestry)
                before = E.evaluate_merged(ConstantMerged(), paired, limit=14)
                paired.base.reviewed_holds = [dict(bin_id=0, start_record=84, end_record_exclusive=96,
                                                   channel_mask=[True] * 12)]
                after = E.evaluate_merged(ConstantMerged(), paired, limit=14)
                self.assertEqual(before["channels"], after["channels"])
                self.assertEqual(before["cohorts"], after["cohorts"])
                self.assertEqual(before["evaluation_sampling"], after["evaluation_sampling"])
                self.assertEqual(after["steady_hold_jitter"]["intervals"][0]["channels"]["rightEyeLid"]["count"], 12)
            finally:
                paired.base.close()
                train.close()

    def test_later_unsafe_checkpoint_cannot_replace_earlier_safe_checkpoint(self):
        from test_reliability import make_capture
        import bin_temporal_dataset as TD
        torch.set_num_threads(2)
        old, safe, unsafe = report(.1, .3, .3), report(.11, .2, .2), report(.2, .005, .005)
        self.assertLess(E.score(unsafe), E.score(safe))
        with tempfile.TemporaryDirectory() as folder:
            path, _ = make_capture(folder)
            index = Path(folder) / "index.npz"
            TD.build_index([path], index)
            train = TD.BinTemporalDataset("train", index, split_mode="block", augment=False)
            val = TD.BinTemporalDataset("val", index, split_mode="block", augment=False)
            try:
                with patch.object(T.Dev, "get_device", return_value=torch.device("cpu")), \
                     patch.object(E, "evaluate_expression", side_effect=[safe, unsafe]):
                    checkpoint, metrics = T.fit_expr(train, val, steps=2, batch=2, workers=0,
                        lr_head=.001, lr_backbone=.001, warmup=0, val_every=1, pretrained=False,
                        ckpt_dir=Path(folder) / "fit", validation_paired=object(), baseline_report=old)
                saved = torch.load(checkpoint, weights_only=True)
                self.assertEqual(saved["step"], 0)
                self.assertEqual(metrics["channels"], safe["channels"])
                self.assertTrue(metrics["pretrained_comparison"]["eligible"])
            finally:
                train.close()
                val.close()

    def test_stages_cannot_compound_drift_relative_to_pretrained(self):
        # B is within .02 of A, but .03 worse than the untouched baseline on active squint.
        baseline = report(.1, .3, .3)
        stage_a = report(.115, .2, .2)
        stage_b = report(.13, .05, .05)
        self.assertTrue(E.choose_candidate(stage_a, stage_b)[0])
        final, meta, starts = self.run_stages(baseline, stage_a, stage_b)
        self.assertEqual(meta["selected"], "stage_a")
        self.assertEqual(final, b"stageA")
        self.assertFalse(meta["comparisons"]["stage_b_to_baseline"]["eligible"])
        self.assertEqual(starts[-1], "model_best.pt")

    def test_both_bad_stages_export_untouched_baseline_and_b_starts_from_baseline(self):
        baseline = report(.1, .3, .3)
        final, meta, starts = self.run_stages(baseline, report(.2, .01, .01), report(.21, .005, .005))
        self.assertEqual(meta["selected"], "baseline")
        self.assertEqual(meta["baseline"], baseline)
        self.assertEqual(meta["stage_b_start"], "baseline")
        self.assertEqual(starts, ["baseline.pt", "baseline.pt"])
        self.assertTrue(final.startswith(b"PK"))  # real torch checkpoint, not either mock candidate

    def test_b_can_recover_after_a_is_rejected(self):
        _, meta, _ = self.run_stages(report(.1, .3, .3), report(.2, .01, .01), report(.105, .2, .2))
        self.assertEqual(meta["selected"], "stage_b")

    def run_stages(self, baseline, stage_a, stage_b):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = nn.Linear(2, 4)
        starts = []
        def fit(*args, **kwargs):
            starts.append(Path(kwargs["init"]).name)
            folder = kwargs["ckpt_dir"]
            folder.mkdir()
            path = folder / "model_best.pt"
            path.write_bytes(folder.name.encode())
            self.assertEqual(kwargs["baseline_report"], baseline)
            return path, stage_a if folder.name == "stageA" else stage_b
        with tempfile.TemporaryDirectory() as folder, patch.object(T, "build_expr_net", return_value=Tiny()), \
                patch.object(T.Dev, "get_device", return_value=torch.device("cpu")), \
                patch.object(E, "evaluate_expression", return_value=baseline), patch.object(T, "fit_expr", side_effect=fit):
            path, meta = T.fit_expr_stages(None, None, val_every=1, pretrained=False,
                ckpt_dir=Path(folder), validation_paired=object(), stage_a_steps=1, stage_b_steps=1)
            return path.read_bytes(), meta, starts


if __name__ == "__main__":
    unittest.main()
