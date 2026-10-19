import copy
import unittest

from evaluate_experiment import digest, evaluate


def fixture():
    protocol = {"schema_version": 1, "status": "frozen", "target_metric": "squint_mae",
                "hardware": "fixture PC", "backend": "CPU", "dataset_sha256": "1" * 64,
                "baseline_sha256": "2" * 64, "candidate_sha256": "3" * 64,
                "license_review": "synthetic test only", "clip_ids": [str(n) for n in range(10)],
                "expression_channels": ["squint"], "gaze_axes": ["yaw", "pitch"]}
    clips = []
    for clip_id in protocol["clip_ids"]:
        baseline = {"target_error": .2, "invalid_count": 0, "expression_mae": {"squint": .2},
                    "gaze_axis_mae_deg": {"yaw": 1, "pitch": 1}, "response_lag_ms": 20,
                    "inference_ms": [2] * 100}
        candidate = copy.deepcopy(baseline)
        candidate["target_error"] = .15
        candidate["expression_mae"]["squint"] = .15
        clips.append({"clip_id": clip_id, "independent_unit_id": clip_id, "reviewed_labels": True,
                      "blind_final": True, "measured_frame_period_ms": 10, "sample_count": 100,
                      "baseline": baseline, "candidate": candidate})
    return protocol, {"schema_version": 1, "protocol_sha256": digest(protocol), "hardware": "fixture PC",
                      "backend": "CPU", "clips": clips}


class ExperimentTests(unittest.TestCase):
    def test_supported_gain_requires_review_not_automatic_activation(self):
        report = evaluate(*fixture())
        self.assertEqual(report["decision"], "eligible_for_review")
        self.assertFalse(report["automatic_promotion"])

    def test_correlated_clips_cannot_fake_independent_sample_count(self):
        protocol, results = fixture()
        for clip in results["clips"]:
            clip["independent_unit_id"] = "same-session"
        report = evaluate(protocol, results)
        self.assertEqual(report["decision"], "retain_baseline")
        self.assertEqual(report["independent_units"], 1)

    def test_better_target_does_not_override_latency_regression(self):
        protocol, results = fixture()
        results["clips"][0]["candidate"]["inference_ms"] = [3] * 100
        self.assertEqual(evaluate(protocol, results)["decision"], "retain_baseline")

    def test_masks_and_missing_clips_cannot_hide_failures(self):
        protocol, results = fixture()
        del results["clips"][0]["candidate"]["expression_mae"]["squint"]
        with self.assertRaises(ValueError): evaluate(protocol, results)
        protocol, results = fixture()
        results["clips"].pop()
        with self.assertRaises(ValueError): evaluate(protocol, results)

    def test_changed_protocol_and_nonfinite_input_are_rejected(self):
        protocol, results = fixture()
        protocol["target_metric"] = "changed after comparison"
        with self.assertRaises(ValueError): evaluate(protocol, results)
        protocol, results = fixture()
        results["clips"][0]["candidate"]["target_error"] = float("nan")
        with self.assertRaises(ValueError): evaluate(protocol, results)

    def test_unreviewed_or_tuning_data_cannot_pass_as_blind(self):
        for field in ("reviewed_labels", "blind_final"):
            protocol, results = fixture()
            results["clips"][0][field] = False
            with self.assertRaises(ValueError): evaluate(protocol, results)


if __name__ == "__main__":
    unittest.main()
