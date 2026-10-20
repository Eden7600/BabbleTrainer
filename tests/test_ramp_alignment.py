import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ramp_alignment as RA


def schedule(seconds):
    # Same shape as Baballonia's RampSchedule.
    t = np.mod(seconds, 4.0)
    return np.select([t < .5, t < 2, t < 2.5], [0, (t - .5) / 1.5, 1], (4 - t) / 1.5)


class RampAlignmentTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(0, 12000, 1000 / 60)  # one 12 s block at 60 fps
        n = len(t)
        routines = ["squintramp", "lidramp", "squint"]  # the hold block must stay untouched
        self.ds = SimpleNamespace(
            timestamp=np.tile(t, 3), bin_id=np.zeros(3 * n, int), pos=np.arange(3 * n), block=np.repeat([0, 1, 2], n),
            side=np.zeros(3 * n, int), routine=np.repeat(routines, n), label=np.zeros((3 * n, 6), np.float32))
        self.ds.label[:n, 4] = schedule(t / 1000)
        self.ds.label[n:2 * n, 2] = schedule(t / 1000)
        self.ds.label[2 * n:, 4] = 1
        self.n, self.t = n, t

    def test_recovers_lag_and_shifts_only_ramp_labels(self):
        rows = RA.ramp_rows(self.ds)
        self.assertEqual(2 * self.n, len(rows))
        rng = np.random.default_rng(0)
        late = schedule((self.t - 300) / 1000)
        preds = np.zeros((len(rows), 6))
        preds[:self.n, 4] = 0.7 * late + 0.1 + rng.normal(0, .05, self.n)  # scale/offset must not matter
        preds[self.n:, 2] = late + rng.normal(0, .05, self.n)
        lag, corr = RA.estimate_lag(self.ds, rows, preds)
        self.assertEqual(300, lag)
        self.assertGreater(corr, .95)
        aligned = RA.shifted(self.ds, rows, lag)
        np.testing.assert_allclose(aligned[self.n + 60:, 2], late[60:], atol=1e-4)

    def test_recovers_a_face_that_leads_the_cue(self):
        rows = RA.ramp_rows(self.ds)
        early = schedule((self.t + 300) / 1000)   # anticipating the rhythm
        preds = np.zeros((len(rows), 6))
        preds[:self.n, 4] = early
        preds[self.n:, 2] = early
        self.assertEqual(-300, RA.estimate_lag(self.ds, rows, preds)[0])

    def test_align_applies_in_place_and_skips_captures_without_ramps(self):
        hold = SimpleNamespace(**{k: v[2 * self.n:] for k, v in vars(self.ds).items()})
        self.assertIsNone(RA.align([hold], [hold], lambda x: None))
        late = schedule((self.t - 200) / 1000)
        lookup = {"squintramp": (4, 0), "lidramp": (2, 1)}

        class Probe(SimpleNamespace):
            # Each "image" is just its row index so predict() can look up the lagged schedule.
            def __getitem__(self, i):
                return (SimpleNamespace(numpy=lambda: np.array([i])),)
        probe = Probe(**{k: v.copy() for k, v in vars(self.ds).items()})

        def predict(x):
            out = np.zeros((len(x), 4))
            for row, (i,) in enumerate(x):
                c, block = lookup[str(probe.routine[i])]
                out[row, c - 2] = late[i - block * self.n]
            return out
        before = self.ds.label[2 * self.n:].copy()
        report = RA.align([self.ds], [probe], predict)
        self.assertEqual(200, report["lag_ms"])
        np.testing.assert_array_equal(before, self.ds.label[2 * self.n:])
        np.testing.assert_allclose(self.ds.label[60:self.n, 4], late[60:], atol=1e-4)


class ShapeTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(0, 12000, 1000 / 60)
        self.schedule = schedule(self.t / 1000)

    def test_follows_the_real_motion_but_stays_monotonic_and_anchored(self):
        rng = np.random.default_rng(1)
        eased = schedule((self.t - 150) / 1000) ** 2                # starts late, eases in
        response = .1 + .6 * eased + rng.normal(0, .03, len(self.t))  # base model scale, noisy
        response[np.argmax(self.schedule)] += .3                     # an overshoot spike
        out = RA.shape(self.schedule, response)
        self.assertIsNotNone(out)
        self.assertLess(np.abs(out - eased).mean(), .05)
        self.assertLess(out[self.schedule < .02].mean(), .05)
        self.assertGreater(out[self.schedule > .98].mean(), .9)
        rising = (self.t % 4000 > 600) & (self.t % 4000 < 1900)
        self.assertTrue((np.diff(out)[rising[1:]] >= -1e-9).all())

    def test_untrustworthy_response_keeps_the_schedule(self):
        rng = np.random.default_rng(2)
        self.assertIsNone(RA.shape(self.schedule, .5 + rng.normal(0, .01, len(self.t))))   # flat
        self.assertIsNone(RA.shape(self.schedule, rng.random(len(self.t))))               # unrelated


class ConfirmOpenTests(unittest.TestCase):
    def test_marks_only_clearly_open_eyes_in_open_blocks(self):
        n = 6
        ds = SimpleNamespace(routine=np.array(["open", "open", "open", "gaze", "blink", "squint"]),
                             bin_id=np.zeros(n, int), pos=np.arange(n), side=np.zeros(n, int),
                             label=np.full((n, 6), .5, np.float32), label_mask=np.zeros((n, 6), bool))
        closure = {0: .02, 1: .9, 2: .1, 3: .05, 4: .01, 5: .0}   # row 1 is a blink inside the open block

        class Probe(SimpleNamespace):
            def __getitem__(self, i):
                return (SimpleNamespace(numpy=lambda: np.array([i])),)
        probe = Probe(**{k: v.copy() for k, v in vars(ds).items()})
        predict = lambda x: np.array([[closure[int(i)], 0, 0, 0] for (i,) in x])
        self.assertEqual(3, RA.confirm_open([ds], [probe], predict))
        np.testing.assert_array_equal(ds.label_mask[:, 2], [True, False, True, True, False, False])
        np.testing.assert_array_equal(ds.label[[0, 2, 3], 2], 0)
        self.assertEqual(.5, ds.label[1, 2])


if __name__ == "__main__":
    unittest.main()
