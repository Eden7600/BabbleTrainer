import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eye_seg import geometry


class GeometryTests(unittest.TestCase):
    def test_open_eye_ratios(self):
        m = np.zeros((128, 128), np.uint8)
        m[40:60, 20:100] = 1      # aperture 20 px tall
        m[40:60, 50:70] = 2       # iris 20 px wide, clipped by the lids
        m[45:55, 55:65] = 3       # pupil 10 px wide
        openness, pupil, width = geometry(m)
        self.assertEqual(20, width)
        self.assertAlmostEqual(1.0, openness)
        self.assertAlmostEqual(0.5, pupil)

    def test_closed_and_fragmented(self):
        self.assertEqual(0.0, geometry(np.zeros((128, 128), np.uint8))[0])
        m = np.zeros((128, 128), np.uint8)
        m[50:54, 40:50] = 2
        m[50:52, 70:80] = 2       # two iris fragments; the gap column must not crash
        openness, _, width = geometry(m)
        self.assertEqual(40, width)
        self.assertAlmostEqual(4 / 40, openness)


if __name__ == "__main__":
    unittest.main()
