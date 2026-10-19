import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capture_manifest as CM
from test_reliability import make_capture


class CaptureManifestTests(unittest.TestCase):
    def test_legacy_prompt_wink_masks_only_remove_unsupported_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture, manifest = make_capture(Path(tmp))
            for block in manifest["blocks"]:
                if block["routine_id"] == "open":
                    block["routine_id"] = "winkleft"
                    block["annotation_provenance"] = "prompt"
                    block["label_validity"][2], block["label_validity"][8] = True, False
            path = Path(str(capture) + ".manifest.json")
            path.write_text(json.dumps(manifest))
            original = path.read_bytes()
            result = CM.read_manifest(capture)
            self.assertEqual(result["prompt_wink_lid_masks_removed"], 24)
            self.assertFalse(result["masks"][12:24, 2].any())
            self.assertFalse(result["masks"][12:24, 8].any())
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()


class FrameCacheTests(unittest.TestCase):
    def test_cached_frames_match_decoding_from_the_capture(self):
        import numpy as np
        import bin_temporal_dataset as TD
        with tempfile.TemporaryDirectory() as tmp:
            capture, _ = make_capture(Path(tmp))
            index = Path(tmp) / "index.npz"
            TD.build_index([capture], index)
            cached = TD.BinTemporalDataset("train", index, split_mode="block", augment=False)
            direct = TD.BinTemporalDataset("train", index, split_mode="block", augment=False)
            direct._frames_path = None
            try:
                self.assertIsNotNone(cached._frames_path)
                n = int(cached.bin_rec_start[1])
                for pos in range(-1, n + 1):
                    for side in (0, 1):
                        a, b = cached._load_frame(0, pos, side), direct._load_frame(0, pos, side)
                        self.assertEqual(a is None, b is None, (pos, side))
                        if a is not None:
                            np.testing.assert_array_equal(a, b)
            finally:
                cached.close(); direct.close()
