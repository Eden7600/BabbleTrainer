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
