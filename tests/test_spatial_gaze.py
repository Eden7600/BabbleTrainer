import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_dual as M

# The shipped checkpoints live with the app: Baballonia/src/Baballonia.Desktop/Calibration/Windows/Trainer.
SHIPPED = Path(__file__).resolve().parents[2] / "Baballonia/src/Baballonia.Desktop/Calibration/Windows/Trainer"


@unittest.skipUnless((SHIPPED / "gaze_model_best.pt").exists(), "shipped checkpoints not present")
class SpatialGazeTests(unittest.TestCase):
    def test_spatial_warm_start_keeps_shipped_convs(self):
        net = M.build_gaze_net(spatial=True)
        loaded = M.warm_start_gaze(net, SHIPPED / "gaze_model_best.pt")
        shipped = M.gaze_state(SHIPPED / "gaze_model_best.pt")
        self.assertEqual(len(shipped) - 1, loaded)  # only fc_gaze.weight changes shape and starts fresh
        torch.testing.assert_close(net.state_dict()["conv6.weight"], shipped["conv6.weight"])
        self.assertEqual((2, 106 * 16), tuple(net.fc_gaze.weight.shape))

    def test_gaze_net_for_matches_either_head(self):
        self.assertFalse(M.gaze_net_for(M.gaze_state(SHIPPED / "gaze_model_best.pt")).spatial)
        self.assertTrue(M.gaze_net_for(M.build_gaze_net(spatial=True).state_dict()).spatial)

    def test_export_spatial_keeps_output_contract(self):
        import onnxruntime as ort
        from export_dual_onnx import export
        with tempfile.TemporaryDirectory() as tmp:
            gaze = Path(tmp) / "gaze.pt"
            torch.save({"gaze": M.build_gaze_net(spatial=True).state_dict()}, gaze)
            out = Path(tmp) / "eye.onnx"
            export(str(gaze), str(SHIPPED / "model_best.pt"), str(out), check=False)
            sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            y = sess.run(None, {sess.get_inputs()[0].name: torch.rand(1, 8, 128, 128).numpy()})[0]
            self.assertIn(y.shape[1], (12, 15))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless((SHIPPED / "model_best.pt").exists(), "shipped checkpoints not present")
class PresenceExportTests(unittest.TestCase):
    def test_export_appends_presence_in_side_order(self):
        import json
        import onnx
        import onnxruntime as ort
        import config as C
        from eye_seg import EyeSegNet
        from export_dual_onnx import export
        with tempfile.TemporaryDirectory() as tmp:
            seg_path = Path(tmp) / "seg.pt"
            torch.save({"model": EyeSegNet().state_dict()}, seg_path)
            presence = M.load_presence(seg_path)
            out = Path(tmp) / "eye.onnx"
            export(str(SHIPPED / "gaze_model_best.pt"), str(SHIPPED / "model_best.pt"), str(out), presence=presence)
            names = json.loads({p.key: p.value for p in onnx.load(out).metadata_props}["blendshape_names"])
            self.assertEqual(C.PRESENCE_NAMES + C.PUPIL_NAMES, names[-6:])
            x = torch.zeros(1, 8, 128, 128)
            x[:, C.EXPORT_FLIP_CH[0]] = torch.rand(128, 128)   # only the eye that feeds right* gets an image
            sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            y = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0][0]
            with torch.no_grad():
                right = presence(x[:, C.EXPORT_FLIP_CH[0]:C.EXPORT_FLIP_CH[0] + 1])[0].numpy()
                left = presence(x[:, C.EXPORT_NOFLIP_CH[0]:C.EXPORT_NOFLIP_CH[0] + 1])[0].numpy()
            for k, quantity in enumerate(("EyePresent", "EyePupil", "EyePupilVisible")):
                self.assertAlmostEqual(right[k], y[names.index("right" + quantity)], places=4)
                self.assertAlmostEqual(left[k], y[names.index("left" + quantity)], places=4)
