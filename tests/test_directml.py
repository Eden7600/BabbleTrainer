"""Opt-in hardware regression: TEST_DIRECTML=1 python -m unittest discover -s tests."""
import copy
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BACKBONE", "mobilenetv4_conv_small_050.e3000_r224_in1k")
import torch


@unittest.skipUnless(os.environ.get("TEST_DIRECTML") == "1", "Set TEST_DIRECTML=1 on a DirectML-capable machine")
class DirectMLTests(unittest.TestCase):
    def test_merged_mirror_matches_cpu_with_and_without_consensus(self):
        import torch_directml
        from model_dual import build_expr_net, DualTaskTower, MergedDualEye, build_consensus_heads, expr_feat_dim
        from evaluation import _NeutralGaze
        device = torch_directml.device()
        torch.manual_seed(19)
        expression = build_expr_net(pretrained=False).eval()
        inputs = torch.rand(2, 8, 128, 128)
        # An asymmetric pattern makes an incorrect/no mirror observable.
        inputs[0, :, :, :64] = 0
        inputs[1, :, :, 64:] = 1
        for consensus in (None, build_consensus_heads(expr_feat_dim(expression)).eval()):
            cpu = MergedDualEye(DualTaskTower(expression, _NeutralGaze()), consensus).eval()
            gpu = copy.deepcopy(cpu).to(device).eval()
            with torch.no_grad():
                expected = cpu(inputs)
                actual = gpu(inputs.to(device)).cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
