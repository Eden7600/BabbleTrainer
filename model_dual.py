from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from model import _small_init_head

GAZE_OUT = C.GAZE_SLICE[1] - C.GAZE_SLICE[0]   # 2 (pitch, yaw)
EXPR_OUT = C.EXPR_SLICE[1] - C.EXPR_SLICE[0]   # 4 (lid, widen, squint, brow)


class GlobalMaxPool2d(nn.Module):
    def forward(self, x):
        return x.amax(dim=(2, 3), keepdim=True)


class MicroChad(nn.Module):

    def __init__(self, out_count=2, in_ch=None, spatial=False):
        super().__init__()
        self.spatial = spatial
        in_ch = C.TEMPORAL_FRAMES if in_ch is None else in_ch
        self.conv1 = nn.Conv2d(in_ch, 14, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(14, 21, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(21, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 47, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(47, 70, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(70, 106, kernel_size=3, stride=1, padding=1)
        self.fc_gaze = nn.Linear(106 * 4 * 4 if spatial else 106, out_count)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.adaptive = GlobalMaxPool2d()
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.pool(self.act(self.conv3(x)))
        x = self.pool(self.act(self.conv4(x)))
        x = self.pool(self.act(self.conv5(x)))
        x = self.act(self.conv6(x))
        x = torch.flatten(x if self.spatial else self.adaptive(x), 1)
        return self.sigmoid(self.fc_gaze(x))


def _build_expr_tower(backbone, pretrained, out_ch):
    """MNv4/timm (or torchvision) expression backbone: TEMPORAL_FRAMES-ch in -> out_ch linear outputs."""
    in_ch = C.TEMPORAL_FRAMES
    if backbone in ("torchvision", "mobilenet_v2"):
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights  # lazy: the timm prod path excludes torchvision
        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        net = mobilenet_v2(weights=weights)
        old = net.features[0][0]
        new = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=(old.bias is not None))
        with torch.no_grad():
            if pretrained:
                w = old.weight.mean(dim=1, keepdim=True)
                new.weight.copy_(w.repeat(1, in_ch, 1, 1) * (3.0 / in_ch))
            else:
                nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
        net.features[0][0] = new
        net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(net.last_channel, out_ch))
        _small_init_head(net.classifier[1])
        return net
    import timm
    net = timm.create_model(backbone, pretrained=pretrained, num_classes=out_ch, drop_rate=0.2, in_chans=in_ch)
    _small_init_head(net.get_classifier())
    return net


class DualTaskTower(nn.Module):
    def __init__(self, expr: nn.Module, gaze: nn.Module):
        super().__init__()
        self.expr = expr
        self.gaze = gaze

    def forward(self, x):
        return torch.cat([self.gaze(x), self.expr(x)], dim=1)   # [B, GAZE_OUT + EXPR_OUT] = [B, 6]

    def head_parameters(self):
        expr_head = self.expr.classifier if hasattr(self.expr, "classifier") else self.expr.get_classifier()
        yield from expr_head.parameters()
        yield from self.gaze.fc_gaze.parameters()


def build_eye_tower(backbone: str | None = None, pretrained: bool = True) -> nn.Module:
    backbone = backbone or C.BACKBONE
    expr = _build_expr_tower(backbone, pretrained, EXPR_OUT)
    gaze = MicroChad(out_count=GAZE_OUT, spatial=C.GAZE_SPATIAL)
    return DualTaskTower(expr, gaze)


def build_expr_net(backbone: str | None = None, pretrained: bool = True) -> nn.Module:
    return _build_expr_tower(backbone or C.BACKBONE, pretrained, EXPR_OUT)


def build_gaze_net(spatial=None) -> nn.Module:
    return MicroChad(out_count=GAZE_OUT, spatial=C.GAZE_SPATIAL if spatial is None else spatial)


def gaze_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return ckpt.get("gaze", ckpt.get("student", ckpt)) if isinstance(ckpt, dict) else ckpt


def gaze_net_for(state_dict) -> nn.Module:
    """A MicroChad whose head matches a saved state dict (max-pooled or spatial)."""
    return MicroChad(out_count=GAZE_OUT, spatial=state_dict["fc_gaze.weight"].shape[1] != 106)


def warm_start_from(net, ckpt_path) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    src = ckpt.get("teacher", ckpt.get("student", ckpt)) if isinstance(ckpt, dict) else ckpt
    expr = net.expr if isinstance(net, DualTaskTower) else net
    tgt = expr.state_dict()
    expr.load_state_dict(src)   # strict full load (stem + backbone + head)
    print(f"warm-start: full load {len(tgt)} expr tensors from {Path(ckpt_path).name}", flush=True)
    return len(tgt)


def warm_start_gaze(gaze_net, ckpt_path) -> int:
    src = gaze_state(ckpt_path)
    own = gaze_net.state_dict()
    # A max-pooled checkpoint can seed a spatial net: every conv carries over, only the head starts fresh.
    keep = {k: v for k, v in src.items() if own[k].shape == v.shape}
    gaze_net.load_state_dict(keep, strict=len(keep) == len(own))
    fresh = sorted(set(own) - set(keep))
    print(f"warm-start gaze: loaded {len(keep)} MicroChad tensors from {Path(ckpt_path).name}"
          + (f"; fresh: {', '.join(fresh)}" if fresh else ""), flush=True)
    return len(keep)


def expr_feat_dim(expr_net) -> int:
    dev = next(expr_net.parameters()).device
    was_training = expr_net.training
    expr_net.eval()
    with torch.no_grad():
        d = expr_features(expr_net, torch.zeros(1, C.TEMPORAL_FRAMES, C.DUAL_INPUT_SIZE,
                                                C.DUAL_INPUT_SIZE, device=dev)).shape[1]
    expr_net.train(was_training)
    return int(d)


def expr_features(expr_net, x):
    if hasattr(expr_net, "forward_features") and hasattr(expr_net, "forward_head"):   # timm
        return expr_net.forward_head(expr_net.forward_features(x), pre_logits=True)
    fmap = expr_net.features(x)                                                       # torchvision MNv2
    return torch.flatten(F.adaptive_avg_pool2d(fmap, 1), 1)


def expr_forward(expr_net, x):
    if hasattr(expr_net, "forward_features") and hasattr(expr_net, "forward_head"):   # timm
        feat = expr_net.forward_head(expr_net.forward_features(x), pre_logits=True)
        return feat, expr_net.get_classifier()(feat)
    fmap = expr_net.features(x)                                                       # torchvision MNv2
    feat = torch.flatten(F.adaptive_avg_pool2d(fmap, 1), 1)
    return feat, expr_net.classifier(feat)


class ConsensusHead(nn.Module):

    def __init__(self, feat_dim, out_count, hidden=None):
        super().__init__()
        hidden = C.EXPR_CONSENSUS_HIDDEN if hidden is None else hidden
        self.fc1 = nn.Linear(feat_dim * 2, hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, out_count)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_left, feat_right):
        x = torch.cat([feat_left, feat_right], dim=1)
        return self.sigmoid(self.fc2(self.act(self.fc1(x))))


def build_consensus_heads(feat_dim, groups=None, hidden=None) -> nn.ModuleList:
    groups = C.CONSENSUS_GROUPS if groups is None else groups
    return nn.ModuleList([ConsensusHead(feat_dim, len(g), hidden) for g in groups])


class EyePresence(nn.Module):
    """Eye state for one equalized 128x128 eye frame from the segmentation net's encoder: headset-worn
    probability, plus pupil/iris ratio and pupil-visible probability when the checkpoint has state heads."""

    def __init__(self, seg):
        super().__init__()
        self.seg = seg
        self.pupil = getattr(seg, "has_state", False)

    def forward(self, x):
        if not self.pupil:
            return torch.sigmoid(self.seg.presence_logit(x))[:, None]
        presence, _, pupil, visible = self.seg.eye_state(x)
        return torch.stack([torch.sigmoid(presence), pupil, torch.sigmoid(visible)], 1)


def load_presence(path) -> nn.Module:
    from eye_seg import load_seg
    seg = load_seg(path)
    if not seg.has_presence:
        raise ValueError(f"{path} has no presence head; train it with eye_seg.py state")
    return EyePresence(seg).eval()


def presence_outputs(presence, x):
    """(B, 2 or 6) eye state in output side order from each eye's newest frame, grouped by quantity
    (presence, then pupil, then pupil visible); mirrors MergedDualEye's side logic."""
    left = presence(x[:, C.EXPORT_NOFLIP_CH[0]:C.EXPORT_NOFLIP_CH[0] + 1])   # the un-flipped eye feeds leftEye*
    right = presence(x[:, C.EXPORT_FLIP_CH[0]:C.EXPORT_FLIP_CH[0] + 1])
    if C.SWAP_OUTPUT_SIDES:
        right, left = left, right
    first, second = (right, left) if C.OUTPUT_SIDE_ORDER[0] == "right" else (left, right)
    return torch.cat([t for k in range(first.shape[1]) for t in (first[:, k:k + 1], second[:, k:k + 1])], dim=1)


class MergedDualEye(nn.Module):
    def __init__(self, tower: nn.Module, consensus: nn.ModuleList | None = None, presence: nn.Module | None = None):
        super().__init__()
        self.tower = tower
        self.consensus = consensus
        self.presence = presence
        self.right_first = (C.OUTPUT_SIDE_ORDER[0] == "right")
        # Channel fusion (deployment): both eyes emit one shared value for these EXPR channels.
        self.fuse_source = C.EXPR_FUSE_SOURCE
        flat = [c for g in C.CONSENSUS_GROUPS for c in g]            # consensus output order (expr-local)
        self._cons_col = {c: i for i, c in enumerate(flat)}          # expr-local -> combined column
        self.fuse_channels = list(C.EXPR_FUSE_CHANNELS)
        if self.fuse_source == "consensus":
            # consensus-sourced fusion only covers the consensus channels (widen/squint/brow)
            self.fuse_channels = [c for c in self.fuse_channels if c in self._cons_col]
        gather_idx = None
        if self.consensus is not None and self.fuse_source == "consensus" and self.fuse_channels:
            n = C.DUAL_NUM_PER_EYE                                   # 6 per eye
            idx = list(range(2 * n + len(flat)))                    # identity over the 15-vector
            right_off, left_off = (0, n) if self.right_first else (n, 0)
            comb_off = 2 * n                                         # combined block starts at 12
            for c in self.fuse_channels:
                pe = GAZE_OUT + c                                    # within-eye index of the fused channel
                idx[right_off + pe] = comb_off + self._cons_col[c]
                idx[left_off + pe] = comb_off + self._cons_col[c]
            gather_idx = torch.tensor(idx, dtype=torch.long)
        self.register_buffer("gather_idx", gather_idx, persistent=False)  # None when not consensus-fusing

    def _eye(self, x):
        feat, expr_logits = expr_forward(self.tower.expr, x)
        out = torch.cat([self.tower.gaze(x), expr_logits], dim=1)   # gaze (sigmoid) + expr (linear)
        return out, feat

    def _fuse_mean(self, out_left, out_right):
        lcols, rcols = [], []
        for i in range(out_left.shape[1]):
            c = i - GAZE_OUT
            if c in self.fuse_channels:
                m = 0.5 * (out_left[:, i:i + 1] + out_right[:, i:i + 1])
                lcols.append(m); rcols.append(m)
            else:
                lcols.append(out_left[:, i:i + 1]); rcols.append(out_right[:, i:i + 1])
        return torch.cat(lcols, dim=1), torch.cat(rcols, dim=1)

    def forward(self, x):
        noflip = x[:, C.EXPORT_NOFLIP_CH, :, :]                      # canonical-space eye -> leftEye*
        mirror_input = x[:, C.EXPORT_FLIP_CH, :, :]
        # torch-directml 0.2.5 can crash natively on flip; validation needs no GPU mirror kernel.
        flipped = (torch.flip(mirror_input.cpu(), dims=[-1]).to(x.device)
                   if x.device.type == "privateuseone" else torch.flip(mirror_input, dims=[-1]))
        out_left, feat_left = self._eye(noflip)
        out_right, feat_right = self._eye(flipped)
        out_right = torch.cat([out_right[:, 0:1], 1.0 - out_right[:, 1:2], out_right[:, 2:]], dim=1)
        right_side, left_side = (out_left, out_right) if C.SWAP_OUTPUT_SIDES else (out_right, out_left)
        if self.fuse_channels and self.fuse_source == "mean":
            right_side, left_side = self._fuse_mean(right_side, left_side)
        merged = torch.cat([right_side, left_side] if self.right_first else [left_side, right_side], dim=1)
        full = merged.clamp(0.0, 1.0)
        if self.consensus is not None:
            combined = torch.cat([h(feat_left, feat_right) for h in self.consensus], dim=1).clamp(0.0, 1.0)
            full = torch.cat([full, combined], dim=1)
            if self.gather_idx is not None:
                full = full.index_select(1, self.gather_idx)
        if self.presence is not None:
            full = torch.cat([full, presence_outputs(self.presence, x)], dim=1)
        return full
