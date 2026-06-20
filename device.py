import math
import os
from contextlib import nullcontext

import torch


def get_device():
    """DirectML (any-vendor GPU on Windows), falling back to CPU if torch_directml is unavailable.
    DEVICE=cpu forces CPU."""
    if os.environ.get("DEVICE", "auto").lower() == "cpu":
        return torch.device("cpu")
    try:
        import torch_directml
        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


def autocast_ctx(device):
    """No-op context — DirectML/CPU train fp32 (no mixed-precision autocast)."""
    return nullcontext()


def make_scaler(device):
    """Disabled GradScaler — a pass-through (scale/step/update are no-ops) so the training loop is
    unchanged while running fp32 on DirectML/CPU."""
    return torch.amp.GradScaler("cpu", enabled=False)


def to_input(t, device):
    """Move a batch to the device (NCHW — DirectML/CPU don't use channels_last)."""
    return t.to(device)


def model_to(net, device):
    """Move a module to the device (NCHW)."""
    return net.to(device)


def pin_memory(device) -> bool:
    return False


class ForeachAdamW(torch.optim.Optimizer):
    """Lerp-free, foreach AdamW — mathematically identical to torch.optim.AdamW but built only from
    ops DirectML runs natively (mul/add/addcmul/sqrt/addcdiv). torch's AdamW uses aten::lerp for the
    first-moment update, which falls back to CPU on DirectML and costs ~90 ms/step on MNv4; this stays
    on-device (~25 ms). Honors per-group lr + decoupled weight_decay (set by train_dual's LR schedule).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue
            grads = [p.grad for p in params]
            m, v, steps = [], [], []
            for p in params:
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["step"] += 1
                m.append(st["m"]); v.append(st["v"]); steps.append(st["step"])
            t = steps[0]  # all params in a group step together every iteration
            if wd:
                torch._foreach_mul_(params, 1 - lr * wd)              # decoupled weight decay
            torch._foreach_mul_(m, b1); torch._foreach_add_(m, grads, alpha=1 - b1)
            torch._foreach_mul_(v, b2); torch._foreach_addcmul_(v, grads, grads, value=1 - b2)
            denom = torch._foreach_sqrt(v)
            torch._foreach_div_(denom, math.sqrt(1 - b2 ** t))
            torch._foreach_add_(denom, eps)
            torch._foreach_addcdiv_(params, m, denom, value=-lr / (1 - b1 ** t))
        return loss


def make_adamw(param_groups, device, weight_decay):
    """The lerp-free ForeachAdamW (torch's AdamW lerp falls back to CPU on DirectML)."""
    return ForeachAdamW(param_groups, weight_decay=weight_decay)
