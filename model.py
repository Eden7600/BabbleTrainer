import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


def reg_elementwise(pred, target):
    if C.REG_LOSS == "l1":
        return (pred - target).abs()
    if C.REG_LOSS == "huber":
        return F.smooth_l1_loss(pred, target, reduction="none", beta=C.HUBER_BETA)
    return (pred - target) ** 2


def _small_init_head(linear: nn.Linear) -> None:
    nn.init.normal_(linear.weight, std=0.01)
    nn.init.zeros_(linear.bias)


class EMATeacher:

    def __init__(self, student: nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(student).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module):
        ep, sp = list(self.model.parameters()), list(student.parameters())
        try:                                            # fused foreach fast path (DirectML)
            torch._foreach_mul_(ep, self.decay)
            torch._foreach_add_(ep, sp, alpha=1 - self.decay)
        except Exception:                               # generic fallback for backends without foreach
            for e, p in zip(ep, sp):
                e.mul_(self.decay).add_(p, alpha=1 - self.decay)
        eb, sb = list(self.model.buffers()), list(student.buffers())
        try:
            if eb:
                torch._foreach_copy_(eb, sb)
        except Exception:
            for e, b in zip(eb, sb):
                e.copy_(b)


def _cpu_state_dict(m: nn.Module) -> dict:
    m = getattr(m, "_orig_mod", m)
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}


def save_checkpoint(path, student: nn.Module, teacher: EMATeacher, step: int, extra: dict | None = None,
                    consensus: nn.Module | None = None):
    ckpt = {
        "student": _cpu_state_dict(student),
        "teacher": _cpu_state_dict(teacher.model),
        "step": step,
        "backbone": C.BACKBONE,
        **(extra or {}),
    }
    if consensus is not None:
        ckpt["consensus"] = _cpu_state_dict(consensus)
    torch.save(ckpt, path)
