import math
import os
from contextlib import nullcontext

import torch


def _bench_directml_device(d, iters=50, warmup=5):
    """Average wall-clock seconds of a 2048x2048 matmul on DirectML device ``d``.

    DirectML dispatches ops asynchronously, so the original veryold tester (which just stopped the
    clock right after issuing ``x @ x``) mostly timed kernel *submission*, not execution. We force
    completion by reading a scalar back to the host (``.item()``), which blocks until the matmul is
    done — the matmul's result also makes the queue flush. We likewise materialize the input and run
    a few warmup iterations first so adapter init / shader compilation isn't charged to the timings."""
    import time

    x = torch.randn(2048, 2048, device=d)
    for _ in range(warmup):
        (x @ x).sum().item()  # init adapter, compile kernels, flush the queue

    times = []
    for _ in range(iters):
        x = torch.randn(2048, 2048, device=d)
        x.sum().item()              # ensure the input is fully materialized before timing
        start = time.perf_counter()
        y = x @ x
        y.sum().item()              # block until the matmul actually finishes
        times.append(time.perf_counter() - start)
    return sum(times) / len(times)


def _fastest_directml_device(torch_directml):
    """Benchmark every DirectML adapter with a 2048x2048 matmul and return the fastest one's index.

    Ported from the veryold trainermin.py GPU tester (warm each adapter, time matmuls, keep the
    lowest average), with reliable async-aware timing — see _bench_directml_device. Returns None if
    no adapter could be timed (caller falls back to CPU)."""
    best_time = float("inf")
    best_idx = None
    for i in range(torch_directml.device_count()):
        try:
            avg = _bench_directml_device(torch_directml.device(i))
            print(i, torch_directml.device_name(i), avg, flush=True)
        except Exception:
            continue  # skip an adapter that fails to allocate/benchmark

        if avg < best_time:
            best_time = avg
            best_idx = i
    return best_idx


def get_device():
    """Fastest DirectML adapter (any-vendor GPU on Windows), falling back to CPU if torch_directml is
    unavailable. DEVICE=cpu forces CPU; DEVICE=<int> forces a specific DirectML adapter index."""
    forced = os.environ.get("DEVICE", "auto").lower()
    if forced == "cpu":
        return torch.device("cpu")
    try:
        import torch_directml
        if forced.isdigit():
            return torch_directml.device(int(forced))
        if torch_directml.device_count() <= 1:
            return torch_directml.device()
        best_idx = _fastest_directml_device(torch_directml)
        if best_idx is None:
            return torch.device("cpu")
        print("Using DirectML device:", torch_directml.device_name(best_idx), flush=True)
        return torch_directml.device(best_idx)
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
