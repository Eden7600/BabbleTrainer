import math
import os
import random
from contextlib import nullcontext

import numpy as np
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
    """Pick a training device.

    Priority: an NVIDIA CUDA GPU (native, fastest) -> the fastest DirectML adapter (any-vendor GPU on
    Windows) -> CPU. Overrides via the DEVICE env var:
      DEVICE=cpu           -> force CPU
      DEVICE=cuda[:<idx>]  -> force a CUDA GPU (errors out if CUDA is unavailable)
      DEVICE=<int>         -> CUDA index <idx> when CUDA is present, else DirectML adapter index
    """
    forced = os.environ.get("DEVICE", "auto").lower()
    if forced == "cpu":
        return torch.device("cpu")

    cuda_ok = torch.cuda.is_available()
    if forced.startswith("cuda"):
        # explicit request: honor it even if is_available() is wrong, but warn if it looks down
        if not cuda_ok:
            print("DEVICE=cuda requested but torch.cuda.is_available() is False; trying anyway.", flush=True)
        return torch.device(forced)
    if cuda_ok and (forced in ("auto", "") or forced.isdigit()):
        idx = int(forced) if forced.isdigit() else torch.cuda.current_device()
        print("Using CUDA device:", torch.cuda.get_device_name(idx), flush=True)
        return torch.device("cuda", idx)

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


def _amp_dtype(device):
    """Autocast dtype for a CUDA device, or None for fp32 (CPU / DirectML).

    Prefer bf16 on Ampere+ (same exponent range as fp32, so no loss scaling needed); fall back to
    fp16 (with a real GradScaler) on older CUDA GPUs."""
    if getattr(device, "type", None) != "cuda":
        return None
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def autocast_ctx(device):
    """Mixed-precision autocast on CUDA (bf16/fp16); no-op fp32 on DirectML/CPU."""
    dt = _amp_dtype(device)
    if dt is None:
        return nullcontext()
    return torch.autocast("cuda", dtype=dt)


def make_scaler(device):
    """GradScaler enabled only for CUDA fp16 (bf16 and fp32 don't need loss scaling). The training
    loop's scaler.scale/step/update calls become pass-throughs when it is disabled."""
    enabled = _amp_dtype(device) == torch.float16
    return torch.amp.GradScaler("cuda" if enabled else "cpu", enabled=enabled)


def to_input(t, device):
    """Move a batch to the device (NCHW). non_blocking pairs with pinned host memory on CUDA; it is
    ignored on DirectML/CPU."""
    return t.to(device, non_blocking=True)


def model_to(net, device):
    """Move a module to the device (NCHW)."""
    return net.to(device)


def pin_memory(device) -> bool:
    """Pin DataLoader host buffers for faster async H2D copies on CUDA only."""
    return getattr(device, "type", None) == "cuda"


def _parse_cpu_list(s):
    out = set()
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def _cpu_topology():
    """(physical_cores, logical_cpus) within the current CPU-affinity mask."""
    try:
        allowed = set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed = set(range(os.cpu_count() or 1))
    logical = len(allowed)
    try:
        cores = set()
        for c in allowed:
            sl = open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip()
            sibs = _parse_cpu_list(sl) & allowed
            cores.add(min(sibs) if sibs else c)
        phys = len(cores) or logical
    except Exception:
        phys = max(1, logical // 2)
    return phys, logical


def configure_runtime(device):
    """CPU: torch intra-op = physical cores (HT/idle lanes left for the bursty data workers; override
    BB_TORCH_THREADS). CUDA: cudnn autotune (shapes are fixed) + TF32 (BB_CUDNN_BENCH=0 disables)."""
    env = os.environ.get("BB_TORCH_THREADS")
    if getattr(device, "type", None) == "cuda":
        if env:
            torch.set_num_threads(max(1, int(env)))
        if os.environ.get("BB_CUDNN_BENCH", "1") != "0":
            torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return
    if env:
        torch.set_num_threads(max(1, int(env)))
    else:
        phys, _ = _cpu_topology()
        torch.set_num_threads(max(1, phys))


def seed_everything(seed):
    """Seed python/numpy/torch (+CUDA) RNGs for reproducible training. Worker RNGs are seeded in
    worker_init from the (now-deterministic) per-worker torch seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init(_):
    """DataLoader workers: single-thread cv2/torch (BB_CV2_THREADS=0 disables) and seed python/numpy
    RNGs from the worker's torch seed so augmentation/mixup are reproducible."""
    n = int(os.environ.get("BB_CV2_THREADS", "1"))
    if n > 0:
        import cv2
        cv2.setNumThreads(n)
        torch.set_num_threads(1)
    s = torch.initial_seed() % (2 ** 32)
    random.seed(s)
    np.random.seed(s)


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
