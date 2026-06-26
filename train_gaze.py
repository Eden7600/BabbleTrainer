import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import config as C
import device as Dev
import bin_temporal_dataset as TD
from model import _cpu_state_dict, EMATeacher
from model_dual import build_gaze_net, warm_start_gaze
from train_dual import default_workers, _eval_batches

GAZE_A, GAZE_B = C.GAZE_SLICE     # (0, 2)


def save_gaze_checkpoint(path, net, step, extra=None):
    torch.save({"gaze": _cpu_state_dict(net), "step": step, "arch": "microchad",
                **(extra or {})}, path)


@torch.no_grad()
def evaluate_gaze(net, ds, device, n=4000):
    net.eval()
    preds, tgts, gvs = [], [], []
    for xb, lbl, gv in _eval_batches(ds, n):
        x = Dev.to_input(xb.float().div_(255.0), device)
        with Dev.autocast_ctx(device):
            preds.append(net(x).float().cpu())
        tgts.append(lbl)
        gvs.append(gv)
    p = torch.cat(preds).clamp(0, 1).numpy()                 # [N,2]
    t = torch.cat(tgts).numpy()[:, GAZE_A:GAZE_B]            # [N,2]
    gv = torch.cat(gvs).numpy() > 0.5
    sel = gv if gv.any() else np.ones(len(p), bool)
    mse = float(((p[sel] - t[sel]) ** 2).mean())
    mae_deg = float(np.abs(p[sel] - t[sel]).mean()) * 2 * C.GAZE_RANGE_DEG
    rs = []
    for c in range(GAZE_B - GAZE_A):
        tc, pc = t[sel, c], p[sel, c]
        rs.append(float(np.corrcoef(tc, pc)[0, 1]) if tc.std() > 1e-6 and pc.std() > 1e-6 else 0.0)
    net.train()
    return mse, rs, mae_deg


def fit_gaze(train_ds, val_ds, *, steps=C.GAZE_STEPS, batch=C.GAZE_BATCH, workers=None,
             lr=C.GAZE_LR, weight_decay=C.GAZE_WEIGHT_DECAY, warmup=C.GAZE_WARMUP,
             eta_min=C.GAZE_ETA_MIN, val_every=500, init=None, ckpt_dir, log="plain",
             temporal=False, lambda_temp=C.GAZE_LAMBDA_TEMP, mean_teacher=False,
             lambda_cons=C.GAZE_LAMBDA_CONS, cons_rampup=C.GAZE_CONS_RAMPUP,
             unlabeled_ds=None, unlabeled_batch=None,
             progress_offset=0, progress_total=None, emit_header=True, emit_completion=True):
    if workers is None:
        workers = default_workers()
    device = Dev.get_device()
    total = progress_total if progress_total is not None else steps
    print(f"[gaze] device: {device.type} | MicroChad | in_chans: {C.TEMPORAL_FRAMES} | "
          f"batch: {batch} | workers: {workers}", flush=True)

    w = train_ds.gaze_sampler_weights()
    sampler = WeightedRandomSampler(torch.as_tensor(w), num_samples=steps * batch, replacement=True)
    loader_kw = dict(batch_size=batch, sampler=sampler, num_workers=workers,
                     pin_memory=Dev.pin_memory(device), drop_last=True)
    if workers > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=4, timeout=180)
    loader = DataLoader(train_ds, **loader_kw)

    net = Dev.model_to(build_gaze_net(), device)
    if init:
        warm_start_gaze(net, init)
    net = net.train()
    scaler = Dev.make_scaler(device)
    opt = Dev.make_adamw([{"params": list(net.parameters()), "lr": lr}], device, weight_decay)
    use_aj = (temporal or mean_teacher) and unlabeled_ds is not None
    teacher = EMATeacher(net, 0.999) if (use_aj and mean_teacher) else None
    margin = torch.tensor(C.GAZE_TEMPORAL_MARGIN, device=device).view(1, -1)
    unlab_loader = unlab_it = None
    if use_aj:
        ub = unlabeled_batch or batch
        uw = min(2, workers)
        ukw = dict(batch_size=ub, shuffle=True, num_workers=uw,
                   pin_memory=Dev.pin_memory(device), drop_last=True)
        if uw > 0:
            ukw.update(persistent_workers=True, prefetch_factor=4, timeout=180)
        unlab_loader = DataLoader(unlabeled_ds, **ukw)
        unlab_it = iter(unlab_loader)
    deployed = lambda: teacher.model if teacher is not None else net   # what evaluate/save use

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    last_loss = 0.0
    t0 = time.time()
    if log == "frontend" and emit_header:
        print("\n=== Epoch 1/1 ===\n", flush=True)
    it = iter(loader)
    for step in range(steps):
        imgs, tgt, _gv = next(it)
        imgs = Dev.to_input(imgs, device)
        tgt = tgt.to(device)

        if step < warmup:
            cur = lr * (step + 1) / warmup
        else: 
            cur = eta_min + 0.5 * (lr - eta_min) * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup)))
        for g in opt.param_groups:
            g["lr"] = cur

        with Dev.autocast_ctx(device):
            pred = net(imgs)
            loss = F.mse_loss(pred, tgt[:, GAZE_A:GAZE_B]) 
            if use_aj:
                # UNLABELED footage stream consistency + temporal only here
                try:
                    cur_s, cur_w, nbr_w, pair_ok = next(unlab_it)
                except StopIteration:
                    unlab_it = iter(unlab_loader)
                    cur_s, cur_w, nbr_w, pair_ok = next(unlab_it)
                cur_s = Dev.to_input(cur_s, device); cur_w = Dev.to_input(cur_w, device)
                nbr_w = Dev.to_input(nbr_w, device); pair_ok = pair_ok.to(device)
                if mean_teacher:
                    pred_s = net(cur_s)
                    with torch.no_grad():
                        t_pred = teacher.model(cur_w)
                    lam = lambda_cons * min(1.0, step / max(1, cons_rampup)) ** 2
                    loss = loss + lam * F.mse_loss(pred_s, t_pred)
                if temporal:
                    pw = net(cur_w); pn = net(nbr_w)
                    d = (pw - pn).abs()
                    l_temp = (F.relu(d - margin) ** 2).mean(dim=1)
                    loss = loss + lambda_temp * (l_temp * pair_ok).sum() / pair_ok.sum().clamp_min(1.0)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if teacher is not None:
            teacher.update(net)
        last_loss = float(loss.item())

        if log == "frontend":
            print("\rBatch %u/%u, Loss: %.6f" % (progress_offset + step + 1, total, last_loss),
                  end="", flush=True)

        if step % val_every == 0 or step == steps - 1:
            vmse, vr, gmae = evaluate_gaze(deployed(), val_ds, device)
            tag = ""
            if vmse < best:
                best = vmse
                save_gaze_checkpoint(ckpt_dir / "model_best.pt", deployed(), step,
                                     {"val_gaze_mse": vmse, "val_gaze_mae_deg": gmae})
                tag = " *"
            if log == "plain":
                sps = (time.time() - t0) / (step + 1)
                rstr = " ".join(f"{nm}={r:.2f}" for nm, r in zip(C.DUAL_PER_EYE[GAZE_A:GAZE_B], vr))
                print(f"[{step:5d}/{steps}] loss={last_loss:.4f} val_gaze_mse={vmse:.4f} "
                      f"r({rstr}) gazeMAE={gmae:.1f}deg ({sps:.2f}s/step){tag}", flush=True)

    save_gaze_checkpoint(ckpt_dir / "model_last.pt", deployed(), steps)
    if log == "frontend" and emit_completion:
        print("\nEpoch 1/1 completed in %.2fs. Average loss: %.6f\n" % (time.time() - t0, last_loss), flush=True)
    return ckpt_dir / "model_best.pt", best
