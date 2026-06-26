import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

import config as C
import device as Dev
import bin_temporal_dataset as TD
from model import EMATeacher, save_checkpoint, reg_elementwise
from model_dual import (build_expr_net, warm_start_from, build_consensus_heads, expr_feat_dim,
                        expr_features)

GAZE_A, GAZE_B = C.GAZE_SLICE  
EXPR_A, EXPR_B = C.EXPR_SLICE  


_DECONF = None


def deconfuse_weight(tgt: torch.Tensor) -> torch.Tensor:
    global _DECONF
    if _DECONF is None or _DECONF.device != tgt.device:
        _DECONF = torch.tensor(C.EXPR_DECONFUSE, device=tgt.device, dtype=torch.float32)
    f = 1.0 - tgt.float().unsqueeze(2) * _DECONF.unsqueeze(0)
    return f.clamp_min(0.0).prod(dim=1).to(tgt.dtype)


def _eval_batches(ds, n, chunk=256):
    cached = getattr(ds, "_eval_cache", None)
    if cached is not None and cached[0] == n:
        return cached[1]
    idx = np.random.default_rng(0).choice(len(ds), size=min(n, len(ds)), replace=False)
    batches = []
    for s in range(0, len(idx), chunk):
        items = [ds[int(j)] for j in idx[s:s + chunk]]
        xb = (torch.stack([it[0] for it in items]) * 255.0).round_().clamp_(0, 255).to(torch.uint8)
        batches.append((xb, torch.stack([it[1] for it in items]), torch.stack([it[2] for it in items])))
    ds._eval_cache = (n, batches)
    return batches


@torch.no_grad()
def evaluate_expr(net, ds, device, n=4000):
    net.eval()
    preds, tgts = [], []
    for xb, lbl, _gv in _eval_batches(ds, n):
        x = Dev.to_input(xb.float().div_(255.0), device)
        with Dev.autocast_ctx(device):
            preds.append(net(x).float().cpu())
        tgts.append(lbl)
    p = torch.cat(preds).clamp(0, 1).numpy()
    t = torch.cat(tgts).numpy()[:, EXPR_A:EXPR_B]
    val_mse = float(((p - t) ** 2).mean())
    rs = []
    for c in range(EXPR_B - EXPR_A):
        tc, pc = t[:, c], p[:, c]
        rs.append(float(np.corrcoef(tc, pc)[0, 1]) if tc.std() > 1e-6 and pc.std() > 1e-6 else 0.0)
    net.train()
    return val_mse, rs


def default_workers():
    return min(8, max(2, (os.cpu_count() or 8) // 4))


def fit_expr(train_ds, val_ds, *, steps, batch=64, workers=None, lr_head, lr_backbone, weight_decay=1e-4,
             warmup, val_every, active_boost=4.0, pretrained=True, init=None, ckpt_dir, log="plain",
             temporal=False, lambda_temp=C.EXPR_LAMBDA_TEMP, mean_teacher=False,
             lambda_cons=C.EXPR_LAMBDA_CONS, cons_rampup=C.EXPR_CONS_RAMPUP,
             unlabeled_ds=None, unlabeled_batch=None, deconfuse=False,
             lr_warmup_head=None, freeze_backbone_warmup=False,
             even=False, even_weights=None, paired_ds=None, paired_batch=None,
             consensus=False, lambda_consensus=C.LAMBDA_CONSENSUS,
             progress_offset=0, progress_total=None, emit_header=True, emit_completion=True):
    
    if workers is None:
        workers = default_workers()
    device = Dev.get_device()
    total = progress_total if progress_total is not None else steps
    lr_warmup_head = lr_head if lr_warmup_head is None else lr_warmup_head
    do_even = even and paired_ds is not None
    do_cons = consensus and paired_ds is not None
    use_paired = do_even or do_cons
    even_weights = list(even_weights) if even_weights is not None else list(C.EXPR_EVEN_WEIGHTS)
    print(f"[expr] device: {device.type} | backbone: {C.BACKBONE} | in_chans: {C.TEMPORAL_FRAMES} | "
          f"batch: {batch} | workers: {workers} | even={do_even}"
          f"{('@' + str(even_weights)) if do_even else ''} consensus={do_cons} "
          f"freeze_warmup={freeze_backbone_warmup}", flush=True)

    w = train_ds.sampler_weights(active_boost, gaze_boost=0.0)
    sampler = WeightedRandomSampler(torch.as_tensor(w), num_samples=steps * batch, replacement=True)
    loader_kw = dict(batch_size=batch, sampler=sampler, num_workers=workers,
                     pin_memory=Dev.pin_memory(device), drop_last=True)
    if workers > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=4, timeout=180)
    loader = DataLoader(train_ds, **loader_kw)

    use_aj = (temporal or mean_teacher) and unlabeled_ds is not None
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
    paired_loader = paired_it = None
    pbatch = paired_batch or C.EXPR_EVEN_BATCH
    if use_paired:
        pwk = min(2, workers)
        pw_ = paired_ds.sampler_weights(active_boost)
        psamp = WeightedRandomSampler(torch.as_tensor(pw_), num_samples=steps * pbatch, replacement=True)
        pkw = dict(batch_size=pbatch, sampler=psamp, num_workers=pwk,
                   pin_memory=Dev.pin_memory(device), drop_last=True)
        if pwk > 0:
            pkw.update(persistent_workers=True, prefetch_factor=4, timeout=180)
        paired_loader = DataLoader(paired_ds, **pkw)
        paired_it = iter(paired_loader)

    net = Dev.model_to(build_expr_net(pretrained=pretrained), device)
    if init:
        warm_start_from(net, init)
    net = net.train()
    teacher = EMATeacher(net, 0.999)
    scaler = Dev.make_scaler(device)

    cons_heads = build_consensus_heads(expr_feat_dim(net)).to(device) if do_cons else None

    head = list(net.classifier.parameters())
    if cons_heads is not None:
        head = head + list(cons_heads.parameters())
    hid = {id(p) for p in head}
    bb = [p for p in net.parameters() if id(p) not in hid]
    opt = Dev.make_adamw([{"params": bb, "lr": lr_backbone},
                          {"params": head, "lr": lr_head}], device, weight_decay)
    if freeze_backbone_warmup and warmup > 0:
        for p in bb:
            p.requires_grad_(False)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    last_loss = 0.0
    t0 = time.time()
    if log == "frontend" and emit_header:
        print("\n=== Epoch 1/1 ===\n", flush=True)
    margin = torch.tensor(C.EXPR_TEMPORAL_MARGIN, device=device).view(1, -1)
    even_w = torch.tensor(even_weights, device=device, dtype=torch.float32).view(1, -1)
    it = iter(loader)
    for step in range(steps):
        imgs, tgt, _gv = next(it)        
        imgs = Dev.to_input(imgs, device)
        tgt = tgt.to(device)[:, EXPR_A:EXPR_B]

        if freeze_backbone_warmup and warmup > 0 and step == warmup:
            for p in bb:
                p.requires_grad_(True)

        # LR schedule (groups: [backbone, head])
        if step < warmup:
            if freeze_backbone_warmup:
                bb_lr, hd_lr = 0.0, lr_warmup_head  
            else:
                bb_lr = lr_backbone * (step + 1) / warmup 
                hd_lr = lr_head * (step + 1) / warmup
        else:
            cos = 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
            bb_lr, hd_lr = lr_backbone * cos, lr_head * cos
        opt.param_groups[0]["lr"] = bb_lr
        opt.param_groups[1]["lr"] = hd_lr

        if use_aj:
            try:
                cur_s, cur_w, nbr_w, pair_ok = next(unlab_it)
            except StopIteration: 
                unlab_it = iter(unlab_loader)
                cur_s, cur_w, nbr_w, pair_ok = next(unlab_it)
            cur_s = Dev.to_input(cur_s, device); cur_w = Dev.to_input(cur_w, device)
            nbr_w = Dev.to_input(nbr_w, device); pair_ok = pair_ok.to(device)

        if use_paired:
            try:
                pl, pr, pll, prl = next(paired_it)
            except StopIteration:
                paired_it = iter(paired_loader)
                pl, pr, pll, prl = next(paired_it)
            pl = Dev.to_input(pl, device); pr = Dev.to_input(pr, device)
            pll = pll.to(device)[:, EXPR_A:EXPR_B]; prl = prl.to(device)[:, EXPR_A:EXPR_B]

        with Dev.autocast_ctx(device):
            B = imgs.shape[0]
            parts = [imgs]
            if use_aj:
                parts += [cur_s, cur_w, nbr_w]
            if use_paired:
                parts += [pl, pr]
            outs = net(torch.cat(parts, dim=0)) if len(parts) > 1 else net(imgs)
            o = B
            pred = outs[:B]
            if use_aj:
                ub = cur_s.shape[0]
                pred_s = outs[o:o + ub]; o += ub
                pw = outs[o:o + ub]; o += ub
                pn = outs[o:o + ub]; o += ub
            if use_paired:
                pb = pl.shape[0]
                pred_pl = outs[o:o + pb]; o += pb
                pred_pr = outs[o:o + pb]; o += pb

            if deconfuse:
                w = deconfuse_weight(tgt)                  
                reg = (reg_elementwise(pred, tgt) * w).sum() / w.sum().clamp_min(1.0)
            else:
                reg = reg_elementwise(pred, tgt).mean()
            loss = reg + C.LAMBDA_RANGE * (F.relu(-pred) ** 2 + F.relu(pred - 1) ** 2).mean()

            if do_even:
                agree = (1.0 - (pll - prl).abs()).clamp(0.0, 1.0)
                even_l = (even_w * agree * (pred_pl - pred_pr) ** 2).sum() / agree.sum().clamp_min(1.0)
                loss = loss + even_l

            if use_aj and mean_teacher:
                with torch.no_grad():
                    t_pred = teacher.model(cur_w)
                lam = lambda_cons * min(1.0, step / max(1, cons_rampup)) ** 2
                loss = loss + lam * F.mse_loss(pred_s, t_pred)
            if use_aj and temporal:
                d = (pw - pn).abs()           
                l_temp = (F.relu(d - margin) ** 2).mean(dim=1)
                loss = loss + lambda_temp * (l_temp * pair_ok).sum() / pair_ok.sum().clamp_min(1.0)

            if do_cons:
                with torch.no_grad():
                    tfeat = expr_features(teacher.model, torch.cat([pl, pr], dim=0))
                pb = pl.shape[0]
                tf_l, tf_r = tfeat[:pb].detach(), tfeat[pb:].detach()
                tgt_shared = 0.5 * (pll + prl)       
                cons_loss = 0.0
                for h, g in zip(cons_heads, C.CONSENSUS_GROUPS):
                    cons_loss = cons_loss + reg_elementwise(h(tf_l, tf_r), tgt_shared[:, g]).mean()
                loss = loss + lambda_consensus * cons_loss

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        teacher.update(net)
        last_loss = float(loss.item())

        if log == "frontend":
            print("\rBatch %u/%u, Loss: %.6f" % (progress_offset + step + 1, total, last_loss),
                  end="", flush=True)

        if step % val_every == 0 or step == steps - 1:
            vmse, vr = evaluate_expr(teacher.model, val_ds, device)
            tag = ""
            if vmse < best:
                best = vmse
                save_checkpoint(ckpt_dir / "model_best.pt", net, teacher, step, {"val_mse": vmse},
                                consensus=cons_heads)
                tag = " *"
            if log == "plain":
                sps = (time.time() - t0) / (step + 1)
                rstr = " ".join(f"{nm}={r:.2f}" for nm, r in zip(C.DUAL_PER_EYE[EXPR_A:EXPR_B], vr))
                print(f"[{step:5d}/{steps}] loss={last_loss:.4f} val_mse={vmse:.4f} "
                      f"r({rstr}) ({sps:.2f}s/step){tag}", flush=True)

    save_checkpoint(ckpt_dir / "model_last.pt", net, teacher, steps, consensus=cons_heads)
    if log == "frontend" and emit_completion:
        print("\nEpoch 1/1 completed in %.2fs. Average loss: %.6f\n" % (time.time() - t0, last_loss), flush=True)
    return ckpt_dir / "model_best.pt", best


def fit_expr_stages(train_ds, val_ds, *, batch=64, workers=None, weight_decay=1e-4, val_every,
                    active_boost=4.0, pretrained=True, init=None, ckpt_dir,
                    unlabeled_ds=None, unlabeled_batch=None, paired_ds=None, paired_batch=None,
                    deconfuse=True, even=True, even_weights=None, consensus=True,
                    lambda_cons=C.EXPR_LAMBDA_CONS,
                    stage_a_steps=C.EXPR_STAGE_A_STEPS, stage_b_steps=C.EXPR_STAGE_B_STEPS,
                    log="plain", progress_offset=0, progress_total=None,
                    emit_header=True, emit_completion=True):

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    total = progress_total if progress_total is not None else (stage_a_steps + stage_b_steps)
    a_dir, b_dir = ckpt_dir / "stageA", ckpt_dir / "stageB"

    a_ckpt, _ = fit_expr(
        train_ds, val_ds, steps=stage_a_steps, batch=batch, workers=workers,
        lr_head=C.EXPR_STAGE_A_LR_HEAD, lr_backbone=C.EXPR_STAGE_A_LR_BACKBONE,
        lr_warmup_head=C.EXPR_STAGE_A_LR_WARMUP_HEAD, freeze_backbone_warmup=True,
        warmup=C.EXPR_STAGE_A_WARMUP, weight_decay=weight_decay, val_every=val_every,
        active_boost=active_boost, pretrained=pretrained, init=init, ckpt_dir=a_dir,
        temporal=False, mean_teacher=False, unlabeled_ds=None,
        even=even, even_weights=even_weights, paired_ds=paired_ds, paired_batch=paired_batch, consensus=False,
        deconfuse=deconfuse, log=log, progress_offset=progress_offset, progress_total=total,
        emit_header=emit_header, emit_completion=False)

    b_ckpt, best = fit_expr(
        train_ds, val_ds, steps=stage_b_steps, batch=batch, workers=workers,
        lr_head=C.EXPR_STAGE_B_LR_HEAD, lr_backbone=C.EXPR_STAGE_B_LR_BACKBONE, warmup=0,
        weight_decay=weight_decay, val_every=val_every, active_boost=active_boost,
        pretrained=False, init=str(a_ckpt), ckpt_dir=b_dir,
        temporal=unlabeled_ds is not None, mean_teacher=unlabeled_ds is not None,
        unlabeled_ds=unlabeled_ds, unlabeled_batch=unlabeled_batch, lambda_cons=lambda_cons,
        even=even, even_weights=even_weights, paired_ds=paired_ds, paired_batch=paired_batch,
        consensus=consensus and C.DUAL_CONSENSUS_HEAD,
        deconfuse=deconfuse, log=log, progress_offset=progress_offset + stage_a_steps,
        progress_total=total, emit_header=False, emit_completion=emit_completion)
    final = ckpt_dir / "model_best.pt"
    shutil.copyfile(b_ckpt, final)
    return final, best
