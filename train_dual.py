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
import training_progress as Progress
import evaluation as Eval

GAZE_A, GAZE_B = C.GAZE_SLICE  
EXPR_A, EXPR_B = C.EXPR_SLICE  


_DECONF = None


def deconfuse_weight(tgt: torch.Tensor, valid=None) -> torch.Tensor:
    global _DECONF
    if _DECONF is None or _DECONF.device != tgt.device:
        _DECONF = torch.tensor(C.EXPR_DECONFUSE, device=tgt.device, dtype=torch.float32)
    drivers = tgt if valid is None else tgt * valid
    f = 1.0 - drivers.float().unsqueeze(2) * _DECONF.unsqueeze(0)
    return f.clamp_min(0.0).prod(dim=1).to(tgt.dtype)



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
             progress_offset=0, progress_total=None, emit_header=True, emit_completion=True,
             validation_paired=None, preserve_channels=(), baseline_report=None):
    
    if workers is None:
        workers = default_workers()
    if validation_paired is None:
        raise ValueError("Merged checkpoint selection requires paired validation inputs")
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
    if not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("No usable expression labels; repeat calibration")
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
        ukw = dict(batch_size=min(ub, len(unlabeled_ds)), shuffle=True, num_workers=uw,
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
    preserved = None
    if preserve_channels:
        # A fixed encoder plus fixed output rows preserves unsupported outputs without pseudo-labels.
        for p in bb:
            p.requires_grad_(False)
        net.eval()
        classifier = [m for m in net.classifier.modules() if isinstance(m, torch.nn.Linear)][-1]
        teacher_classifier = [m for m in teacher.model.classifier.modules() if isinstance(m, torch.nn.Linear)][-1]
        preserved = (classifier.weight[preserve_channels, :].detach().clone(),
                     classifier.bias[preserve_channels].detach().clone())

        def restore_preserved_outputs():
            with torch.no_grad():
                for target in (classifier, teacher_classifier):
                    target.weight[preserve_channels, :] = preserved[0]
                    target.bias[preserve_channels] = preserved[1]
                # EMA arithmetic can round unchanged values; copy frozen parameters exactly.
                for source, target in zip(net.parameters(), teacher.model.parameters()):
                    if not source.requires_grad:
                        target.copy_(source)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_report = None
    best_eligible = False
    last_loss = 0.0
    t0 = time.time()
    if log == "frontend" and emit_header:
        print("\n=== Epoch 1/1 ===\n", flush=True)
    margin = torch.tensor(C.EXPR_TEMPORAL_MARGIN, device=device).view(1, -1)
    even_w = torch.tensor(even_weights, device=device, dtype=torch.float32).view(1, -1)
    it = iter(loader)
    for step in range(steps):
        imgs, tgt, _gv, valid = next(it)
        imgs = Dev.to_input(imgs, device)
        tgt = tgt.to(device)[:, EXPR_A:EXPR_B]
        valid = valid.to(device)[:, EXPR_A:EXPR_B].float()

        if freeze_backbone_warmup and warmup > 0 and step == warmup and not preserve_channels:
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
                pl, pr, pll, prl, plm, prm = next(paired_it)
            except StopIteration:
                paired_it = iter(paired_loader)
                pl, pr, pll, prl, plm, prm = next(paired_it)
            pl = Dev.to_input(pl, device); pr = Dev.to_input(pr, device)
            pll = pll.to(device)[:, EXPR_A:EXPR_B]; prl = prl.to(device)[:, EXPR_A:EXPR_B]
            pair_valid = (plm.to(device)[:, EXPR_A:EXPR_B] & prm.to(device)[:, EXPR_A:EXPR_B]).float()

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
                w = deconfuse_weight(tgt, valid) * valid
                reg = (reg_elementwise(pred, tgt) * w).sum() / w.sum().clamp_min(1.0)
            else:
                reg = (reg_elementwise(pred, tgt) * valid).sum() / valid.sum().clamp_min(1)
            loss = reg + C.LAMBDA_RANGE * (F.relu(-pred) ** 2 + F.relu(pred - 1) ** 2).mean()

            if do_even:
                agree = (1.0 - (pll - prl).abs()).clamp(0.0, 1.0) * pair_valid
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
                    cons_loss = cons_loss + (reg_elementwise(h(tf_l, tf_r), tgt_shared[:, g]) * pair_valid[:, g]).sum() / pair_valid[:, g].sum().clamp_min(1)
                loss = loss + lambda_consensus * cons_loss

        if not torch.isfinite(loss):
            raise ValueError("Expression optimization produced a non-finite loss")
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        teacher.update(net)
        if preserved is not None:
            restore_preserved_outputs()
        last_loss = float(loss.item())

        Progress.emit("expression", progress_offset + step + 1, total, {"train_loss": last_loss})
        if log == "frontend" and not Progress.structured():
            print("\rBatch %u/%u, Loss: %.6f" % (progress_offset + step + 1, total, last_loss),
                  end="", flush=True)

        if step % val_every == 0 or step == steps - 1:
            report = Eval.evaluate_expression(teacher.model, cons_heads, validation_paired, device)
            vmse, vr = Eval.score(report), []
            tag = ""
            eligible, reason = (Eval.choose_candidate(baseline_report, report) if baseline_report is not None
                                else (True, "No pretrained comparison requested"))
            # Retain the best safe checkpoint, rather than discarding it when a later,
            # lower overall error hides a regression on an active expression.
            if (eligible and not best_eligible) or (eligible == best_eligible and vmse < best):
                best = vmse
                best_eligible = eligible
                best_report = report
                best_report["pretrained_comparison"] = dict(eligible=eligible, reason=reason)
                save_checkpoint(ckpt_dir / "model_best.pt", net, teacher, step, {"validation": report},
                                consensus=cons_heads)
                tag = " *"
            if log == "plain":
                sps = (time.time() - t0) / (step + 1)
                rstr = " ".join(f"{nm}={r:.2f}" for nm, r in zip(C.DUAL_PER_EYE[EXPR_A:EXPR_B], vr))
                print(f"[{step:5d}/{steps}] loss={last_loss:.4f} validation_macro_mae={vmse:.4f} "
                      f"r({rstr}) ({sps:.2f}s/step){tag}", flush=True)

    save_checkpoint(ckpt_dir / "model_last.pt", net, teacher, steps, consensus=cons_heads)
    if log == "frontend" and emit_completion:
        print("\nExpression stage completed in %.2fs. Last batch loss: %.6f\n" % (time.time() - t0, last_loss), flush=True)
    return ckpt_dir / "model_best.pt", best_report


def fit_expr_stages(train_ds, val_ds, *, batch=64, workers=None, weight_decay=1e-4, val_every,
                    active_boost=4.0, pretrained=True, init=None, ckpt_dir,
                    unlabeled_ds=None, unlabeled_batch=None, paired_ds=None, paired_batch=None,
                    deconfuse=True, even=True, even_weights=None, consensus=True,
                    lambda_cons=C.EXPR_LAMBDA_CONS,
                    stage_a_steps=C.EXPR_STAGE_A_STEPS, stage_b_steps=C.EXPR_STAGE_B_STEPS,
                    log="plain", progress_offset=0, progress_total=None,
                    emit_header=True, emit_completion=True, validation_paired=None, preserve_channels=()):

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    total = progress_total if progress_total is not None else (stage_a_steps + stage_b_steps)
    a_dir, b_dir = ckpt_dir / "stageA", ckpt_dir / "stageB"

    device = Dev.get_device()
    baseline = Dev.model_to(build_expr_net(pretrained=pretrained), device).eval()
    baseline_heads = None
    if init:
        warm_start_from(baseline, init)
        original = torch.load(init, map_location="cpu", weights_only=True)
        if isinstance(original, dict) and original.get("consensus") and C.DUAL_CONSENSUS_HEAD:
            baseline_heads = build_consensus_heads(expr_feat_dim(baseline)).to(device)
            baseline_heads.load_state_dict(original["consensus"])
            baseline_heads.eval()
    baseline_report = Eval.evaluate_expression(baseline, baseline_heads, validation_paired, device)
    baseline_ckpt = ckpt_dir / "baseline.pt"
    save_checkpoint(baseline_ckpt, baseline, EMATeacher(baseline, .999), 0,
                    {"validation": baseline_report}, consensus=baseline_heads)
    del baseline, baseline_heads

    a_ckpt, a_report = fit_expr(
        train_ds, val_ds, steps=stage_a_steps, batch=batch, workers=workers,
        lr_head=C.EXPR_STAGE_A_LR_HEAD, lr_backbone=C.EXPR_STAGE_A_LR_BACKBONE,
        lr_warmup_head=C.EXPR_STAGE_A_LR_WARMUP_HEAD, freeze_backbone_warmup=True,
        warmup=C.EXPR_STAGE_A_WARMUP, weight_decay=weight_decay, val_every=val_every,
        active_boost=active_boost, pretrained=False, init=str(baseline_ckpt), ckpt_dir=a_dir,
        temporal=False, mean_teacher=False, unlabeled_ds=None,
        even=even, even_weights=even_weights, paired_ds=paired_ds, paired_batch=paired_batch, consensus=False,
        deconfuse=deconfuse, log=log, progress_offset=progress_offset, progress_total=total,
        emit_header=emit_header, emit_completion=False, validation_paired=validation_paired,
        preserve_channels=preserve_channels, baseline_report=baseline_report)

    use_a, a_reason = Eval.choose_candidate(baseline_report, a_report)

    b_ckpt, best = fit_expr(
        train_ds, val_ds, steps=stage_b_steps, batch=batch, workers=workers,
        lr_head=C.EXPR_STAGE_B_LR_HEAD, lr_backbone=C.EXPR_STAGE_B_LR_BACKBONE, warmup=0,
        weight_decay=weight_decay, val_every=val_every, active_boost=active_boost,
        pretrained=False, init=str(a_ckpt if use_a else baseline_ckpt), ckpt_dir=b_dir,
        temporal=unlabeled_ds is not None, mean_teacher=unlabeled_ds is not None,
        unlabeled_ds=unlabeled_ds, unlabeled_batch=unlabeled_batch, lambda_cons=lambda_cons,
        even=even, even_weights=even_weights, paired_ds=paired_ds, paired_batch=paired_batch,
        consensus=consensus and C.DUAL_CONSENSUS_HEAD,
        deconfuse=deconfuse, log=log, progress_offset=progress_offset + stage_a_steps,
        progress_total=total, emit_header=False, emit_completion=emit_completion,
        validation_paired=validation_paired, preserve_channels=preserve_channels, baseline_report=baseline_report)
    final = ckpt_dir / "model_best.pt"
    selected, selected_ckpt, selected_report = ("stage_a", a_ckpt, a_report) if use_a else (
        "baseline", baseline_ckpt, baseline_report)
    use_b, b_reason = Eval.choose_candidate(baseline_report, best)
    beats_current, current_reason = Eval.choose_candidate(selected_report, best)
    if use_b and beats_current:
        selected, selected_ckpt = "stage_b", b_ckpt
    shutil.copyfile(selected_ckpt, final)
    reason = ("Both stages rejected against pretrained expression baseline" if selected == "baseline" else
              "Stage B improves on the selected candidate and passes pretrained safeguards" if selected == "stage_b" else
              "Stage A passes pretrained safeguards; stage B does not safely improve it")
    return final, {"selected": selected, "reason": reason, "baseline": baseline_report,
                  "stage_a": a_report, "stage_b": best,
                  "comparisons": {"stage_a_to_baseline": dict(eligible=use_a, reason=a_reason),
                                  "stage_b_to_baseline": dict(eligible=use_b, reason=b_reason),
                                  "stage_b_to_selected": dict(eligible=beats_current, reason=current_reason)},
                  "stage_b_start": "stage_a" if use_a else "baseline"}
