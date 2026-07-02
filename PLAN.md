# BabbleBurn — Rust/Burn rewrite of BabbleTrainer

Escape torch-directml's CPU-dispatch wall (CubeCL kernel fusion), keep the no-CUDA portable
download (wgpu / DX12 / Vulkan), stay actively maintained. Backend: `wgpu` for train+infer;
`ndarray` for CPU parity tests.

**Quality goalpost = the ORIGINAL ~15-min stock train, not the optimized 7.6-min one.**
Stock 4-seed held-out means (eval_model.py on user_cal.bin):
expr r EyeLid 0.9955 / Widen 0.9682 / Squint 0.9904 / Brow 0.9683, expr MSE ~0.0072, gaze MAE 17.40°.

---

## Phase 0 — De-risking spike (do before committing to the full rewrite)

- [~] **burn-import MNv4-conv-small, parity vs ONNX Runtime < 1e-4** (in progress). ONNX op set is
      a best case: only Conv (46, BN folded), Relu, Add, GlobalAveragePool, Flatten, Gemm — all
      burn-import-supported. Artifacts in `spike/` (backbone.onnx, input_ref/ort_out/feat_ref .npy).
- [ ] **wgpu smoke**: build+run the same model on the wgpu backend (DX12/Vulkan) — not just ndarray —
      to confirm the GPU path works on this box.
- [ ] **DEPLOYMENT DECISION (the blocker)**: Burn has NO ONNX export. Pick the ship path and prove it:
      - v1 dev/validation: Burn-train → port weights to PyTorch → existing `export_dual_onnx.py`.
      - ship (on-device, no torch): hand-rolled Rust ONNX graph builder for the FIXED MergedDualEye
        graph, verified in-process with the `ort`/`tract` crates against the < 1e-4 check.
      - end-state: Babble runtime embeds Burn-native inference, drop ONNX (out-of-repo).
      Prove a weight round-trip produces a bit-compatible opset-18 ONNX.
- [ ] Confirm on-disk JPEG dims in user_cal.bin: if already 128×128, INTER_AREA never fires and that
      resize-parity risk vanishes.

## Phase 1 — Foundations
- [ ] Cargo workspace: crates `bin-io`, `data`, `model`, `train`, `eval`, `export`.
- [x] **Backbone (hand-written, trainable) — DONE & parity-verified** (spike/onnx_burn/src/mnv4.rs).
      Transcribed MNv4-conv-small (5 ConvBnAct + 12 UIR, 0.5 width), loaded timm weights via
      SafetensorsStore + PyTorchToBurnAdapter + explicit BN weight/bias->gamma/beta remap. Block-by-block
      parity vs timm PASS: logits 6.1e-5, worst stage 5.9e-4. Key facts learned:
        - HEAD is MobileNetV3-style: global_pool -> conv_head(1x1 as FC) -> norm_head(relu) -> classifier
          (NOT conv_head-before-pool). conv_head output is [B,1280,1,1].
        - safetensors carries no container-type, so the adapter can't auto-rename BN weight/bias;
          needs explicit remap `(bn1|bn|norm_head)\.weight$ -> $1.gamma` (and .bias -> .beta).
        - param paths mirror timm exactly => BurnToPyTorchAdapter gives the inverse map (interim export).
        - forward_features returns the 1280-d pre-logits (Mean-Teacher / ConsensusHead input).
- [ ] Graduate the backbone from spike/onnx_burn into the real workspace `model` crate.
- [ ] Eval harness + fixed eval set producing the goalpost metrics (or reuse Python eval on exported ONNX).
- [ ] **Convert the shipped pretrain ASSETS to Burn records** (one-time): the Babble-specific 4-ch expr
      pretrain `model_best.pt`, the gaze `gaze_model_best.pt`, and `footage_unlabeled.npz`. The frontend
      warm-starts from these (not timm) when present; needs the same timm->Burn key-map inverted.

## Phase 2 — Data pipeline
- [ ] `.bin` parser: byteorder record decode, flag bitmasks (GOOD_DATA / EXPR_UNLABELED), label extract,
      is_blank (std ddof=0 on pre-resize pixels < 2.0), temporal 4-frame stack + zero-pad + no-cross-bin
      predecessor guard, index build (rayon), serialize (bincode/ndarray-npy — rebuild fresh, don't reuse .npz).
- [ ] **Preprocessing (same code train + inference to kill train/serve skew)**: JPEG decode, resize,
      equalizeHist — hand-match cv2's cdf_min formula (imageproc's differs algorithmically), canonical flip.
- [ ] Augmentation (EXPR): geom warp (Border::Replicate + Bilinear), photometric (bright/contrast/gamma/blur,
      cv2 ksize→sigma `0.3*((k-1)*0.5-1)+0.8`), mixup (neutral↔expr blend + label blend), weak/strong.
- [ ] Augmentation (GAZE): the SEPARATE gaze aug path (`_aug_stack_gaze`): different spatial/intensity/blur
      probabilities + `_occlude_lids` occlusion (GAZE_OCCLUDE_PROB, default 0 but code exists). Distinct
      from the expr aug — don't conflate.
- [ ] Datasets: main (labeled + mixup), paired-eye (canon/mirror order MUST match MergedDualEye flip),
      unlabeled footage (cur_strong/cur_weak/nbr_weak + pair_ok temporal neighbor).
- [ ] **Weighted sampler** (no WeightedRandomSampler in Burn): ~30-line WeightedIndex wrapper, with
      replacement, `.take(steps*batch)`; gaze 3-way stratified (closed/expr-gaze/neut-gaze, empty-bucket
      renorm, normalize-by-count); 3 concurrent stream iterators.

## Phase 3 — Model + losses
- [ ] DualTaskTower, MicroChad gaze (GlobalMaxPool = two max_dim reductions), ConsensusHead
      (fc-relu-fc-sigmoid on detached L/R feats), MergedDualEye forward (flip / 1−x gaze-X mirror /
      side order / fuse / gather_idx — integer logic ports verbatim).
- [ ] EMA teacher on the INNER (non-autodiff) backend (`model.valid()`); lerp params, COPY BN buffers,
      skip integer counters. (Drop EMA_EVERY striding + fp16 twin — DirectML cruft.)
- [ ] 7 losses: deconfuse-weighted MSE, range hinge (keep clean `(p-p.clamp(0,1))²`), agree-gated
      evenness, MT consistency w/ quadratic rampup, temporal margin hinge, consensus MSE, huber/l1 option.
- [ ] Optim: stock burn AdamW (decoupled WD); per-group LR via `GradientsParams::from_module` per
      submodule + `step(lr, grads)`; warmup+cosine; backbone freeze = build head-only grads (also skips WD).

## Phase 4 — Training loops
- [ ] fit_expr two-stage curriculum (A: supervised+mixup+even+frozen-backbone warmup then unfreeze;
      B: +Mean-Teacher+temporal+consensus). Clean separate per-stream forwards (drop mega-batch concat).
- [ ] fit_gaze supervised-only loop (the MT/temporal path is dead — don't port).
- [ ] Validation + best-ckpt on the EMA TEACHER (BN inference mode); Pearson-r (std-guard) + MAE-deg.
- [ ] Checkpoint save/load (NamedMpk record); warm-start contract (stable key layout).

## Phase 5 — Export + frontend
- [ ] Deployment export per the Phase-0 decision → opset-18 ONNX matching the current interface,
      metadata_props, blendshape names, fusion/gather, passing the < 1e-4 ORT check.
- [ ] Frontend CLI (clap): warm-start discovery, footage discovery, fit_expr→fit_gaze→export sequence,
      VRAM/worker caps for SteamVR + VR game, the batch→schedule rescale invariant (sqrt-LR, EMA-tau-in-
      samples, proportional steps).

## Phase 6 — Validate vs goalpost
- [ ] End-to-end Burn train on user_cal.bin → export → eval; compare per-channel r / MSE / gaze MAE to
      the stock goalpost above.
- [ ] Convergence re-tuning (LR/schedule) if numerics shift moved it.
- [ ] Perf: wall-clock vs 15-min stock and 7.6-min optimized; per-phase VRAM.

## Phase 7 — Packaging & integration (ship it — this is the payoff)
- [ ] **Release build + distribution size**: `cargo build --release` (LTO/opt profile), measure the final
      binary + wgpu vs the ~450 MB torch-directml stack. This is the whole point — must actually be small.
- [ ] Bundle the pretrain assets (Burn records from Phase 1) beside the binary; frozen-path discovery via
      `std::env::current_exe()` (replaces the PyInstaller app_dir logic).
- [ ] **GUI integration contract**: reproduce the stdout progress protocol the Babble desktop app parses
      — `=== Epoch 1/1 ===`, `Batch <n>/<total>, Loss: <f>`, `Training completed successfully!`,
      `Model exported to ONNX: <path>` — plus exit codes and the `<bin> <onnx>` CLI arg shape.
- [ ] **wgpu hardware/vendor floor**: DirectML ran on any DX12 GPU; wgpu needs DX12/Vulkan. Validate on
      older Intel/AMD iGPUs in the user base; document the minimum, and the CPU/adapter fallback behavior.
- [ ] VRAM-failure UX: graceful message + the auto-scale path (parity with the current trainer's guidance).

## Cross-cutting / open questions
- [ ] **Parity-test discipline (do throughout, not just Phase 6)**: loss-by-loss unit tests on tiny fixed
      inputs vs a PyTorch reference; dataset byte-parity on the deterministic path; the block-by-block
      backbone check (already passing) is the template. Catches drift early instead of at end-to-end.
- [ ] wgpu training stability at ~1M-param scale; keep f16 OFF for v1 (burn#4949 autotune NaN); f32 first.
- [ ] Determinism: accept non-bit-identical RNG vs Python (per-user retrain absorbs it).
- [ ] CubeCL autotune warmup once up front so JIT compile doesn't spike CPU/GPU mid-train near VR.
- [ ] Torchvision MNv2 fallback: don't port (dev-only; every real entrypoint forces mnv4).

## Intentionally DROPPED (perf features not needed for a correct stack)
- Decoded-frame mmap disk cache (BB_FRAME_CACHE): Rust's true multithreaded loading removes the GIL/spawn
  bottleneck it existed to dodge; an optional in-RAM cache suffices. Revisit only if loading is the floor.
- All DirectML workarounds (ForeachAdamW, MMP, fp16 twin, EMA/LOSS striding, mega-batch concat, uint8-H2D,
  adapter benchmark) — see the transfer review drop-list; they evaporate under Burn/CubeCL.
