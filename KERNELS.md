# BabbleBurn kernel roadmap — what CubeCL needs to match DirectML

Measured on RTX (DX12), MNv4-conv-small-050, 4ch 128x128, batch 64.
Train step (fwd+bwd+AdamW): **DirectML ~52 ms** vs **Burn/wgpu ~210 ms** (4.0x) vs **Burn/CUDA ~537 ms** (10x).
Profiler (`CUBECL_DEBUG_OPTION=profile-medium`): every conv runs as naive `DirectConv2dKernel`
(~900 us/invocation) plus `copy_kernel` layout passes.

## Per-op forward, batch 64 (ms)

| op                        | Burn wgpu | DirectML | ratio |
|---------------------------|-----------|----------|-------|
| stem conv3x3 s2 4-32 @128 | 2.20      | 0.84     | 2.6x  |
| pointwise 1x1 32-96 @32   | 1.59      | 0.38     | 4.2x  |
| depthwise 3x3 g96 @32     | 2.18      | 0.50     | 4.4x  |
| depthwise 5x5 g96 @16     | 1.60      | 0.31     | 5.2x  |
| pointwise 1x1 64-256 @4   | 0.38      | 0.16     | 2.4x  |
| batchnorm 96 @32          | 1.61      | 0.38     | 4.2x  |

Uniform 2.4-5.2x: not one hot op — the whole conv/BN kernel set is naive on wgpu.

## Network inventory (what each kernel class covers)

MNv4-conv-small-050 per forward pass:
- **28 pointwise 1x1 convs** (UIR expand+project x12, 3 CBA 1x1, conv_head) — GEMM-shaped
- **15 depthwise convs** (3x3/5x5, groups = expanded channels 96-288)
- **3 dense 3x3 convs** (stem + 2 downsamples)
- **~45 BatchNorms** (one per conv)
- Training = each of those in fwd + bwd-data + bwd-weight, plus AdamW over ~232 tensors.

Current burn grouped bwd-weight loops over groups in **host code**: one launch per group,
up to 288 launches for a single depthwise layer's weight grad, x15 layers -> thousands of
dispatches per step. This is why the CUDA backend is *slower* than wgpu.

## Must-have kernels (close the 4x)

1. **Depthwise conv2d forward** — dedicated kernel: thread-per-output-tile, filter weights in
   registers (k*k values per channel), channel-tiled mapping, vec4 loads. This is the classic
   MobileNet kernel every fast framework has; CubeCL has none.
2. **Depthwise conv2d backward-data** — same structure with flipped filter/offsets.
3. **Grouped/depthwise backward-weight as ONE kernel** — grid over (group, kh, kw), parallel
   reduction over batch x spatial. Kills the per-group host loop entirely. Benefits all grouped
   convs, biggest single win for training.
4. **Conv1x1 -> batched GEMM lowering + competitive tiled WGSL matmul** — pointwise convs are
   plain GEMMs but still hit DirectConv2dKernel at 4.2x. Reshape NCHW [N,C,H,W] -> [N, C, HW]
   and dispatch matmul directly; matmul itself needs subgroup-op tiling on wgpu (no tensor
   cores in WGSL yet — see references).
5. **Fused BatchNorm training fwd + bwd** — currently composed from primitives (multiple full
   passes over the tensor); 4.2x slow, ~45 instances, fwd+bwd. Single-pass Welford fwd, two-pass
   fused bwd.

Expected result if 1-5 reach DirectML per-op parity: 210 ms -> ~50-60 ms/step on wgpu, i.e.
full-run time at or below the optimized Python trainer, with none of DirectML's CPU-fallback
class of stalls (lerp/EMA is a fused elementwise on Burn).

## Want (secondary wins)

6. **Conv epilogue fusion** (BN-scale+bias + ReLU folded into conv output write) — cuts ~45
   memory round-trips per pass.
7. **Multi-tensor fused AdamW** — one launch for all 232 params instead of ~232 x 4 elementwise
   launches per step.
8. **fp16 storage/compute with fp32 master weights + fp32 BN stats** (wgpu `shader-f16`) — the
   DirectML fp16 blockers (mixed-BN bwd unsupported, BN stats overflow) don't apply because we
   control precision per-op. ~2x memory bandwidth on everything.
9. **NHWC layout support** — profiler shows `copy_kernel` transpose passes around convs;
   NHWC end-to-end removes them and vectorizes channel loads (what DirectML metacommands use).
10. **Winograd 3x3 dense** — low priority, only 3 dense 3x3 convs in the net.

## Training-recipe impact (missing aspects from the quality post-mortem)

The full recipe (aug, curriculum, multi-task + consensus head, EMA teacher, gaze-stutter filter)
adds **no new must-have kernels**:
- Aug (affine/brightness/noise): elementwise/gather — CubeCL fuses these; or stay on CPU
  (data feed measured at 2 ms/batch, not a bottleneck).
- EMA teacher: elementwise lerp — fine on Burn (was a CPU-fallback disaster on DirectML).
- Consensus + MT heads: small linears — covered by the GEMM path (#4).
- Curriculum/BN-freeze: control flow only.
- Losses: elementwise + reduce — fine.
The recipe mildly increases GEMM + elementwise load, reinforcing #4/#5/#7.

## Where the code lives (verified 2026-07, upstream restructured)

Kernels now span three repos:
- **tracel-ai/burn** `crates/burn-cubecl/src/kernel/conv/` — backend glue + the naive `direct.rs`
  (DirectConv2dKernel) + autotune sets + the grouped fallbacks. **This is where a depthwise
  kernel would land**, next to `direct.rs`.
- **tracel-ai/cubek** `crates/cubek-convolution/` — the implicit-GEMM conv library. Has NO
  `groups` field at all — entirely groups==1 by construction. Not where depthwise goes.
- **tracel-ai/cubecl** — the `#[cube]` language + runtimes (wgpu/spirv/cuda/hip/metal/cpu).

Confirmed routing (`burn-cubecl/.../conv/base.rs`): `if options.groups != 1 -> conv_direct`
else implicit GEMM. Backward-weight fallback for grouped literally does
`for g in 0..options.groups { conv_forward_nhwc(...) }` — one full conv launch per group
(a 960ch depthwise = 960 launches). No depthwise kernel and no Winograd exist anywhere in
the stack; **no open PR is adding one.**

Upstream hook: **burn issue #4598** (open, unassigned) — "YOLOX-nano training ~36.5x slower
than PyTorch," repro uses depthwise, reporter measured backward **956 ms vs 17.9 ms**. Same
root cause as ours; a depthwise kernel PR closes it.

wgpu tensor cores: not available on the WGSL path (only via burn-wgpu `vulkan`/SPIR-V
passthrough with cooperative-matrix HW; naga `coop_mat` merged wgpu#8251 Dec-2025 but CubeCL
hasn't wired it). **Irrelevant here** — depthwise is bandwidth-bound and needs no tensor cores.

## Existing implementations to review (all permissive, port sources)

| source | file | license | has | why |
|--------|------|---------|-----|-----|
| **TensorFlow CUDA** | `core/kernels/depthwise_conv_op_gpu.h` | Apache-2.0 | **fwd + dgrad + wgrad**, NHWC+NCHW, `Small` variants (shared-mem tile ≤32x32 = our deep layers, warp-reduced filter grad) | best single source: complete training triple, matches burn's channels-last |
| **tfjs WebGPU** | `tfjs-backend-webgpu/src/depthwise_conv2d_vec4_webgpu.ts` (+`_nchw_shared`, `conv_backprop_depthwise`) | Apache-2.0 | fwd (vec4/NHWC, 4px/thread, input-col reuse, fma) + dgrad | proves the exact WGSL vec4 pattern a `#[cube]` kernel would emit |
| **PyTorch ATen** | `aten/src/ATen/native/cuda/DepthwiseConv2d.cu` | BSD-3 | fwd + bwd templated on `(kSize, stride)`, warp-reduced wgrad | cleanest comptime-specialization model for the `#[cube]` port |
| ncnn | `src/layer/vulkan/shader/convolutiondepthwise*_pack4.comp` | BSD-3 | fwd only (pack4/fp16) | fp16 vec4 GLSL reference |
| TVM topi | `python/tvm/topi/cuda/depthwise_conv2d.py` @v0.18.0 | Apache-2.0 | fwd + **backward schedules** | tiling-parameter design reference |
| MNN | `source/backend/cuda/execution/ConvDepthWiseExecution.cu` | Apache-2.0 | fwd only | secondary |

Primary ports: **TF NHWC(+Small)** and **tfjs vec4 WGSL** for forward; **PyTorch (kSize,stride)
templates** for the backward pair. Autotune: add a `Tunable` in each of the three
`conv/{forward,backward_data,backward_weight}/tune.rs` sets gated on `groups == in_channels`;
`ConvAutotuneKey` already carries `groups`, and today the GEMM candidates error out for grouped
shapes so `conv_direct`/fallback run unopposed — the new kernel just wins the (currently rigged)
race.

Fused BN-training and multi-tensor AdamW: also absent upstream (BN = 2-pass tensor-op
composition in `burn-nn/.../norm/batch.rs`; AdamW = per-tensor op chain in
`burn-optim/.../adamw.rs`). No open PRs. Secondary targets after the depthwise trio.

## Effort estimate

Kernels are written once in the `#[cube]` Rust DSL and JIT to WGSL + SPIR-V + CUDA/HIP/Metal +
CPU, so one implementation covers wgpu-DX12, wgpu-Vulkan, CUDA, and CPU parity. #1-3 (the
depthwise trio) is the canonical, well-referenced kernel with three permissive port sources —
est. 2-4 weeks including autotune registration and parity tests vs ndarray. #4 matmul quality is
CubeCL/cubek-core territory (and partly wants wgpu subgroup-matrix). #5 BN fusion is
straightforward. Upstreaming to tracel-ai/burn (closes #4598) is the right move — it benefits
every MobileNet-class model on Burn and avoids carrying a fork.

Backend note: burn-tch (LibTorch) is the only Burn backend with dedicated depthwise kernels
today, but in **fp32** MobileNet it hits ATen's native `DepthwiseConv2d.cu`, not cuDNN
(cuDNN-depthwise is gated to fp16/specific shapes) — and it needs CUDA+LibTorch, defeating the
no-CUDA goal.
