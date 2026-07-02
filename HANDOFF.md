# BabbleBurn — Depthwise/Grouped Conv Kernel Development: Architectural Requirements & Handoff

**Status:** ready for implementation. All paths, line numbers, versions, and commands below were
verified against the repos cloned into the workspace root on 2026-07-02. Companion doc:
[KERNELS.md](KERNELS.md) (higher-level roadmap + measured per-op numbers).

**Audience:** an engineer (or a fresh session) picking up GPU-kernel work in Rust/CubeCL with no
prior context on this project.

---

## 0. TL;DR — what and why

BabbleBurn is a Rust/Burn rewrite of an eye-tracking model trainer (MobileNetV4-conv-small, a
depthwise-conv-heavy network) whose goal was to escape `torch-directml`'s limitations and train
faster with **no CUDA runtime dependency** (via Burn's `wgpu` backend: DX12/Vulkan).

It works and is numerically correct, but it trains **~4x slower** than the existing
torch-DirectML path. Root cause is fully diagnosed (source + profiler): **CubeCL has no optimized
depthwise/grouped convolution kernel.** Every depthwise layer falls back to a naive direct kernel
(forward), a transpose-based path (backward-data), and — worst — a **host-side per-group loop that
launches one conv per channel** (backward-weight). MobileNetV4-small is ~90% depthwise/pointwise
conv, so this one gap is the entire slowdown.

**The job:** write three specialized `#[cube]` kernels — depthwise **forward**, **backward-data**,
**backward-weight** — register them as autotune candidates, validate correctness against the
ndarray reference, and benchmark them to DirectML parity. Kernels are written once in CubeCL's
`#[cube]` DSL and JIT-compile to WGSL + SPIR-V + CUDA + Metal + CPU, so one implementation fixes
the wgpu **and** CUDA backends. This is also an upstream contribution that closes an open,
unassigned Burn issue ([#4598](https://github.com/tracel-ai/burn/issues/4598)).

**Target result:** train-step (batch 64) from ~210 ms → ≤70 ms (DirectML is ~52 ms) on an RTX
3060, with correctness within existing test tolerances and end-to-end training quality unchanged.

---

## 1. Reference hardware & toolchain (the box this was measured on)

- GPU: **NVIDIA GeForce RTX 3060** (also a Microsoft Remote Display Adapter — ignore it).
- OS: Windows 10 Pro 19045. Shell examples are PowerShell.
- Rust: **rustc/cargo 1.96.0**. Burn `main` declares `rust-version = "1.95"`, edition 2024. No
  `rust-toolchain.toml` in the repo — the installed 1.96 stable is fine.
- **VRAM constraint (important):** this GPU is also used for VR. Do not saturate VRAM during
  profiling; keep batch sizes modest and release between runs.

---

## 2. Repos in the workspace (all cloned, all editable)

Workspace root: `c:/Users/dan2wik/Documents/GitHub/Babal-training/`

| Dir | Upstream | Cloned HEAD | Role |
|-----|----------|-------------|------|
| [burn/](../burn/) | https://github.com/tracel-ai/burn | `dd6fdf3` (v0.22.0-pre.1, `main`) | **Primary edit target.** Backend glue, the naive `direct.rs` kernel, autotune sets, grouped fallbacks, and the whole test+bench harness live here. |
| [cubecl/](../cubecl/) | https://github.com/tracel-ai/cubecl | `ac24631` (`main`) | The `#[cube]` language + runtimes (wgpu/spirv/cuda/…) + examples. Read for DSL patterns. Edit only if a kernel needs a new language/runtime primitive. |
| [cubek/](../cubek/) | https://github.com/tracel-ai/cubek | `07817ed` (`main`) | The implicit-GEMM conv library (groups==1 only — **not** where depthwise goes). Read `ConvolutionParams` as the comptime-params pattern to copy. |
| [kernel_refs/](../kernel_refs/) | — | fetched 2026-07-02 | Downloaded reference depthwise kernels to port from (see §5). |
| [BabbleBurn/spike/onnx_burn/](spike/onnx_burn/) | (this project) | branch `BabbleBurn` | The end-to-end acceptance harness: real MNv4 net, real data, ONNX export, benchmarks. Pins burn **0.21.0** from crates.io. |

> **Version reality check (read before you `cargo build`):** burn `main` is **0.22.0-pre.1** and
> has undergone a "backendless tensor" redesign (`Tensor<const D, K>` — no `B` backend generic),
> removed `NamedMpkFileRecorder`, and changed `Module::save_file/load_file` signatures. The spike
> is written against **0.21.0** and **will not compile against main** without a rewrite; also
> `burn-import` (used in the spike's `build.rs`) is a separate crate family with no main-compatible
> release. See §7 for the integration strategy that resolves this. The **kernel code itself**
> operates on `CubeTensor<R>` (unaffected by the backendless change) — only the spike is affected.

---

## 3. Root cause, precisely (file:line in [burn/](../burn/), HEAD dd6fdf3)

Routing lives in `crates/burn-cubecl/src/kernel/conv/base.rs`. For `groups != 1`:

- **Forward** (`conv_forward_nhwc`, line 82-99): grouped → `conv_direct` (line 88). One kernel
  launch, but it's the naive `direct.rs` kernel. Its vectorization is computed over
  `channels_per_group` (`direct.rs:267-276`), which is **1 for depthwise**, so vec4 loads collapse
  to scalar — the core inefficiency. (~2.2 ms vs DirectML's 0.50 ms for depthwise 3×3.)
- **Backward-data** (`conv_data_backward`, line 167-186): grouped or strided →
  `conv_data_backward_fallback` (`backward_data/fallback.rs:15`), implemented as a
  `conv_transpose2d` (line 45-88). Not a dedicated dgrad kernel.
- **Backward-weight** (`conv_weight_backward`, line 120-144): grouped →
  `conv_weight_backward_fallback` → **`conv_weight_grad_groups`** (`backward_weight/fallback.rs:52`)
  with **`for g in 0..options.groups {` at line 77** — one full `conv_forward_nhwc` launch **per
  group**. A 288-channel depthwise layer = 288 launches for a single layer's weight gradient. This
  is the launch-storm and the single biggest training cost; it's also why the CUDA backend is even
  *slower* than wgpu (537 ms vs 210 ms/step — launch overhead dominates).

The naive `direct_conv2d_kernel` (`direct.rs:33-110`) is generic/N-dimensional despite the "2d"
name; it maps one thread to one (vectorized) output element, NHWC/channels-last, no shared memory,
no register-cached weights. Profiler (`CUBECL_DEBUG_OPTION=profile-medium`) shows it at
**~901 µs/invocation** — the empirical smoking gun.

**No depthwise kernel and no Winograd exist anywhere in burn/cubecl/cubek, and no open PR adds
one.** cubek's conv is implicit-GEMM, groups==1 by construction (`ForwardBlueprint` etc. have no
`groups` field; `cubek-convolution/src/components/error.rs:25` literally rejects groups != 1).

---

## 4. Measured baseline (RTX 3060, batch 64) — the numbers to beat

Train step (fwd+bwd+AdamW), full MNv4-conv-small-050 expr net:

| backend | ms/step | vs DirectML |
|---------|---------|-------------|
| torch-DirectML | ~52 | 1.0x |
| **Burn/wgpu (DX12 or Vulkan — identical)** | **~210** | **4.0x** |
| Burn/CUDA (cubecl-cuda) | ~537 | 10x |

Per-op forward (batch 64), the localization that proves it's the conv/BN kernel set, not one op:

| op | Burn wgpu (ms) | DirectML (ms) | ratio |
|----|----------------|---------------|-------|
| stem conv3x3 s2 4→32 @128 | 2.20 | 0.84 | 2.6x |
| pointwise 1x1 32→96 @32 | 1.59 | 0.38 | 4.2x |
| depthwise 3x3 g96 @32 | 2.18 | 0.50 | 4.4x |
| depthwise 5x5 g96 @16 | 1.60 | 0.31 | 5.2x |
| pointwise 1x1 64→256 @4 | 0.38 | 0.16 | 2.4x |
| batchnorm 96 @32 | 1.61 | 0.38 | 4.2x |

Reproduce: spike benches `cargo run --release --bin bench` / `--bin op_bench` (wgpu) vs
`python spike/bench_directml.py` / `spike/op_bench_directml.py` (see §9).

---

## 5. Reference implementations to port ([kernel_refs/](../kernel_refs/), all permissive)

| file | source | license | has | take |
|------|--------|---------|-----|------|
| `depthwise_conv_op_gpu.h` (78 KB) | TensorFlow CUDA | Apache-2.0 | **fwd + dgrad + wgrad**, NHWC+NCHW, `Small` variants (shared-mem input tile for ≤32×32 = our deep layers) + warp-reduced filter grad | **Primary source.** Complete training triple; NHWC matches burn's channels-last. |
| `depthwise_conv2d_vec4_webgpu.ts` (5.4 KB) | tfjs WebGPU | Apache-2.0 | fwd, **vec4 over channels**, 4 output px/thread, input-column reuse, fma | The exact WGSL vec4-NHWC pattern a `#[cube]` kernel should emit. **Read first for the forward.** |
| `depthwise_conv2d_nchw_shared_webgpu.ts` | tfjs WebGPU | Apache-2.0 | fwd, shared-mem input tile (NCHW) | shared-mem tiling reference |
| `conv_backprop_depthwise_webgpu.ts` | tfjs WebGPU | Apache-2.0 | **dgrad in WGSL** | backward-data reference |
| `depthwise_conv2d_webgpu.ts` | tfjs WebGPU | Apache-2.0 | fwd (general NHWC) | baseline correctness reference |
| `DepthwiseConv2d.cu` (31 KB) | PyTorch ATen | BSD-3 | fwd + bwd templated on `(kernel_size, stride)`, warp-reduced wgrad | cleanest comptime-specialization model for the `#[cube]` port |
| `convolutiondepthwise_pack4.comp` / `.comp` | ncnn Vulkan | BSD-3 | fwd (pack4/fp16 GLSL) | fp16 vec4 GLSL reference |
| `depthwise_conv2d.py` (12 KB) | TVM topi | Apache-2.0 | fwd + **backward schedules** | tiling-parameter design reference |

The winning tricks (present across these) that the naive kernel lacks: **filter weights held in
registers** (compile-time-unrolled kernel loops), **channel-tiled thread mapping**, **vec4 loads
across the channel dim**, **multi-output-per-thread** with input-row reuse, and **warp/plane
reduction** for the weight gradient.

---

## 6. What to implement

Three kernels + autotune registration. Each new kernel file goes in
`burn/crates/burn-cubecl/src/kernel/conv/` and is registered in that dir's `mod.rs`
(`mod depthwise;` + `pub(crate) use depthwise::*;`, mirroring `direct` at `mod.rs:8/19`).

### 6.1 Depthwise forward (`conv_depthwise` → `depthwise_conv2d_kernel`)

- **Applicability:** `groups == in_channels && out_channels % in_channels == 0` (depthwise, incl.
  channel-multiplier). Error fast otherwise (that's how non-applicable candidates are skipped).
- **Layout:** NHWC (burn's conv internals are channels-last; `base.rs` permutes NCHW↔NHWC around
  all conv kernels via `ops/base.rs:183-219`).
- **Core idea (fixes the scalar-collapse bug):** vectorize **across the channel dim**. For
  depthwise, output channel *c* uses input channel *c* and its own filter, so a `Vector<f32, 4>`
  (vec4) load of 4 consecutive channels + 4 corresponding filter taps computes 4 outputs at once —
  exactly what the naive kernel fails to do (it vectorizes over `channels_per_group`=1). This is
  the tfjs `depthwise_conv2d_vec4` pattern.
- **Register-cache filters:** pass `kernel_size`/`stride`/`dilation` as **comptime** params so the
  spatial filter loops fully unroll and taps live in registers. Copy cubek's `ConvolutionParams`
  pattern (a `Copy+Hash` struct passed `#[cube(comptime)]`;
  `cubek-convolution/src/components/config.rs:37`, used in
  `.../global/layout/im2col.rs` with `#[unroll] for i in 0..spatial_dims`).
- **Optional (deep layers):** shared-memory input tile for small spatial sizes (the TF `Small`
  variant), 4 output px/thread with input-column reuse (tfjs).
- **Pattern to copy for host glue:** `direct.rs:223` (`conv_direct`) — output alloc via
  `empty_device_dtype` (`ops/numeric.rs:170`), `into_contiguous_aligned` for channel-contiguity
  (`kernel/contiguous.rs:41`), launch-arg converters on `CubeTensor` (`tensor/base.rs:206+`),
  `calculate_cube_count_elemwise` + `CubeDim::new`, `direct_conv2d_kernel::launch_unchecked`.

### 6.2 Depthwise backward-data (dgrad)

- Gradient w.r.t. input. Structurally a depthwise conv of `out_grad` with the **spatially-flipped
  filter** and swapped padding/stride handling (transposed conv). Reference:
  `conv_backprop_depthwise_webgpu.ts` and TF's `DepthwiseConv2dBackpropInputGPUKernelNHWC`.
- Replaces the `conv_transpose2d` fallback path for grouped/strided
  (`backward_data/fallback.rs:45`).

### 6.3 Depthwise backward-weight (wgrad) — **highest priority**

- Gradient w.r.t. filter weights. This replaces the **per-group host loop**
  (`backward_weight/fallback.rs:77`) — the launch storm — with **one kernel launch**.
- Structure: grid over `(channel, kh, kw)`; each thread/plane reduces `out_grad * input` over the
  batch×spatial domain. Use **plane (warp) reductions** — `plane_sum` is available in CubeCL
  (`cubecl/crates/cubecl-core/src/frontend/plane.rs:213`, gated on the `Plane` runtime feature;
  real use in `cubecl/examples/sum_things/src/lib.rs:29`). References: TF
  `DepthwiseConv2dBackpropFilterGPUKernelNHWCSmall` (warp-reduced) and PyTorch
  `conv_depthwise2d_grad_weight_kernel`.

### 6.4 Register as autotune candidates

Add a `.with(Tunable::new("conv_depthwise", |(input, weight, bias, options)| conv_depthwise::<R,N>(...)))`
to the `TUNER.init(|| TunableSet::new(...)…)` builder chain in each of:

- `burn/crates/burn-cubecl/src/kernel/conv/forward/tune.rs` (`conv_autotune`, chain lines 26-74; add after `"conv_direct"` at line 31).
- `burn/crates/burn-cubecl/src/kernel/conv/backward_weight/tune.rs` (`wgrad_autotune`, chain lines 30-74; after `"wgrad_fallback"` at line 37).
- `burn/crates/burn-cubecl/src/kernel/conv/backward_data/tune.rs` (`dgrad_autotune`, chain lines 33-71; after the fallback at line 40).

The candidate fn must return `Result<CubeTensor<R>, ConvSetupError>` and error fast on
non-depthwise keys. **No key changes needed:** `ConvAutotuneKey` (`tune_key.rs:5-27`) already
carries `groups`, `in_channels`, `out_channels`, `kernel_size`, `stride`, etc., so depthwise
shapes autotune under a distinct key. Today the GEMM candidates error out for grouped shapes, so
only `conv_direct`/fallback compete — the new kernel just has to beat the naive one.

### 6.5 CubeCL `#[cube]` DSL primer (cloned cubecl HEAD `ac24631`)

- **First reads:** `cubecl/examples/gelu/src/lib.rs` (47 lines: complete kernel + launch, vec4,
  `comptime!`), then `cubecl/examples/sum_things/src/lib.rs` (comptime `Option` params, `#[unroll]`,
  `plane_sum`).
- **API notes (this HEAD; differs from older docs):** vectorization type is **`Vector<P, N: Size>`**
  (not `Line<T>`); lane count is the `N` type param, `vector_size` arg on generated launch fns;
  `.vector_size()` replaces `.line_size()`. Comptime optionals are `ComptimeOption<T>` or plain
  `#[comptime] x: Option<T>`. (`Line`/`CubeOption` names in old blog posts no longer exist.)
- `#[cube(launch_unchecked)]` for the kernel fn; `#[cube]` for device-side helpers; `comptime!(...)`
  for host-evaluated constants baked into the kernel.
- **Version-pinning trap:** cubek `main` pins cubecl rev `0b6bcc33`; burn `main` pins cubecl
  `3beb9afa` + cubek `3ec66d7` (`burn/Cargo.toml:226-229`). The cloned cubecl/cubek HEADs are
  *newer* than what burn builds against. **Write kernels against the revs burn pins**, or use the
  commented-out path overrides in `burn/Cargo.toml:231-239`. If you only touch
  `burn-cubecl` (the DSL kernel + glue), you build against burn's pinned cubecl automatically and
  avoid the skew entirely — prefer that.

---

## 7. Development environment / Cargo integration

There are two validation surfaces and they want different burn versions. Do both, in this order:

### 7.1 Kernel dev + unit validation → burn `main` (primary)

Develop the kernels in the cloned [burn/](../burn/) `main` checkout and validate with burn's own
`crates/burn-backend-tests` harness (§8, §9). Why main: (a) it already has depthwise benches and
grouped-conv correctness+autodiff tests, (b) its cubecl API matches the DSL examples and
`kernel_refs`, (c) it's the upstream PR target (closes #4598). No spike needed for kernel-level
correctness or per-op speed. Just build/test/bench inside the burn checkout.

### 7.2 End-to-end acceptance → spike on a v0.21.0 branch (secondary)

The spike (real MNv4 training on `user_cal.bin`, ONNX export, DirectML head-to-head, quality vs the
goalpost model) pins burn **0.21.0** and can't move to main (backendless-tensor rewrite +
`burn-import` has no main release). To run the kernel end-to-end there, backport it onto a branch
off the `v0.21.0` tag and patch the spike onto that branch:

```powershell
# in the burn checkout (v0.21.0 tag already fetched shallow)
git -C c:/Users/dan2wik/Documents/GitHub/Babal-training/burn switch -c kernels-0.21 v0.21.0
```

Append to `BabbleBurn/spike/onnx_burn/Cargo.toml` (it is its own workspace root — the patch belongs
here). This is the **complete** patch set:

```toml
[patch.crates-io]
burn = { path = "../../../burn/crates/burn" }
burn-store = { path = "../../../burn/crates/burn-store" }
```

Why sufficient: `burn` and `burn-store` are the only crates.io burn crates the spike names; once
patched, `crates/burn`'s workspace **path** deps pull in the whole local tree (`burn-cubecl`,
`burn-wgpu`, `burn-cuda`, `burn-ndarray`) so kernel edits flow through automatically. The patch
applies graph-wide including the `burn-import → burn-onnx → burn` build-dep subgraph, so no version
skew. `burn-import` stays a registry dep (it only emits generated source). **Do not** bump the
`kernels-0.21` workspace version — the patch stops matching `^0.21.0` the moment it isn't `0.21.x`.

> `[patch.crates-io]` beats path-deps-on-main because (a) main's APIs are incompatible with the
> spike source and (b) `burn-import` output can't compile against main at all. Reverting is
> deleting two lines.

**Caveat:** the v0.21.0 tag pins cubecl **0.10.0** / cubek **0.2.0** (crates.io), whose DSL is the
older `Line`/`line_size` API. So the kernel body may need a small API translation between the
main version (`Vector`) and the 0.21 backport (`Line`). The kernel *logic* is identical; budget a
few hours for the rename pass. If kernels need edits inside cubecl itself (not just burn-cubecl),
add a cubecl-0.10 checkout + `[patch.crates-io]` for `cubecl`/`cubecl-common`/`cubek` too.

**Upstreaming flow (endgame):** iterate on `kernels-0.21` for fast end-to-end feedback, then
cherry-pick the kernel commits onto `main`, adapt the glue to the backendless architecture, and
open the PR against tracel-ai/burn (reference #4598).

---

## 8. How to test (correctness)

All tests live in `burn/crates/burn-backend-tests/` (the old `testgen` layout is gone on main).
Backend is chosen by cargo feature; crate-local aliases are in
`burn/crates/burn-backend-tests/.cargo/config.toml`. **Run from inside that crate dir.**

**Nothing below has been executed** — commands are derived from the verified alias/feature graph.
First build recompiles the local burn tree for the selected features (mind VRAM).

```powershell
cd c:\Users\dan2wik\Documents\GitHub\Babal-training\burn\crates\burn-backend-tests

# forward correctness (fixed expected values), wgpu:
cargo test-wgpu --test tensor conv2d
# kernel-vs-ndarray-reference (random inputs, the key correctness gate for a new kernel):
cargo test-wgpu --test cubecl conv2d
# backward / gradients (autodiff vs hard-coded expected grads, incl. grouped):
cargo test-wgpu --test autodiff conv2d
# same on the SPIR-V/Vulkan path:
cargo test-vulkan --test cubecl conv2d
```

Aliases: `test-wgpu = "test --release --no-default-features --features wgpu,std,fusion"`;
`test-wgpu-nofuse` drops fusion; `test-vulkan` uses `features vulkan,std,fusion`.

**Existing grouped/depthwise coverage to extend** (add depthwise cases mirroring these):
- Forward: `tests/tensor/float/module/conv2d.rs` — `test_conv2d_groups` (cin=cout=groups=2, i.e.
  depthwise), `test_conv2d_groups_multiple_channels`.
- Kernel-vs-reference: `tests/cubecl/conv2d.rs` — `conv2d_should_match_reference_backend`
  (groups=2), `conv2d_weight_backward_should_run` (wgrad vs ndarray).
- Autodiff: `tests/autodiff/conv2d.rs` — `test_conv2d_groups`, **`test_conv2d_groups_stride_2`**
  (cin=cout=groups=4 = depthwise). Compares x/weight/bias grads to precomputed literals at
  `Tolerance::rel_abs(0.01, 0.01)`; each runs twice (plain + gradient-checkpointing).

**Add** true MobileNet depthwise shapes (e.g. 96→96 groups=96 3×3 @32, and 96→96 groups=96 5×5
@16) to `tests/cubecl/conv2d.rs` (fwd + wgrad + dgrad vs ndarray) — this is the primary correctness
gate before benchmarking. The ndarray backend is the ground-truth reference
(`ReferenceDevice = NdArrayDevice::Cpu`).

Acceptance: all three (fwd/dgrad/wgrad) match the ndarray reference within the existing
`rel_abs(0.01, 0.01)` tolerance on depthwise shapes, on both `wgpu` and `vulkan`, with and without
fusion.

---

## 9. How to benchmark (speed)

### 9.1 In-repo per-op (divan) — the kernel-level result

`burn/crates/burn-backend-tests/benches/conv_ops.rs` already has a **`conv2d_depthwise`** group with
MobileNet/ConvNeXt shapes (`depthwise_3x3_4x32x56x56`, `depthwise_7x7_4x24x56x56`,
`depthwise_3x3_stride2_4x64x56x56`, …). Compare the new kernel against the naive baseline:

```powershell
cd c:\Users\dan2wik\Documents\GitHub\Babal-training\burn\crates\burn-backend-tests
cargo bench-wgpu --bench conv_ops -- conv2d_depthwise      # after kernel is registered
cargo bench-ndarray --bench conv_ops -- conv2d_depthwise   # CPU reference (optional)
```

`bench-wgpu = "bench --no-default-features --features wgpu,std,fusion"`. `bench_synced` inserts a
device sync inside the timed region (accurate GPU timing). To A/B the kernel vs the naive path,
temporarily force the strategy or gate the new candidate off, or diff two runs. The stale
`--features simd,gemm` in the file's doc-comment is wrong (not features of this crate).

### 9.2 CubeCL profiling (confirm the right kernel is chosen + its cost)

```powershell
$env:CUBECL_DEBUG_LOG    = "stdout"          # on Windows use stdout or an explicit file path
$env:CUBECL_DEBUG_OPTION = "profile-medium"  # or profile-full
$env:CUBECL_AUTOTUNE_LEVEL = "full"          # exhaustive kernel search (minimal|balanced|extensive|full)
# delete the autotune cache to force re-tuning (default cache = <first-parent-with-Cargo.toml>/target;
# fallback %LOCALAPPDATA%\cubecl):
cargo bench-wgpu --bench conv_ops -- conv2d_depthwise
```

Baseline showed `DirectConv2dKernel | ~901µs`; success = the new `depthwise` kernel is selected and
its per-invocation time drops toward DirectML's ~0.3-0.5 ms/layer. (Vars are parsed by cubecl:
`cubecl/crates/cubecl-runtime/src/config/base.rs:58,93,118`. A `cubecl.toml`/`burn.toml` `[cubecl]`
section also works.)

### 9.3 End-to-end (spike, after §7.2 backport) — the acceptance result

```powershell
cd c:\Users\dan2wik\Documents\GitHub\Babal-training\BabbleBurn\spike\onnx_burn
cargo run --release --bin bench        # MNv4 train step, wgpu, batch sweep
cargo run --release --bin op_bench     # per-op, wgpu
# DirectML side-by-side (external — DirectML is not a Burn backend):
python ..\bench_directml.py
python ..\op_bench_directml.py
# full training + ONNX for quality check:
#   run the fit_expr_fast / fit_gaze_fast entry, then bridge_export.py to ONNX
```

---

## 10. Results we're after (acceptance criteria)

On the RTX 3060, batch 64:

1. **Correctness (gate):** depthwise fwd + dgrad + wgrad match the ndarray reference within
   `rel_abs(0.01, 0.01)` on `wgpu` and `vulkan`, fusion on and off. All existing conv tests still
   pass.
2. **Per-op parity (primary):** depthwise 3×3/5×5, pointwise 1×1, within **~1.3x of DirectML**
   (from today's 4.2-5.2x). i.e. depthwise 3×3 @32 ≤ ~0.65 ms (from 2.18), depthwise 5×5 @16 ≤
   ~0.40 ms (from 1.60).
3. **Train step (headline):** MNv4 expr net fwd+bwd+AdamW **≤ 70 ms/step** (from ~210), targeting
   DirectML's ~52 ms. The wgrad kernel (killing the per-group launch storm) is the biggest lever;
   expect the CUDA backend to improve dramatically too.
4. **End-to-end (acceptance):** full training on `user_cal.bin` completes **faster than the
   optimized Python/DirectML trainer** (~7.6 min baseline), and the exported ONNX matches the
   goalpost model's quality — expression activation and gaze-during-expression on par with
   `tuned_temporal_eye_tracking_20260701_092449.onnx`. (Remember the gaze-stutter data filter and
   the full training recipe are separate concerns tracked elsewhere; the kernel work must not
   regress them.)
5. **Upstream:** kernels cherry-picked to `main`, PR opened against tracel-ai/burn referencing
   #4598, correctness + `conv2d_depthwise` bench improvement demonstrated in-repo.

---

## 11. Secondary work (after the depthwise trio; also absent upstream)

Not required to hit the headline number, but the next wins (details in [KERNELS.md](KERNELS.md)):

- **Fused BatchNorm train fwd+bwd** — today it's a 2-pass tensor-op composition
  (`burn/crates/burn-nn/src/modules/norm/batch.rs`, `forward_train`); ~45 instances at 4.2x.
  Single-pass Welford fwd + fused bwd.
- **1×1-conv → batched GEMM lowering + a competitive tiled WGSL matmul** — pointwise convs are
  GEMMs but still hit the direct kernel at 4.2x. (No tensor cores on the WGSL path — only via the
  `vulkan`/SPIR-V cooperative-matrix route; naga `coop_mat` merged wgpu#8251 Dec-2025 but unused by
  CubeCL. Depthwise needs no tensor cores; pointwise-GEMM would benefit but that's cubek/core
  territory.)
- **Multi-tensor fused AdamW** — today per-parameter op chains (`burn/crates/burn-optim/src/optim/adamw.rs`),
  ~232 tensors × ~10 ops/step. No foreach/multi-tensor step exists.
- **Conv+BN+ReLU epilogue fusion**, **fp16 storage w/ fp32 master + fp32 BN stats** (wgpu
  `shader-f16` — the DirectML fp16 blockers don't apply since precision is per-op here), **NHWC
  end-to-end** (removes the `copy_kernel` transpose passes the profiler showed).

---

## 12. Open questions / risks (verified caveats)

- **Nothing here was compiled.** All cargo commands are derived from verified aliases + feature
  graphs + file layout; the first real build is the first correctness check.
- **cubecl API skew:** cloned cubecl/cubek HEADs are newer than the revs burn pins. Build kernels
  against burn's pinned revs (or path-override) to avoid `Vector`/`Line`-class drift. See §6.5.
- **0.21 vs main DSL:** the spike backport (§7.2) is on cubecl 0.10 (`Line` API); main kernel is on
  the newer `Vector` API. Expect a mechanical rename pass between the two, not a logic rewrite.
- **`plane_sum` gating:** plane ops require the `Plane` runtime feature; the wgrad kernel should
  fall back gracefully (or the candidate errors out) where planes are unavailable.
- **Shallow clones:** all three repos are `--depth 1`. `git fetch --unshallow` (or deepen) if you
  need history, e.g. to base `kernels-0.21` with full ancestry.
- **VRAM:** shared with VR use — keep benchmark batch sizes modest, release between runs.
- **`burn-import`** lives in a separate repo/crate family (not in the burn checkout); it can't be
  path-depped and has no main-compatible release. This is why the spike stays on 0.21.

---

## 13. Companion artifacts

- [KERNELS.md](KERNELS.md) — roadmap, full per-op table, upstream survey, effort estimate.
- [PLAN.md](PLAN.md) — the original 7-phase BabbleBurn roadmap.
- [kernel_refs/](../kernel_refs/) — the downloaded reference kernels (§5).
- Spike source: [spike/onnx_burn/src/](spike/onnx_burn/src/) — `mnv4.rs` (the net), `bench.rs`,
  `op_bench.rs`, `bench_cuda.rs` (benchmarks), `data.rs`/`fast.rs` (data pipeline).
- Goalpost model: `../tuned_temporal_eye_tracking_20260701_092449.onnx`.
