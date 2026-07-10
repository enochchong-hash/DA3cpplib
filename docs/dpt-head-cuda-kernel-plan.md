# Hand-Written CUDA Kernels for the DPT Head

*Status: **IMPLEMENTED** (see §8 for as-built results — warm server total 52.5 → 38.8 ms, ~26 fps). §§1–7 are the original plan, kept for context; the implementation deviated from it in one important way: profiling revealed the biggest head cost was ggml's naive **transposed-conv** kernel, not im2col conv traffic.*

## 1. Why the head, and what's on the table

Warm forward on the RTX 5060 (q8 base @504×336, `DA_FUSED=0 DA_PROFILE=1`):

| Segment | Time | Character |
|---|---|---|
| ViT-B backbone (12 blocks) | ~24 ms | Large matmuls + flash attention — cuBLAS-class, **no headroom** for hand-written kernels |
| DPT head | ~35 ms | Many *small* 3×3 convs + upsamples on [H/14..H/2]-scale feature maps — **the target** |

The head's cost profile is the classic im2col weakness: `ggml_conv_2d` materializes an im2col tensor (K²·C_in × H·W) in global memory, then runs a GEMM. For the head's shapes (C=256, small spatial dims, K=3) the im2col write+read traffic rivals the FLOPs. A fused implicit-GEMM kernel eliminates that intermediate entirely.

**Realistic win: 10–15 ms of the ~35 ms head → warm forward ~48 ms → ~35–38 ms; server total from ~56 ms → low-40s.** (~25 fps sequential, closer to the real-time target.)

## 2. What exactly the head computes

Per `dpt_head.cpp` / `dpt_blocks.cpp` (ggml graph, all F32 activations):

1. **Reassemble stage** ×4: per out-layer feature `[2·768, N_patch]` → 1×1 conv (projection) → transposed-conv / identity / strided-conv resample to 4 pyramid scales.
2. **Fusion stage** ×4 (RefineNet-style): each `FeatureFusionBlock` = 2 × `ResidualConvUnit` (each: ReLU → 3×3 conv → ReLU → 3×3 conv → add) + upsample (`ggml_interpolate`, align-corners bilinear) + 3×3 conv.
3. **Output stage**: 3×3 conv (256→128) → 2× upsample → 3×3 conv (128→32) → ReLU → 1×1 conv (32→output_dim) → (depth = exp, conf = exp+1 on host).

Roughly 14 3×3 convs, 6 1×1 convs, 5 upsamples. The 3×3 convs at the two largest scales (H/4: 126×84, H/2: 252×168, C=256/128) dominate.

## 3. Kernel design

### K1: fused ReLU→conv3×3(→ReLU→conv3×3→add) — the ResidualConvUnit
- Implicit GEMM, output-tile-per-block: each block computes a `TM×TN` tile of (C_out × H·W).
- Load the 3×3×C_in weight slice into shared memory once per block; stream input tiles (with 1-px halo) through shared memory.
- Fuse the *leading* ReLU into the load, the *trailing* add into the epilogue. Two kernel launches per RCU (one per conv) rather than one giant fusion — keeps register pressure sane, still kills all im2col traffic.
- F32 accumulate; optionally TF32 tensor-core `mma` for the inner product (Blackwell TF32 ≈ 2× F32 FMA) — validate parity ≤1e-3 first.

### K2: fused bilinear-upsample→conv3×3
- The fusion block's `interpolate(×2, align_corners) → conv3×3` pair reads/writes a full-size intermediate. Computing the upsample *inside* the conv's input fetch (each tap is a 4-point lerp) removes one full feature-map round-trip at the two biggest scales.

### K3 (cheap, do first): batched 1×1 convs
- The six 1×1 convs are pure GEMMs; ggml already does them as matmuls. Just verify they hit the MMQ/cuBLAS path and skip custom work here (likely nothing to gain — this is the null-hypothesis check).

## 4. Integration path (mirrors the Winograd precedent)

The codebase already has exactly this pattern for CPU: `winograd_conv3x3()` returns a `ggml_custom_4d` node and `dpt_blocks.cpp:conv3x3()` picks an implementation per `DA_CONV` env / `gpu_mode()`. Plan:

1. Add `cuda_conv3x3_fused()` in a new `src/cuda_head_kernels.cu` guarded by `DA_GGML_CUDA`, registered as a ggml **custom op** whose compute callback launches our kernel on the ggml CUDA stream (`ggml_backend_cuda_context::stream()`); tensors arrive as device pointers since the graph runs on the CUDA backend. (ggml custom ops run on the *CPU* backend by default — the callback must be registered via the CUDA backend's custom-op hook, or the node built as a `ggml_map_custom` variant that ggml-cuda's `supports_op` accepts. **Spike this first** — it's the main integration risk; fallback is bypassing ggml for the head entirely, see §6.)
2. Gate with `DA_CONV=cuda_fused` (env, default off) so A/B against im2col is one env var, same as the existing `winograd|direct|im2col` switches.
3. Parity gate: extend the ctest suite comparing head output vs im2col path, tolerance 1e-4 relative (F32) / 1e-3 (TF32).

## 5. Validation & measurement protocol

- Correctness: per-kernel unit tests against `ggml_conv_2d` output on random tensors of the head's exact shapes; end-to-end depth-map diff vs baseline (≤1 u8 level mean, as done for GPU preprocessing).
- Perf: `DA_PROFILE=1` head-segment timing + 20-request server median, per kernel enabled — keep a table in `performance-optimization.md`; kill any kernel that doesn't beat im2col by ≥2 ms.
- Occupancy sanity: `nsys`/`ncu` if available; else timing-only.

## 6. Risks / fallbacks

| Risk | Mitigation |
|---|---|
| ggml custom-op-on-CUDA plumbing not supported in vendored ggml | Fallback: run the whole head *outside* ggml — the engine change is small since `build_feats_graph` already exposes the 4 feat tensors; a standalone CUDA head reading them via device pointers avoids ggml entirely (bigger effort, full control) |
| TF32 accuracy drift | Ship F32 kernels first; TF32 as opt-in env |
| Upstream divergence (mudler/depth-anything.cpp moves) | Keep everything behind `DA_CONV=cuda_fused` + one new file; rebase cost stays small |
| Effort vs payoff | Decide after K3/spike: if the head segment can't realistically drop below ~25 ms, stop at the current 56 ms pipeline |

## 7. Effort estimate

| Step | Estimate |
|---|---|
| Spike: custom-op-on-CUDA-stream plumbing (§4.1) | 0.5–1 day |
| K1 implicit-GEMM conv3×3 + parity tests | 2–3 days |
| K2 fused upsample-conv | 1–2 days |
| TF32 variant + tuning (optional) | 1–2 days |

Total: roughly one engineering week for the ~15 ms. Recommended trigger: needing sustained >20 fps (video), which the current 56 ms (~18 fps) almost reaches.

---

## 8. As built (implemented)

### What shipped

| Piece | File | Notes |
|---|---|---|
| ggml-cuda custom-op extension | `third_party/ggml/src/ggml-cuda/ggml-cuda.cu` (local patch) | `GGML_OP_CUSTOM` nodes whose userdata carries a magic-tagged launch descriptor run on the CUDA backend (`supports_op` + dispatch case); everything else still goes to CPU |
| Fused conv3×3 implicit GEMM | `src/cuda_head_kernels.cu` | BM×BN×BK = 64×64×8 tiled SGEMM with virtual-im2col B-gather; fused pre-ReLU / bias / residual epilogue; BM=32 variant for OC≤32 (out2a). Covers all 16 head 3×3 convs incl. fully-fused RCUs (2 launches instead of 5 ops) |
| Pixel-shuffle transposed conv | `src/cuda_head_kernels.cu` | For stride==kernel (resize.0 4×4 s4, resize.1 2×2 s2) transposed conv is a plain GEMM `C[OC·s²][P_in]` + shuffle epilogue — replaces ggml's one-thread-per-output kernel that loops all IC·K² taps with divergent stride checks |
| Integration | `src/dpt_blocks.cpp` (`conv2d`, `conv_transpose2d_p0`, `residual_conv_unit`) | Gated by `DA_CONV=cuda_fused` (set by default in `scripts/start.sh`; `DA_CONV=im2col` reverts). Non-qualifying shapes fall through to stock ggml paths |
| Diagnostics | `DA_CUDA_HEAD_STATS=1` | Per-shape CUDA-event timing of every fused kernel, dumped at exit |

### What profiling actually found (differs from the plan)

Per-shape stats showed the 3×3 convs were only **~11.4 ms** of the ~35 ms head; the plan's im2col-traffic theory explained a third of the cost. The dominant hidden cost was ggml's **conv2d-transpose** CUDA kernel in the reassemble stage (~12.5 ms for two layers whose FLOPs are trivial). Because both resize layers have stride == kernel size, each output pixel receives exactly one tap — the pixel-shuffle GEMM kernel runs them in **0.25 ms** (~50× faster).

### Measured results (q8 base @504×336, RTX 5060)

| Metric | Before | After |
|---|---|---|
| Head segment (unfused profile) | 35.9 ms | **~22 ms** |
| CLI fused-graph infer/iter | 57.8 ms | **42.0 ms** |
| Warm server total (20-req median) | 52.5 ms | **38.8 ms (~26 fps)** |
| Parity vs im2col (relative depth error) | — | max 1.2e-3, mean 5.7e-5 (float summation order; visually identical) |

Also measured: `DA_CONV=direct` (ggml's direct-conv CUDA kernel) is 6× *slower* than im2col on this GPU — the plan's K3 null-hypothesis check, confirmed.

### Remaining headroom — updated after round two (§9)

Superseded in part: TF32 (shallow wmma form) and upsample-conv fusion have since been tried — see §9 for verdicts. What remains genuinely untried, with rough sizes:

1. **Deep-pipelined GEMM** (float4 loads, double buffering, cp.async, BK≥32; FP32 or TF32) — the only credible route to ~30 fps; the current kernels run ~3.5–5 TFLOPs vs ~15+ peak. High effort. This is different from the *shallow* TF32 variant §9 rejected.
2. **Split-K / narrow-BN for tiny-spatial convs** (`layer4_rn` 18×12 IC=768: 8 blocks on 36 SMs, ~1.0 ms for 0.4 GFLOPs; `layer3_rn` similar) — ~1–1.3 ms.
3. **1×1 projections + boundary glue** (im2col copies, layernorm/cont chains) — ~2–3 ms recoverable of the ~11 ms non-custom head time.
4. **Device-resident output** (skip logits readback + host exp + re-upload for JPEG encode; needs a capture-on-device engine hook) — ~0.5–1 ms.

The backbone (~24 ms) remains cuBLAS + flash-attention territory. The `small` checkpoint is ruled out by requirements.

### Maintenance note

The vendored ggml now carries a local patch (custom-op CUDA dispatch, ~40 lines, marked "depth-anything.cpp local extension"). A `git submodule update` in `third_party/ggml` will drop it — re-apply the block from this repo's history or keep the submodule pinned.

---

## 9. Round two: K2, TF32, pre-warm (as built)

| Item | Verdict | Detail |
|---|---|---|
| **K2a: fused upsample→1×1 out-conv** (`k_convup_igemm<...,1>`) | **KEPT, on by default** | Replaces all four `interpolate(×2, align_corners) → conv1×1` pairs in the fusion blocks; the upsampled intermediate never exists. 0.03–0.47 ms per instance; parity *improved* (max rel 8.0e-4 vs 1.2e-3 — fewer materialization steps) |
| **K2b: fused upsample(+PE)→3×3 out2a** (`k_convup_igemm<...,3>`) | **implemented, gated OFF** (`DA_FUSE_OUT2A=1`) | Measured 2.85 ms vs ~2.3 ms unfused — the 9-tap × 4-read bilerp gather at 504×336 is too redundant. Kill-criterion applied |
| **TF32 tensor-core conv** (`k_conv3x3_wmma`, m16n16k8, F32 accumulate) | **implemented, gated OFF** (`DA_TF32=1`) | Parity excellent (max rel 9.2e-4 — F32 accumulate dominates), but *slower* (45–47 vs 42–44 ms/iter): these kernels are bound by the virtual-im2col **gather**, not FMA throughput. A TF32 win would need BK≥32 double-buffered pipelines — noted as future work, not warranted now |
| **Pre-warm** (`DEPTH_UI_PREWARM=off` to skip) | **KEPT, on by default** | Server loads all downloaded variants + one dummy 3:2 inference each at startup (~0.6 s). First request drops from ~330 ms to ~47 ms; `/health` shows `loaded:true` immediately. Other aspect ratios still pay one graph compile on first use |

**Result: warm server total 38.8 → 37.9 ms median (min 35.6), and no cold-start penalty on first request.** The main lesson of this round: both "obvious" wins (TF32 math, fusing the biggest conv's upsample) lost to measurement — the kernels are gather-bound, and redundant gathers cost more than saved intermediates. Everything is env-gated so each decision is re-testable in one variable.

---

## 10. Round three: small-shape items 2 & 3 (as built)

Implemented the two tractable items from the §8 headroom list (item 1, the pipelined GEMM rewrite, is planned separately in [`gemm-rewrite-plan.md`](gemm-rewrite-plan.md); item 4, device-resident output, deliberately deferred).

| Item | Change | Measured |
|---|---|---|
| **2. Narrow tiles for tiny-spatial convs** | Conv kernel template generalized to `<BM,TM,BN,TN>`; shapes with P<256 px dispatch a 32×16 tile | `layer4_rn` (18×12, IC=768): **1.00 → 0.56 ms**. Threshold deliberately excludes 36×24: measured there that extra per-block A-tile reloads made narrow tiles *slower* (0.50→0.56 ms) — the redundant-load trade-off only split-K (GEMM-rewrite plan) solves properly |
| **3. Dedicated 1×1 conv kernels** | `k_conv1x1_igemm` (direct GEMM, input read in place — no im2col copy) + `k_conv1x1_tinyoc` (per-pixel kernel for OC≤4) wired into `conv2d()` for stride-1 pad-0 | The four 1536-ch projections: 0.21–0.42 ms each (≈ parity with ggml's mul_mat, minus the im2col copies); `out2b` (32→2 @504×336): **0.04 ms** and its 21 MB im2col copy eliminated |

End-to-end (q8, 20-request median): warm server **37.9 → 36.0 ms (~28 fps)**, first request ~44 ms. Parity: max 1.03e-3 relative — unchanged band. Honest summary: these were the expected small potatoes (~2 ms combined); the remaining path to 30+ fps is the pipelined GEMM rewrite (plan doc), which now also inherits two measured lessons from this round — split-K is genuinely needed for the small-P shapes, and A-tile reload traffic is the binding constraint when tiles shrink.
