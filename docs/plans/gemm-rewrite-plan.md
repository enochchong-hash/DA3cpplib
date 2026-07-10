# Plan: Deep-Pipelined GEMM Rewrite for the DPT-Head Kernels

*Status: PLANNED. This is the "one large remaining lever" from `dpt-head-cuda-kernel-plan.md` — the only credible route from ~26 fps to ~30+ fps at base-checkpoint quality (smaller checkpoints are ruled out by requirements). It is deliberate, multi-day kernel engineering; everything cheaper has already been tried and measured.*

## 1. Why the current kernels leave ~2× on the table

The shipped kernels (`k_conv3x3_igemm` family) are correctness-first tiled SGEMMs:

| Property | Current | Consequence |
|---|---|---|
| K-tile (BK) | 8 | Only 8 FMAs per smem element staged — sync overhead amortized poorly |
| Global loads | scalar `float` | 4-byte transactions; no `float4`/`cp.async` — memory latency exposed |
| Pipeline | single-buffered | compute stalls while the next tile loads (`__syncthreads` twice per BK step) |
| B-tile (virtual im2col) | gathered **per tap** | each input element is fetched ~9× per block (once per kernel tap) — this is why the kernels are *gather-bound*, and why the shallow TF32 experiment (§9 of the head-kernel doc) made things *slower* |
| Effective throughput | ~3.5–5 TFLOPs | vs ~15+ TFLOPs FP32 peak on the RTX 5060 |

Measured inventory to beat (q8 base @504×336, per forward, `DA_CUDA_HEAD_STATS=1`):

| Shape (W×H, IC→OC) | Time | Bound by |
|---|---|---|
| 144×96, 128→128 ×4 (RCUs) | 3.1 ms | gather + pipeline |
| 288×192, 128→64 (out1) | 1.6 ms | gather + pipeline |
| 504×336, 64→32 (out2a) | 1.9 ms | gather + pipeline |
| 72×48 / 36×24 RCUs + layer_rn | ~3 ms | mixed; smallest are occupancy-bound |
| **Total conv kernels** | **~11 ms** | target: **~5–6 ms** |

## 2. Design

### Phase A — halo-staged B tiles (the single most important change)

Replace the per-tap virtual-im2col gather with **input-tile staging**: for each block's BN-pixel tile and each IC slice, load the corresponding input patch *plus 1-pixel halo* into shared memory **once**, then let all 9 taps read from smem. This converts ~9 global reads per input element into 1, directly attacking the measured bottleneck. Requirements:

- Block pixel tiles must be 2-D (e.g. 16×8 pixels, not a flat 64-run) so the halo is compact: (16+2)×(8+2)×IC_slice floats.
- Smem per stage at IC_slice=8: 18×10×8×4 B ≈ 5.6 KB — fits easily even double-buffered.
- The GEMM inner loop indexes B as `smem[tap_offset + pixel]` — no bounds math in the hot loop (halo pre-zeroed for image borders).

*Expected effect: turns the kernels from gather-bound into compute/pipeline-bound. This must land before TF32 can pay off.*

### Phase B — pipelined loads

- `cp.async` (SM ≥ 8.0; sm_120 supports it) for A and B staging: global→smem without transiting registers, overlapped with compute.
- Two-stage double buffering (`stage ^= 1`), `cp.async.commit_group/wait_group` replacing one of the two `__syncthreads()` per K-step.
- BK 8 → 32; A-tile loads vectorized `float4` (weights are contiguous [OC][IC·9] rows).
- Thread tile 4×4 → 8×8 (register blocking); 256 threads, BM=64, BN=128.
- Register budget ≈ 64 acc + ~40 operand/index → ~110 regs/thread; target ≥ 2 blocks/SM occupancy — tune BM/BN down if the compiler spills.

### Phase C — TF32 tensor cores *on top of the pipeline*

Only after A+B: the §9 experiment proved TF32 without a deep pipeline is a loss. With B staged in smem once and cp.async keeping stages full:

- `mma.sync` m16n16k8 (or m16n8k8) TF32 fragments, F32 accumulators (parity already validated at 9.2e-4 in the shallow experiment — accumulation precision is unchanged).
- Fragment loads from the staged smem tiles (`ldmatrix` is f16-only; TF32 uses regular `lds` — fine, smem bandwidth is ample at BK=32).
- Keep the FP32 path compiled and env-selectable (`DA_TF32`) — same kill-criterion discipline.

### Phase D — epilogue & integration (unchanged machinery)

- Same fused epilogue flags (pre-ReLU / bias / residual / PE), same `GGML_OP_CUSTOM` launch-descriptor mechanism, same `DA_CONV=cuda_fused` gate — the rewrite swaps kernel bodies, not plumbing.
- Small-OC (≤32) and small-P (≤1k pixels) shapes keep their narrow-tile variants; the rewrite targets the five large shapes in the table above.

## 3. Validation protocol

1. **Standalone parity harness first**: a tiny binary that runs each exact head shape with random tensors against `ggml_conv_2d` (im2col reference); gate FP32 ≤ 1e-4 max rel, TF32 ≤ 2e-3. (Round-two lesson: end-to-end PFM diffs are the final gate, but per-shape harnesses localize bugs in minutes instead of hours.)
2. `DA_CUDA_HEAD_STATS=1` per-shape table before/after each phase — kill any phase that doesn't show its predicted win (precedent: shallow TF32 and the out2a fusion were both killed this way).
3. End-to-end: PFM parity vs `DA_CONV=im2col`, then 20-request server median.

## 4. Acceptance / kill criteria

| Gate | Threshold |
|---|---|
| Phase A alone | ≥ 25% on the five big shapes (else stop: theory is wrong) |
| A+B combined | conv total 11 ms → ≤ 7 ms |
| A+B+C (TF32) | conv total ≤ 5.5 ms, parity ≤ 2e-3 |
| End state | server ≤ ~33 ms (≥ 30 fps) |

## 5. Effort & risks

| Phase | Estimate | Main risk |
|---|---|---|
| A (halo staging) | 1–1.5 days | 2-D tile bookkeeping bugs at image borders (halo zeroing) |
| B (cp.async pipeline) | 1–1.5 days | register pressure → occupancy collapse; tune tile sizes |
| C (TF32 mma) | 1 day | fragment/smem layout mismatches; parity drift |
| Harness + integration + docs | 0.5–1 day | — |

Total ≈ one engineering week. Prerequisite reading: the existing kernels in `3rdparty/depth-anything.cpp/src/cuda_head_kernels.cu` and the §8/§9 verdicts in `dpt-head-cuda-kernel-plan.md` — every dead end listed there was measured; don't re-walk them.
