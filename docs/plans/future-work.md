# Future Work & Pending Plans

This document consolidates all pending engineering work for the DA3 depth-estimation service. Items here are **not implemented** — they represent opportunities for future optimization or feature additions.

## Implemented Work (Historical Record)

- **DPT Head CUDA Kernel Plan** — See [`dpt-head-cuda-kernel-plan.md`](dpt-head-cuda-kernel-plan.md)
  - Custom CUDA DPT head with fused im2col + GEMM
  - Achieved ~30 fps on RTX 5060 (vs ~10 fps stock ggml)
  - As-built documentation in `docs/`

## Pending Optimizations

### 1. Deep-Pipelined GEMM (→ link to gemm-rewrite-plan.md)

**Status**: Unimplemented  
**Expected impact**: ~30% additional speedup over current custom kernels

The current custom DPT head kernels use a single-stage GEMM. A deep-pipelined approach with:
- Multi-kernel fusion (overlap memory-bound and compute-bound phases)
- Asynchronous CUDA streams for tensor core utilization
- Shared-memory tiling optimizations

See [`plans/gemm-rewrite-plan.md`](plans/gemm-rewrite-plan.md) for the full technical plan.

### 2. Device-Resident Output

**Status**: Unimplemented  
**Expected impact**: ~0.5–1 ms savings per inference

Skip the logits readback to host memory. Currently the depth map is copied from device to host after inference, then immediately re-uploaded for GPU JPEG encoding. Keeping the depth tensor on-device and passing a device pointer directly to the encoder would eliminate this round-trip.

### 3. 1×1 / Boundary Glue Recovery

**Status**: Unimplemented  
**Expected impact**: ~2–3 ms, improved edge quality

The current im2col implementation has minor artifacts at patch boundaries. A post-processing step to recover 1×1 precision at boundaries would improve both visual quality and numeric accuracy.

### 4. Split-K for Tiny-Spatial Shapes

**Status**: Unimplemented  
**Expected impact**: Improved performance for small images (<256×256)

When spatial dimensions are small, the GEMM becomes bottlenecked by tensor core occupancy. Split-K reduction across multiple kernel launches could improve utilization.

## Pending Features

### 5. Nested/Metric Model Support

**Status**: Unimplemented  
**Required code path**: New `da_capi_load_nested()` API

The engine currently supports only the base DA3 model. Adding support for:
- Nested architectures (deeper backbones)
- Metric depth variants (absolute depth estimation)

This requires extending the C API and potentially adding new kernel configurations.

### 6. Batching / Multi-Stream

**Status**: Not pursued (design decision)

The server is designed for single-user local use. Adding batch inference or multi-stream processing would add complexity without clear benefit for the target use case.

**Exception**: If the service is later deployed in a multi-client scenario, this would need revisiting.

### 7. Raw f32/u16 Output Format

**Status**: Unimplemented  
**Use case**: SDK callers needing pixel-accurate depth values

The current API returns normalized 8-bit depth (JPEG/PNG). Adding a raw output format would enable:
- Point cloud generation
- AR/VR applications requiring metric depth
- Computer vision pipelines needing unquantized data

### 8. Per-Model Depth Orientation Flag

**Status**: Unimplemented  
**Required before**: Adding inverse-depth variants

The current service assumes all models emit "distance" (larger = farther). Supporting models with opposite orientation would require:
- Per-model metadata flags
- Runtime depth flip option
- Documentation updates

## Dependencies & Blockers

| Item | Blocked by |
|------|------------|
| GEMM rewrite | CUDA kernel development bandwidth |
| Metric models | Upstream model availability |
| Raw output | API contract freeze review |
| Orientation flag | Multiple model variants in production |

## Notes

- Performance estimates assume RTX 5060 8 GB, CUDA 12.8
- All optimizations must preserve the existing API contract
- Backward compatibility is a hard requirement
