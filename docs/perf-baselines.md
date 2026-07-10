# Performance Baselines

This document records measured performance numbers for the DA3 depth-estimation service across different hardware configurations.

## Reference Hardware (gemma4 dev box)

**Date**: 2026-07  
**GPU**: RTX 5060 8 GB  
**CUDA**: 12.8  
**Test image**: desk.jpg (1024×673)  
**Model**: q8  
**Resolution**: std (504 px longest side)

### Baseline Numbers

| Metric | Value | Gate |
|--------|-------|------|
| Server total (median) | ~36.0 ms | ≤ 40 ms |
| Inference (median) | ~33.5 ms | ≤ 38 ms |
| GPU preprocess (median) | ~2.2 ms | ≤ 4 ms |
| GPU JPEG encode (median) | ~0.3 ms | ≤ 1 ms |
| First request (prewarmed) | ~44 ms | ≤ 80 ms |
| im2col server (median) | ~52.5 ms | ≤ 60 ms |
| Full-res server (median) | ~800 ms | informational |

### Variant Comparison (warm)

| Variant | Inference | Notes |
|---------|-----------|-------|
| f32 | ~315 ms | Reference quality |
| q8 | ~300 ms | Near-lossless |
| q4 | ~300 ms | Visually indistinguishable |

**Gate**: Spread ≤ 15% (quantization must NOT change warm GPU speed)

### GPU Path Verification

- `nvidia-smi dmon -s u -d 1` shows SM% bursts ≥ 30% during inference
- Startup log: `ggml_cuda_init: found … CUDA devices`
- VRAM with all three variants loaded: ≤ ~1.5 GB

## Measurement Tools

### bench.sh

```sh
./scripts/bench.sh [--variant q8] [--n 20] [--warmup 3] [--res std|full] [--port 8090] [--json]
```

Reports: median, min, max, p95 for each stage (save/model_load/preprocess/infer/encode/server).

### Profiling Knobs

- `DA_PROFILE=1` — Per-stage engine timings
- `DA_CUDA_HEAD_STATS=1` — Per-shape kernel table
- `da3-cli --repeat 20` — Server-independent warm medians

## Notes

- **cuda_fused vs im2col**: The speed difference proves custom kernels are active. No speedup = packaging bug.
- **First request**: Includes model load (~60 ms for q8) + graph warm-up (~200 ms for new aspect ratios).
- **Preprocess**: GPU path only for std res, JPEG/PNG input. Full-res or unknown format falls back to CPU.

## Append New Baselines

To add a new measurement:

```sh
./scripts/bench.sh --json >> docs/perf-baselines.md
```

Format:

```
**Date**: YYYY-MM-DD  
**GPU**: Model  
**Driver**: Version  
**CUDA**: Version  
**Git SHA**: <commit>  
**Results**: <bench.sh --json output>
```
