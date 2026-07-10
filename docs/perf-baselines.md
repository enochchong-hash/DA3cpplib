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

### Variant Comparison (warm, std res)

Warm GPU inference is near-identical across all quants (~32 ms at std res on the
reference GPU) — quantization buys load time, disk, and VRAM, **not** throughput.
(The historical ~300 ms/variant table in performance-optimization.md §4.1 was
measured on the legacy full-res path.)

**Gate**: Spread ≤ 15% (a large spread signals a broken path, e.g. CPU fallback)

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

## Measured Baselines

### 2026-07-10 — release build validation (gemma4 dev box)

**GPU**: RTX 5060 8 GB | **CUDA**: 12.8 | **arch**: native (120a-real) | **image**: desk.jpg | **variant**: q8, std res

```json
{
  "date": "2026-07-10T09:54:25+08:00",
  "variant": "q8", "resolution": "std", "n": 20, "warmup": 3,
  "stages_ms": {
    "save": {"median": 0.1, "min": 0.1, "max": 0.1, "p95": 0.1},
    "model_load": {"median": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0},
    "preprocess": {"median": 1.7, "min": 1.6, "max": 2.1, "p95": 2.1},
    "infer": {"median": 30.7, "min": 30.3, "max": 32.7, "p95": 32.6},
    "encode": {"median": 0.2, "min": 0.2, "max": 0.5, "p95": 0.3},
    "server": {"median": 32.7, "min": 32.3, "max": 35.2, "p95": 34.8}
  },
  "fps": 30.6
}
```

A/B same day: `DA_CONV=im2col` server median **47.9 ms** (~21 fps) vs cuda_fused
**32.7–34.4 ms** (~29–31 fps) — custom kernels confirmed active.
Parity (tests/parity_test.sh): mean abs diff 0.018 u8 levels, max 1 — PASS.
All gates of the table above: **PASS**.

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
