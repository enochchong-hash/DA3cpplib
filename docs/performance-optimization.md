# Depth Anything 3 Service — Performance Optimization Report

*What was done to make inference fast, how each bottleneck was found, the measured impact, and how to profile this yourself. Hardware baseline: RTX 5060 8 GB (Blackwell), CUDA 12.8, test image `desk.jpg` (1024×673 JPEG).*

---

## 1. Results summary

| Configuration | Warm inference | Warm server total | Output |
|---|---|---|---|
| Naive baseline: one-shot CLI per request | — | ~1,250 ms | 1022×672 |
| \+ Persistent server, model resident in VRAM | ~294 ms | ~313 ms | 1022×672 |
| \+ Production-resolution path | ~78 ms | ~83 ms | 504×336 |
| \+ Depth-only graph, skip unused pose head | ~58 ms | ~63 ms | 504×336 |
| \+ GPU preprocessing: nvJPEG decode + CUDA resize/normalize, device-resident input | ~49 ms | ~56 ms | 504×336 |
| \+ GPU JPEG output: nvJPEG grayscale encode, binary response | ~50 ms | ~52.5 ms | 504×336 |
| \+ Hand-written DPT-head CUDA kernels (`DA_CONV=cuda_fused`) | ~36 ms | ~38.8 ms | 504×336 |
| \+ Fused upsample→1×1 convs + startup pre-warm | ~35 ms | ~37.9 ms (first request too) | 504×336 |
| \+ Narrow tiles (tiny-spatial convs) + direct 1×1 GEMM kernels (**current default**) | **~33.5 ms** | **~36.0 ms** | 504×336 |
| Optional high-res mode (`?res=full`) | ~790 ms | ~800 ms | 1022×672 |

Net: **~32× faster** than the naive per-request CLI approach; **~26 fps** sequential. The custom kernels are documented in [`dpt-head-cuda-kernel-plan.md`](dpt-head-cuda-kernel-plan.md) §8 (as-built): fused implicit-GEMM conv3×3 (pre-ReLU/bias/residual epilogue) + a pixel-shuffle GEMM for the stride==kernel transposed convs — the latter replaced ggml's naive conv2d-transpose kernel, which profiling revealed as the single biggest head cost (~12.5 ms → 0.25 ms). Parity vs stock ggml: max 1.2e-3 relative. A/B anytime with `DA_CONV=im2col`.

---

## 2. Optimization 1 — keep the model resident in GPU memory

**Problem.** The CLI (`da3-cli`) is one-shot: every invocation pays process start, CUDA context init, and GGUF model load (~380 ms for f32) before any inference.

**Change.** The service (`src/server.cpp`) loads each model variant **once** via `da_capi_load()` on its first request and caches the `da_ctx*` in a global map for the process lifetime. Weights stay in VRAM; each request ships only the image. Warm requests report `model_load_ms: 0`.

**Impact.** Removes ~380 ms (f32) / ~62 ms (q8) / ~44 ms (q4) + process overhead from every request after the first. All three variants resident together use well under 1.5 GB VRAM.

---

## 3. Optimization 2 — use the production-resolution pipeline (the big one)

**Problem.** The first server version called `da_capi_depth_path()`, which routes to `Engine::depth_image()` → the **legacy** preprocess: it only floors each dimension to a patch multiple, so a 1024×673 input runs the ViT at ~1022×672 — **~4× the pixels the model was designed to process**. Warm inference: ~294 ms.

**How it was found.** The engine has a built-in stage profiler, enabled with the `DA_PROFILE=1` environment variable. Running the CLI with it showed:

```
profile: [fused] preprocess=5.6ms graph(backbone+head)=210.3ms   <- first run (graph compile)
profile: [fused] preprocess=4.7ms graph(backbone+head)=43.5ms    <- warm runs
```

The CLI's warm forward was **43 ms at 504×336**, while the server measured ~294 ms — and the server's output was 1022×672. Reading the engine source explained it: the CLI uses `preprocess_real` (the faithful DA3 InputProcessor: resize longest side to the model's `img_resize_target` = 504), but the C-API function the server called uses the legacy path.

**Change.** The server now calls `da_capi_depth_dense()` (routes to `depth_pose_native` → `preprocess_real`, identical to the reference implementation and `da3-cli`). The legacy near-input-resolution path is kept as an explicit opt-in: `POST /depth?...&res=full` (or the "high-res output" checkbox in the UI).

**Impact.** Warm inference 294 ms → **78 ms**; server total 313 ms → **83 ms**. Note 504 px is not a quality downgrade — it is the resolution DA3 was trained and validated at; the "full-res" mode runs the ViT off its production operating point and mainly yields a denser (not necessarily more accurate) map.

---

## 3b. Optimization 3 — skip the camera-pose head the endpoint discards

**Problem.** `da_capi_depth_dense()` routes to `depth_pose_native`, a DualDPT graph that computes depth **and camera pose**. The `/depth` endpoint throws the pose away. Measured with `DA_PROFILE=1`: depth+pose ~72 ms/iter vs depth-only ~53 ms/iter — **~20 ms of pure waste per request**.

**Change.** The server now bypasses the C API and calls the C++ engine directly (`da::Engine::depth_native()` — production-res, depth+conf, no pose graph). This also drops one full-frame `malloc`+`memcpy` the C API did per result. `is_da2`/`is_mono` model types are routed to their matching depth-only calls.

**Impact.** Warm inference 78 ms → **58 ms** (median over 20, spread 57–60); server total 83 → **63 ms**.

## 3c. Optimization 4 — hardware decode + GPU preprocessing, image stays on GPU

**Problem.** Per request the CPU was doing: JPEG decode (stb, ~8–10 ms), two-step cv2-style resize + ImageNet normalize (~5 ms), then a host→device copy of the CHW tensor. ~13–15 ms of CPU work + a needless CPU round-trip for data that starts (JPEG bits) and ends (graph input) near the GPU.

**Change** (implemented in `src/gpu_preprocess.{h,cu}` + small engine extensions):

- **JPEG**: decoded by **nvJPEG** (hardware/CUDA) straight into device memory.
- **PNG**: no consumer-GPU PNG decoder exists (it's DEFLATE), so PNGs decode on CPU (stb) with one H2D copy of the raw pixels — then join the same GPU pipeline.
- **Resize + normalize**: custom CUDA kernels replicating the engine's cv2-faithful `preprocess_real` exactly — the two-step resize policy (longest side → 504 via INTER_AREA/INTER_CUBIC a=-0.75, then round dims to patch multiples), uint8 saturation (round-half-even) between steps, ImageNet normalize to CHW f32.
- **No CPU round-trip**: three new hooks in the engine — `Backend::add_graph_input_nd_upload` (input filled by callback after graph allocation), `DinoBackbone::build_feats_graph_pre`, `Engine::depth_native_prepared` — let the server copy the preprocessed tensor **device-to-device** into the graph input (`cudaMemcpy D2D`, with an `ggml_backend_buffer_is_host` check for the CPU-buffer edge case).
- Any failure (exotic JPEG, non-DA3 model, no CUDA) falls back to the unchanged CPU path; responses report `gpu_preprocess: true/false` and a separate `preprocess_ms`.

**Impact** (q8, 20-request median): preprocess 2.2 ms on GPU (JPEG; PNG ~19 ms due to CPU DEFLATE), infer 48.7 ms, **server total 56 ms** (was 63). Verified correct: with the same decoder, GPU vs CPU preprocess depth maps differ by mean 0.28/255 (float rounding); nvJPEG vs stb adds a small extra delta (JPEG doesn't mandate bit-exact IDCT) — visually identical.

## 3d. Optimization 5 — GPU JPEG output, binary response (no PNG, no base64)

**Problem.** Every response spent ~5 ms CPU on PNG encode + base64, and base64 inflates the payload by 33% (a 504×336 depth PNG was ~219 KB as base64 JSON).

**Change.** The depth map is now normalized/flipped to 8-bit by a CUDA kernel (min/max via thrust on device) and encoded as a **grayscale JPEG (quality 90) by nvJPEG's encoder** — decode *and* encode are both hardware-path. The HTTP response is the **raw JPEG binary** (`Content-Type: image/jpeg`) with metadata and timings in `X-*` headers; no JSON wrapper, no base64. `?format=json` keeps the old lossless PNG/base64 contract (also the automatic fallback if the GPU encoder errors). One honest caveat: the depth floats currently make a host round-trip (graph readback → H2D for encode, ~0.2 ms for 680 KB) because the engine's compute API reads outputs back; eliminating that needs a capture-on-device engine hook and is bundled into the DPT-head plan.

**Impact** (q8, 20-request median): encode 5 ms → **0.3 ms**; payload 219 KB → **~10 KB**; server total 56 → **52.5 ms** (~19 fps). JPEG is lossy — visually identical for display; SDK users needing exact values use `?format=json`.

## 4. Manual-CUDA audit — what was examined and what it found

The question "can we hand-write CUDA to go faster?" was answered by scanning every op the graph uses and where it executes:

- **No CPU fallbacks inside the GPU graph.** All ops used (`conv_2d` via im2col, `conv_transpose_2d`, `interpolate`/`upscale`, `concat`, `flash_attn_ext`, `soft_max_ext`, matmuls) have CUDA kernels in the vendored ggml. The engine's one *custom* op (`ggml_custom_4d` Winograd conv in `winograd.cpp`) is CPU-only **by design and correctly gated off in GPU mode** (`use_wino = wino_ok && !gpu` in `dpt_blocks.cpp`) — GPU convs go through im2col + cuBLAS-class matmul, which upstream measured 2× faster than the naive direct-conv CUDA kernel. `DA_CONV=winograd|direct|im2col|auto` exists for A/B.
- **Warm forward breakdown** (`DA_FUSED=0 DA_PROFILE=1`, q8 @504): backbone ≈ 24 ms (ViT-B attention/MLP — already cuBLAS + flash-attention; hand-written kernels won't beat these), DPT head ≈ 35 ms (mostly 3×3 convs). The fused single-graph mode (default) totals ~43 ms by keeping features device-resident.
- **CUDA Graphs: tried, no effect.** Rebuilt with `-DGGML_CUDA_GRAPHS=ON` (off by default; needs the flag at configure time) and A/B'd via `GGML_CUDA_DISABLE_GRAPHS=1`: 58 ms median **both ways**. Kernel-launch overhead is not a bottleneck at this graph size/resolution. The flag is now baked into the build (harmless), but it bought nothing.
- **The one genuine hand-written-CUDA candidate**: a fused implicit-GEMM (or Winograd) kernel for the DPT head's 3×3 convs, replacing im2col's materialized intermediate tensors. Potential ~10–15 ms/request; cost: real CUDA engineering + upstream-divergence maintenance. **Not implemented — poor ROI** while requests already run at 63 ms end-to-end. Revisit only if per-frame latency becomes critical (e.g. video).

## 5. Things measured that did NOT speed up inference (documented so nobody retries them blindly)

### 4.1 Quantization (q8_0 / q4_k)

| Variant | Warm inference | Model load | File size |
|---|---|---|---|
| f32 | ~314 ms* | 384 ms | 412 MB |
| q8_0 | ~294 ms* | 62 ms | 149 MB |
| q4_k | ~299 ms* | 44 ms | 104 MB |

*\*measured on the full-res path; the ranking is unchanged on the 504 path.*

On GPU, warm inference is **essentially identical across all quants**. ggml's CUDA kernels dequantize on the fly, the arithmetic per image is the same, and at ~450 GB/s the weight-size difference is worth <1 ms per pass. This matches upstream's own GPU benchmark ("~47 ms/forward across every quant"). Quantization here buys **cold-start time, VRAM, and disk — not throughput**. (On *CPU* quantization does help — memory-bandwidth-bound — but the CPU path of this CUDA build crashes and is irrelevant on this box.)

### 4.2 Verifying the GPU was actually in use (and a measurement pitfall)

A coarse check (`nvidia-smi --query-gpu=utilization.gpu` sampled at 200 ms) showed 1–3% during inference and could be misread as "GPU idle". Three checks proved otherwise:

- Load log: `da::Backend using device: CUDA0`, `offload_weights: 275 weights -> CUDA0`.
- `nvidia-smi dmon -s u` (per-second SM%) during a 30-request loop: **31–100% sm** the whole time, 0% after.
- CPU-time accounting: a full run uses ~0.2 s user CPU — CPU inference would burn multi-core-seconds.

Lesson: for sub-second GPU bursts, use `dmon` (SM%), not the long-window `utilization.gpu` counter.

### 4.3 llama.cpp as a runtime

Not possible — tested directly: `llama_model_load: error loading model: unknown model architecture: 'depthanything3'`. GGUF is only the container; the depth graph exists solely in depth-anything.cpp (which already shares the same ggml kernel library as llama.cpp, so there is no performance left on that table anyway).

---

## 6. Known per-request cost structure (current default path, warm, JPEG input)

| Stage | ~ms | Where |
|---|---|---|
| Receive + temp-file write | <1 | CPU |
| JPEG decode + resize + normalize (nvJPEG + CUDA kernels) | ~2 | **GPU** |
| ViT backbone (cuBLAS + flash attention) | ~24 | **GPU** |
| DPT head (custom fused kernels + remaining ggml ops) | ~11 | **GPU** |
| Depth normalize + grayscale JPEG encode (thrust + nvJPEG) | ~0.3 | **GPU** |
| **Server total** | **~38.8** | |

Caveats:
- **First request per variant**: + model load (44–384 ms) + one-time ggml graph compile (~200 ms).
- **Switching resolution modes** (`std` ↔ `full`) re-triggers the graph compile (~200 ms once per switch) — alternating per request forfeits the warm-graph benefit.
- Requests are serialized behind one mutex (single GPU); throughput ≈ 15 req/s warm at standard resolution.

## 7. How to profile (toolbox used for all of the above)

```sh
DA_PROFILE=1 da3-cli depth --model m.gguf --input img.jpg --png /dev/null --repeat 5
                                  # per-stage: preprocess vs graph forward, warm medians
nvidia-smi dmon -s u -d 1         # per-second SM utilization during a request loop
curl ... /depth | jq .timings_ms  # server-side stage timings on every response
```

## 8. Not pursued (possible future work)

- **Pre-warming**: load all variants + run one dummy inference at startup to eliminate first-request latency entirely (trivial to add if cold starts matter).
- **Hand-written DPT-head conv kernel**: the only remaining manual-CUDA candidate (see §4) — ~10–15 ms potential. Full engineering plan: [`dpt-head-cuda-kernel-plan.md`](dpt-head-cuda-kernel-plan.md).
- **Batching / multi-stream**: irrelevant for a single-user local box; the mutex-serialized design is intentional.
- **Smaller checkpoint**: `depth-anything-small-f32.gguf` (105 MB) would cut forward time further at some quality cost — untested here.
- **GPU PNG *encode* of the result** (~5 ms CPU): could return raw f32/u16 instead of PNG for latency-critical SDK callers — API addition, not an optimization of the current contract.
