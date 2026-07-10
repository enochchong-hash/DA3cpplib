# Depth Anything 3 Service — Setup & Development Guide

*Audience: engineers building, maintaining, or extending this release. Assumes fluency with CMake, C++17, and CUDA builds.*

---

## 1. What this is

A local monocular depth-estimation service built from:

| Component | Location | Role |
|---|---|---|
| [depth-anything.cpp](https://github.com/mudler/depth-anything.cpp) | `3rdparty/depth-anything.cpp/` — git submodule pinned to upstream `f4e17de`, plus **3 local patches** | C++17/ggml port of ByteDance Depth Anything 3. Provides `libdepthanything.a`, a C API (`include/da_capi.h`), and the one-shot `da3-cli` |
| ggml | `3rdparty/depth-anything.cpp/third_party/ggml/` — nested submodule pinned by upstream to `3af5f57`, plus **1 local patch** | Compute backend (CUDA); the patch adds custom-op CUDA dispatch used by our fused DPT-head kernels |
| Local patches | `scripts/patches/{depth-anything.cpp,ggml}/*.patch` | `git format-patch` files applied by `scripts/apply_patches.sh` (`git am --3way`, idempotent) |
| GGUF models | `resources/nnmodels/DepthAnything-Base-{F32,Q8,Q4}/` | DA3 *base* (DualDPT) weights from [mudler/depth-anything.cpp-gguf](https://huggingface.co/mudler/depth-anything.cpp-gguf); fetched by `scripts/download_models.sh` |
| `depth-ui-server` | `src/server.cpp`, `src/gpu_preprocess.cu`, `include/gpu_preprocess.h` | **Our code.** Persistent C++ HTTP server: model residency, GPU preprocess (nvJPEG + CUDA), GPU JPEG output, prewarm |
| Web UI | `resources/html/index.html` + `resources/js/app.js` | Upload page, live camera, client-side render styles |
| cpp-httplib | `3rdparty/cpp-httplib/` (`httplib.h` + `httplib.cpp` + LICENSE) | Vendored copy of the split httplib that llama.cpp uses — the `.cpp` **must** be compiled into the target or linking fails |
| Scripts | `scripts/` | setup / build / start / stop / bench / TLS — see `README.md` |

**Design constraints that shaped this (do not regress):**

1. **No Python anywhere in the inference or serving path.** The server is pure C++ linking the engine directly. (Python may appear in tests/tooling only.)
2. **Relocatable and submodule-friendly.** No absolute paths: scripts resolve their root from `${BASH_SOURCE[0]}`; the binary resolves resources via env vars, then `<exe>/../resources` (`/proc/self/exe`), then a compile-time fallback; CMake uses `CMAKE_CURRENT_SOURCE_DIR` only (never `CMAKE_SOURCE_DIR` — wrong under `add_subdirectory()`). The only sanctioned absolute path is the installed CUDA toolkit (`/usr/local/cuda*`).
3. **Model folders are labeled by quant** (`-F32`, `-Q8`, `-Q4`) because we compare quants side by side.
4. **The API contract is frozen** — see `user-guide.md`; timing fields in responses are the product.

---

## 2. Building from scratch

### 2.1 Prerequisites

- NVIDIA GPU + working driver (`nvidia-smi`). Reference: RTX 5060 8 GB (Blackwell, compute capability 12.0).
- **CUDA toolkit ≥ 12.8** (Blackwell requires it). The scripts auto-detect `/usr/local/cuda*` even when Ubuntu's obsolete `nvidia-cuda-toolkit` apt package (11.5) shadows PATH; `scripts/setup_cuda.sh` installs 12.8 from NVIDIA's apt repo and `--remove-old` purges the 11.5 packages.
- CMake ≥ 3.18, gcc with C++17, OpenSSL headers (optional, enables HTTPS).

### 2.2 One command

```sh
scripts/setup.sh          # prereq checks → submodules → patches → models → build
```

Idempotent. Piecewise equivalents:

```sh
scripts/apply_patches.sh              # submodule init + git am the local patches
scripts/download_models.sh all        # or f32|q8|q4
scripts/build.sh                      # cmake configure + build (single tree)
```

Outputs: `build/depth-ui-server` and `build/3rdparty/depth-anything.cpp/examples/cli/da3-cli`.

`build.sh --arch native` (default) targets the host GPU; pass an explicit arch
number for cross-builds (Blackwell = 120, needs CUDA ≥ 12.8).

### 2.3 How the patch mechanism works (read before touching submodules)

- The release repo pins `3rdparty/depth-anything.cpp` at **upstream** `f4e17de`; our work ships as patch files, applied by `git am --3way` with original authorship. After patching, `git log` inside the submodule shows exactly what is local, and the gitlink shows as *modified* in the release repo — **expected**, don't "fix" it.
- Apply order: ggml patch first (the engine code calls into the patched ggml API), then engine patches.
- `git submodule update --checkout --force` **discards the patches** — re-run `scripts/apply_patches.sh` afterwards. The script detects already-applied/partial states by commit subject.
- To change local engine code: commit inside the submodule, then regenerate patches:
  ```sh
  cd 3rdparty/depth-anything.cpp
  git format-patch f4e17de..HEAD -o ../../scripts/patches/depth-anything.cpp/ \
      -- . ':(exclude)third_party/ggml'      # NEVER include the gitlink hunk
  ```
  (For ggml: `git format-patch 3af5f57..HEAD -o ../../../../scripts/patches/ggml/` from inside `third_party/ggml`.)

### 2.4 Smoke test

```sh
scripts/start_all.sh
curl -s localhost:8090/health
tests/smoke_test.sh          # 13 API-contract checks on an ephemeral port
tests/parity_test.sh         # cuda_fused vs im2col numeric parity
scripts/bench.sh             # stage timings; gates in docs/perf-baselines.md
```

---

## 3. Server architecture (`src/server.cpp`)

Single translation unit. Request flow for `POST /depth`:

```
httplib thread pool (req.body already fully read by httplib)
  → write body to mkstemp("/tmp/depth_ui_XXXXXX")
  → lock g_mutex
      → per-variant engine cache lookup; miss = one-time load (prewarm avoids this)
      → GPU path: nvJPEG decode → CUDA resize/normalize → D2D copy into graph input
      → depth-only fused graph (custom DPT-head CUDA kernels, DA_CONV=cuda_fused)
  → unlock, unlink temp file
  → depth normalize/flip on GPU → nvJPEG grayscale encode → binary JPEG response
    (X-Timings-Ms header; ?format=json = legacy lossless PNG/base64 contract)
```

Key decisions and invariants:

- **Model residency**: engine contexts live in a global map for the process lifetime; prewarm (default, `DEPTH_UI_PREWARM=off` to skip) loads all downloaded variants + runs one dummy inference at startup, so even the first request is warm.
- **Serialization**: one global mutex covers load + inference (single GPU; engine thread-safety undocumented). httplib's pool still overlaps network I/O.
- **Resource resolution**: env `DA3_MODEL_DIR`/`DA3_WWW_DIR` → `<exe>/../resources/…` → compile-time default. `DA3_API_ONLY=1` disables the UI routes (JSON banner at `/`).
- **Depth display convention**: base DA3 emits distance (larger = farther); the encoder flips so PNG/JPEG is near = bright. A model that emits *inverse* depth would need a per-model flip flag (see `docs/plans/future-work.md`).
- **Timing fields** (`X-Timings-Ms` / `timings_ms`) are the point of the tool — keep names and semantics stable.
- **Resolution paths**: default = production preprocess (longest side → 504, what DA3 was trained at); `?res=full` = legacy near-input resolution, ~10× slower, denser but not more accurate. Switching res re-triggers a one-time ggml graph compile (~200 ms).
- **stb symbols**: `STB_IMAGE_WRITE_STATIC` keeps the server's stb copy TU-local so it can't collide with the engine's at link time.
- **httplib vendoring**: `3rdparty/cpp-httplib/httplib.cpp` is compiled into the target (split vendoring, same as llama-server). OpenSSL, if found, enables `CPPHTTPLIB_OPENSSL_SUPPORT` for `--tls`.

Performance history and kernel details: `performance-optimization.md`,
`dpt-head-cuda-kernel-plan.md` (as-built), `plans/gemm-rewrite-plan.md` (pending).

---

## 4. Common tasks

### 4.1 Add a model variant

1. Download the GGUF into `resources/nnmodels/DepthAnything-<Name>/` (add it to `scripts/download_models.sh`).
2. Add an entry to `MODEL_PATHS` in `src/server.cpp` (and the error string listing variants).
3. Add an `<option>` to the variant `<select>` in `resources/html/index.html`.
4. Add a case to `scripts/run_cli.sh`.
5. Update the user guide's variant table; extend `tests/smoke_test.sh` test 3.
6. Rebuild, restart, verify via `/health` and one POST per new variant.

Caveats: DA3 base upstream quants are `f32/f16/q4_k/q8_0` **only** — the `q5_k`/`q6_k` files in the HF repo belong to the older Depth Anything V2 models. The *nested* metric models need `da_capi_load_nested(anyview, metric, threads)` — a new code path, not just a map entry.

### 4.2 Extend the API

`da_capi.h` also exposes pose, dense output with confidence/sky masks, point clouds, and glb/COLMAP export. All follow the same pattern: call under the mutex, free with the matching `da_capi_free_*`. (The server already bypasses the C API for `/depth` and calls `da::Engine` directly to skip the unused pose graph.)

### 4.3 A/B the custom CUDA kernels

Everything is env-gated: `--conv im2col` (or `DA_CONV=im2col`) reverts to stock ggml paths in one variable. `DA_PROFILE=1` prints per-stage engine timings; `DA_CUDA_HEAD_STATS=1` dumps per-shape kernel timings at exit; `da3-cli depth --model … --input … --repeat 20` gives server-independent warm medians.

---

## 5. Gotchas log (hard-won, read before debugging)

- **`pkill -f depth-ui-server` can kill your own shell** — the pattern matches the invoking command line. Use `scripts/stop_all.sh` or `pkill -x depth-ui-server`.
- **llama.cpp cannot load these GGUFs** (`unknown model architecture: 'depthanything3'`). GGUF is just the container — the graph lives in depth-anything.cpp.
- **Ubuntu's `nvidia-cuda-toolkit` apt package is CUDA 11.5** and shadows `/usr/local/cuda` on PATH. The scripts prefer `/usr/local/cuda*` (see `scripts/cuda_env.sh`); `setup_cuda.sh --remove-old` removes the offender.
- **Exit code 127 from scripts** usually means a relative path after a `cd` elsewhere; all scripts resolve their root from their own location, so invoke them by any path from anywhere.
- **`git submodule update` after patching** resets the submodules and drops the local patches — re-run `scripts/apply_patches.sh` (§2.3).
- The engine binds a few host-read tensors on CPU (`4 host-read tensors kept on CPU` in the log) — normal, not a partial-offload bug.
- Output resolution ≠ input resolution (the model processes at its own grid). Don't assume 1:1 pixel mapping to the source image.
- On the original gemma4 dev box, ports **8080–8087** belong to the LLM server fleet (host-specific note); depth uses 8090.
- For sub-second GPU bursts use `nvidia-smi dmon -s u` (per-second SM%), not the long-window `utilization.gpu` counter — the latter reads near-idle during request loops that are actually GPU-bound.
