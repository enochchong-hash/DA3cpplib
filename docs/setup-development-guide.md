# Depth Anything 3 Service — Setup & Development Guide

*Audience: internal engineers replicating, maintaining, or extending this setup. Assumes fluency with CMake, C++17, and CUDA builds.*

---

## 1. What this is

A local monocular depth-estimation service built from:

| Component | Location | Role |
|---|---|---|
| [depth-anything.cpp](https://github.com/mudler/depth-anything.cpp) | `tools/depth-anything.cpp/` (git clone + its ggml submodule) | C++17/ggml port of ByteDance Depth Anything 3. Provides `libdepthanything.a`, a C API (`include/da_capi.h`), and the one-shot `da3-cli` |
| GGUF models | `models/DepthAnything-Base-{F32,Q8,Q4}/` | DA3 *base* (DualDPT) weights from [mudler/depth-anything.cpp-gguf](https://huggingface.co/mudler/depth-anything.cpp-gguf) |
| `depth-ui-server` | `tools/depth_ui/` (`server.cpp`, `index.html`, `CMakeLists.txt`) | **Our code.** Persistent C++ HTTP server + browser upload UI |
| Scripts | `scripts/build_depth.sh`, `run_depth.sh`, `launch_depth_ui.sh` | Build / one-shot CLI / launch (indexed in `scripts/README.md`) |

**Design constraints that shaped this (do not regress):**

1. **No Python anywhere in the inference or serving path.** The server is pure C++ linking the engine directly. (Python may appear in tests/tooling only.)
2. **Reuse the llama.cpp ecosystem where possible.** llama.cpp itself *cannot* run these GGUFs (vision architecture, no graph implementation), but depth-anything.cpp uses the same ggml backend library, and our server's HTTP layer is the exact `cpp-httplib` vendored in `llama/vendor/cpp-httplib` — same code that powers `llama-server`.
3. **Model folders are labeled by quant** (`-F32`, `-Q8`, `-Q4`) because we compare quants side by side.

---

## 2. Replicating from scratch

### 2.1 Prerequisites

- NVIDIA GPU + driver capable of the target arch. Reference box: RTX 5060 8 GB (Blackwell, **compute capability 12.0**), driver 570.x, Ubuntu 22.04.
- **CUDA toolkit 12.8** at `/usr/local/cuda` (Blackwell needs ≥12.8; Ubuntu's `nvidia-cuda-toolkit` apt package is 11.5 — too old. Install from NVIDIA's apt repo, see `scripts/build_cuda.sh` header).
- CMake ≥ 3.18, gcc with C++17.
- A checkout of llama.cpp at `llama/` (only for `vendor/cpp-httplib` — see §4.3 if you don't have one).

### 2.2 Get the engine

```sh
git clone --depth 1 https://github.com/mudler/depth-anything.cpp tools/depth-anything.cpp
cd tools/depth-anything.cpp
git submodule update --init --recursive --depth 1     # pulls third_party/ggml
```

### 2.3 Get the models

```sh
BASE=https://huggingface.co/mudler/depth-anything.cpp-gguf/resolve/main
mkdir -p models/DepthAnything-Base-{F32,Q8,Q4}
curl -L -o models/DepthAnything-Base-F32/depth-anything-base-f32.gguf  $BASE/depth-anything-base-f32.gguf   # 412 MB
curl -L -o models/DepthAnything-Base-Q8/depth-anything-base-q8_0.gguf  $BASE/depth-anything-base-q8_0.gguf  # 149 MB
curl -L -o models/DepthAnything-Base-Q4/depth-anything-base-q4_k.gguf  $BASE/depth-anything-base-q4_k.gguf  # 104 MB
```

Upstream quant availability for **DA3 base**: `f32`, `f16`, `q4_k`, `q8_0` **only**. The `q5_k`/`q6_k` files in that HF repo belong to the older *Depth Anything V2* models (`depth-anything2-*` prefix) — don't grab them by mistake. The repo also has `small`/`large`/`giant`/`mono-large`/`metric-large`/`nested` DA3 variants (nested = best metric-scale depth, needs two GGUFs and `da_capi_load_nested`).

### 2.4 Build

```sh
scripts/build_depth.sh
```

which is equivalent to:

```sh
export PATH=/usr/local/cuda/bin:$PATH
FLAGS="-DCMAKE_BUILD_TYPE=Release -DDA_GGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120"
cmake -B tools/depth-anything.cpp/build $FLAGS -S tools/depth-anything.cpp && cmake --build tools/depth-anything.cpp/build -j
cmake -B tools/depth_ui/build          $FLAGS -S tools/depth_ui          && cmake --build tools/depth_ui/build -j
```

Adjust `CMAKE_CUDA_ARCHITECTURES` for other GPUs (`native` also works). Outputs:

- `tools/depth-anything.cpp/build/examples/cli/da3-cli`
- `tools/depth_ui/build/depth-ui-server`

Note `tools/depth_ui/CMakeLists.txt` does `add_subdirectory(../depth-anything.cpp)` into its own binary dir, so the engine compiles twice (once per build tree). Accepted cost (~2 min) for keeping our code outside the cloned repo.

### 2.5 Smoke test

```sh
scripts/run_depth.sh tools/depth-anything.cpp/assets/samples/desk.jpg /tmp/desk.depth.png
# expect log lines: "ggml_cuda_init: found 1 CUDA devices" and "da::Backend using device: CUDA0"

scripts/launch_depth_ui.sh 8090 &
curl -s localhost:8090/health
curl -s -X POST --data-binary @tools/depth-anything.cpp/assets/samples/desk.jpg \
  -H "Content-Type: image/jpeg" "localhost:8090/depth?variant=q8" | head -c 300
```

---

## 3. Server architecture (`tools/depth_ui/server.cpp`)

Single translation unit, ~250 lines. Request flow for `POST /depth`:

```
httplib thread pool (req.body already fully read by httplib)
  → write body to mkstemp("/tmp/depth_ui_XXXXXX")      [da_capi only accepts file paths]
  → lock g_mutex
      → g_ctxs lookup; on miss: da_capi_load(path, n_threads) → cache in g_ctxs   [lazy, per variant]
      → da_capi_depth_path(ctx, tmp, &h, &w) → float[h*w]
  → unlock, unlink temp file
  → normalize floats → uint8, FLIPPED (see below) → stb_image_write PNG in memory → base64
  → hand-assembled JSON response with per-stage timings (std::chrono::steady_clock)
```

Key decisions and invariants:

- **Model residency**: `da_ctx*` per variant lives in a global map for the process lifetime, so only the first request per variant pays `da_capi_load` (F32 ~380 ms, Q8 ~60 ms, Q4 ~45 ms). All three resident simultaneously is fine on 8 GB.
- **Serialization**: one global mutex covers load + inference. `da_ctx` thread-safety is not documented, and the GPU would be contended anyway. httplib's thread pool still overlaps network I/O and PNG/base64 encoding.
- **Depth display convention**: the base DA3 DualDPT model emits **distance (larger = farther)** — verified empirically (near objects had low values). `depth_to_png()` flips during normalization (`hi - depth[i]`) so the PNG is conventional **near = bright**. If you add a model that emits *inverse* depth (some relative variants), the flip must become per-model.
- **Timing fields** are measured server-side and returned as `timings_ms{save,model_load,infer,encode,server}`; the browser adds its own upload→render total. Keep these stable — they're the point of the tool.
- **Response JSON** is built by hand (only dependency-free strings + base64); `json_escape()` is applied to error strings.

- **Resolution paths (major perf lever, found by profiling)**: `da_capi_depth_path` → `Engine::depth_image` uses the *legacy* preprocess (floor dims to patch multiple ≈ input resolution, e.g. 1022×672), while `da_capi_depth_dense` → `depth_pose_native` uses `preprocess_real`, the *production* DA3 pipeline (longest side → 504, e.g. 504×336 — what the model was trained at and what `da3-cli` uses). ~4× pixel difference ⇒ warm inference **~78 ms (std) vs ~300 ms/~790 ms (full)**. The server defaults to the production path; `?res=full` opts into the legacy path. Note: switching resolution re-triggers ggml graph compile (~200 ms one-off) — alternating res per request forfeits the warm-graph benefit. Enable the engine's own stage profiler with env `DA_PROFILE=1` (prints `preprocess=…ms graph(backbone+head)=…ms`); `da3-cli --repeat N` gives warm medians.

Measured reference numbers (RTX 5060, desk.jpg): warm `infer_ms` ≈ 78 ms (std res) / ~300 ms (full res) — near-identical across **all** quant variants; quantization does not speed up warm GPU inference on this model, it buys load time, disk, and VRAM.

### 3.1 The httplib vendoring gotcha

llama.cpp vendors cpp-httplib **split** into `httplib.h` + `httplib.cpp` (not header-only). You must compile `${GEMMA4_ROOT}/llama/vendor/cpp-httplib/httplib.cpp` into the target or you get `undefined reference to httplib::Server::listen(...)`. Already handled in `tools/depth_ui/CMakeLists.txt`.

### 3.2 stb symbols

`server.cpp` defines `STB_IMAGE_WRITE_STATIC` before `STB_IMAGE_WRITE_IMPLEMENTATION` so its copy of stb_image_write stays TU-local and cannot collide with the engine library's own stb symbols at link time.

### 3.3 Compile-time paths

`CMakeLists.txt` injects `GEMMA4_ROOT` (repo root) and `DEPTH_UI_WWW` (source dir, where `index.html` is served from) as compile definitions. Consequence: the binary hard-codes absolute paths — rebuild if the repo moves (see deployment guide for packaging implications).

---

## 4. Common tasks

### 4.1 Add a model variant

1. Download the GGUF into `models/DepthAnything-<Name>/`.
2. Add an entry to `MODEL_PATHS` in `server.cpp` (and the error string listing variants).
3. Add an `<option>` to the variant `<select>` in `index.html`.
4. Add a case to `scripts/run_depth.sh`.
5. Update `scripts/README.md` + the user guide's variant table.
6. Rebuild (`cmake --build tools/depth_ui/build -j`), restart, verify via `/health` and one POST per new variant.

Caveat: `da_capi_load` signature covers single-GGUF models. The *nested* metric models need `da_capi_load_nested(anyview, metric, threads)` — a new code path, not just a map entry.

### 4.2 Extend the API

`da_capi.h` also exposes pose (`da_capi_pose_path`), dense output with confidence/sky masks (`da_capi_depth_dense`), point clouds (`da_capi_points`), and glb/COLMAP export. All follow the same pattern: call under the mutex, free with the matching `da_capi_free_*`.

### 4.3 If there's no llama.cpp checkout

The only cross-dependency is httplib. Either fetch upstream cpp-httplib (truly header-only; drop the `.cpp` from the target and just `#include`), or copy the two vendored files. Keep the include path in `CMakeLists.txt` pointing wherever they land.

---

## 5. Gotchas log (hard-won, read before debugging)

- **`pkill -f depth-ui-server` kills your own shell** on this harness — the pattern matches the invoking command line. Use `scripts/stop_depth_ui.sh [port]` or `pkill -x depth-ui-server`.
- **llama.cpp cannot load these GGUFs.** Don't try `llama-server -m depth-anything-*.gguf`; GGUF is just the container — the graph lives in depth-anything.cpp.
- **Exit code 127 from launch scripts** usually means a relative path after a `cd` elsewhere; scripts resolve `ROOT_DIR` from their own location, so invoke them by absolute path from anywhere.
- **Ports 8080–8087 are reserved** by the LLM `launch_server.sh` fleet; depth got 8090. `TENSORRT_LLM_PLAN.md` also pencils 8090 for a future TRT-LLM server — if that ever lands, move one of them.
- The engine binds host-read tensors on CPU (`4 host-read tensors kept on CPU` in the log) — normal, not a partial-offload bug.
- Output resolution ≠ input resolution (model processes at its own grid, e.g. 1022×672 from a 1024-wide input). Don't assume 1:1 pixel mapping to the source image.
