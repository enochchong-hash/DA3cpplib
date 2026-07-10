# DA3 Release Repository — Implementation Plan

*Plan for packaging the Depth Anything 3 depth-estimation service (currently living inside the
`gemma4` working repo) into the standalone release repository at `release/da3/`.
This document is the work order for the implementing agent. A final inspection happens after
completion — every acceptance gate in §10 will be checked, so treat them as the definition of done.*

**Source repo (read from):** `/media/user/01DAEF8E3E5F5300/code/gemma4` (called `$GEMMA4` below)
**Release repo (write to):** `/media/user/01DAEF8E3E5F5300/code/gemma4/release/da3` (called `$REL` below — already an independent git repo, one placeholder commit, no remote)

---

## 1. What is being released

A pure-C++ (no Python in the serving path) local depth-estimation service:

| Piece | Today lives at | Role |
|---|---|---|
| Inference engine | `$GEMMA4/tools/depth-anything.cpp` | Clone of [mudler/depth-anything.cpp](https://github.com/mudler/depth-anything.cpp) at upstream `f4e17de` **+ 3 local commits** (see §4) |
| ggml backend | `…/third_party/ggml` (nested submodule) | Upstream-pinned `3af5f57` **+ 1 local commit** (CUDA custom-op dispatch, 33 lines) |
| HTTP server (1st party) | `$GEMMA4/tools/depth_ui/server.cpp`, `gpu_preprocess.{h,cu}` | Persistent server: model residency, GPU preprocess (nvJPEG+CUDA), GPU JPEG encode, prewarm. Port 8090 |
| Web UI (1st party) | `$GEMMA4/tools/depth_ui/index.html` | Single self-contained file (inline JS/CSS): upload, live camera, render styles |
| httplib | `$GEMMA4/llama/vendor/cpp-httplib/` (`httplib.h` + `httplib.cpp` + `LICENSE`) | Split-vendored cpp-httplib (NOT header-only as vendored — the `.cpp` must be compiled in) |
| Models | `$GEMMA4/models/DepthAnything-Base-{F32,Q8,Q4}/*.gguf` | From HF `mudler/depth-anything.cpp-gguf` (412/149/104 MB) |
| Scripts | `$GEMMA4/scripts/{build_depth,run_depth,launch_depth_ui,stop_depth_ui,setup_depth_ui_tls}.sh` | Build / CLI / launch / stop / mkcert TLS |
| Docs | `$GEMMA4/docs/da3cpp/*.md` (7 files) | User, deployment, setup, performance, kernel plan (as-built), GEMM rewrite plan (pending) |

Reference performance to preserve (RTX 5060 8 GB, CUDA 12.8, desk.jpg 1024×673, q8, std res):
**warm server total ~36.0 ms median (~28 fps), first request ~44 ms (prewarmed), infer ~33.5 ms,
GPU preprocess ~2.2 ms, GPU JPEG encode ~0.3 ms.** These are the numbers §9's benchmark must reproduce.

---

## 2. Target layout of `$REL`

```
release/da3/
├── CMakeLists.txt                  # single top-level build (engine builds ONCE — fixes current double build)
├── README.md                       # quickstart: setup.sh → start_all.sh → curl /health
├── .gitignore                      # build/, var/, resources/nnmodels/**/*.gguf, resources/certs/, *.log
├── .gitmodules                     # 3rdparty/depth-anything.cpp
├── 3rdparty/
│   ├── README.md                   # provenance + licenses of everything below
│   ├── depth-anything.cpp/         # git submodule @ upstream f4e17de (patches applied by script)
│   │   └── third_party/ggml/       # nested submodule @ upstream 3af5f57 (patch applied by script)
│   └── cpp-httplib/                # vendored copy: httplib.h, httplib.cpp, LICENSE (+ provenance note)
├── src/                            # 1st-party C++ (see §5 recommendation: no 1stparty/ dir)
│   ├── server.cpp
│   └── gpu_preprocess.cu
├── include/
│   └── gpu_preprocess.h
├── resources/
│   ├── html/index.html             # web UI (markup + CSS)
│   ├── js/app.js                   # UI logic, split out of index.html (§6.3)
│   ├── nnmodels/                   # GGUFs live here — gitignored, filled by download_models.sh
│   │   └── DepthAnything-Base-{F32,Q8,Q4}/   # keep quant-labeled folder convention
│   └── certs/                      # runtime-generated TLS material — gitignored
├── scripts/
│   ├── setup.sh                    # one-shot: prereqs → submodules → patches → models → build
│   ├── apply_patches.sh            # idempotent patch application (§4)
│   ├── build.sh                    # cmake configure + build
│   ├── download_models.sh          # HF fetch into resources/nnmodels/ [f32|q8|q4|all]
│   ├── start.sh                    # granular launcher (§7.1) — the primary entry point
│   ├── start_server.sh             # thin alias: exec start.sh --api-only "$@"   (backend-only)
│   ├── start_all.sh                # everything, sensible defaults, daemonized, health-checked
│   ├── stop_all.sh                 # stop every instance (pidfile first, pkill -x fallback)
│   ├── run_cli.sh                  # one-shot da3-cli wrapper (no HTTP)
│   ├── setup_tls.sh                # mkcert flow (port of setup_depth_ui_tls.sh)
│   ├── bench.sh                    # performance measurement harness (§9)
│   ├── da3.service.in              # systemd unit template
│   └── patches/
│       ├── depth-anything.cpp/     # 0001..0003 *.patch  (git format-patch output)
│       └── ggml/                   # 0001 *.patch
├── tests/
│   ├── smoke_test.sh               # API contract tests (§8.2)
│   ├── parity_test.sh              # cuda_fused vs im2col numeric parity (§8.3)
│   └── MANUAL_UI_CHECKLIST.md      # browser/camera checks (§8.5)
├── var/                            # runtime: var/log/, var/run/ — gitignored
└── docs/
    ├── PLAN.md                     # this file
    ├── user-guide.md               # ┐
    ├── deployment-guide.md         # │ copied from $GEMMA4/docs/da3cpp/, paths adapted (§6.5)
    ├── setup-development-guide.md  # │
    ├── performance-optimization.md # │
    ├── dpt-head-cuda-kernel-plan.md# ┘ (implemented work — historical/engineering record)
    ├── perf-baselines.md           # NEW: measured numbers per host (§9.4)
    └── plans/                      # PENDING work only
        ├── gemm-rewrite-plan.md    # the ~30 fps lever — unimplemented, copied & path-adapted
        └── future-work.md          # NEW: consolidated backlog (§6.5)
```

### Skeleton deltas (the folder was pre-created; adjust it)
- **Delete `1stparty/`** — see recommendation in §5.
- Empty placeholder files `scripts/{start_all,start_server,stop_all}.sh` get real content; `CMakeLists.txt` gets real content.
- Add `var/`, `scripts/patches/{depth-anything.cpp,ggml}/`, `docs/plans/`, `3rdparty/cpp-httplib/`, `tests/`.

---

## 3. Design decisions (already made — do not re-litigate, but record deviations if forced)

1. **Upstream submodule + patch files, not a fork.** `3rdparty/depth-anything.cpp` points at the
   *public upstream* URL pinned to `f4e17de`; our 4 commits of local work ship as `.patch` files in
   `scripts/patches/` and are applied by script. Rationale: no fork hosting needed, provenance is
   explicit, upstream rebases stay cheap, and the release repo is self-contained.
2. **Patches are applied with `git am --3way`** (not `git apply`) so the submodule history shows the
   local commits with original authorship — bisectable and easy to inspect (`git log` inside the
   submodule shows exactly what is local). Consequence: the submodule gitlink will show as dirty in
   the release repo after patching; this is expected and documented in README.
3. **Single build tree.** Top-level CMake `add_subdirectory(3rdparty/depth-anything.cpp)` once, with
   `DA_BUILD_CLI=ON` (we want `da3-cli` for profiling/bench). Today the engine compiles twice
   (once per build dir); the release fixes that.
4. **httplib is vendored by copy**, not submoduled: copy the exact three files from
   `$GEMMA4/llama/vendor/cpp-httplib/` (this precise vendoring is what the server was tested
   against; note the split `.h`/`.cpp` gotcha). Record the `CPPHTTPLIB_VERSION` from the header in
   `3rdparty/README.md`.
5. **stb is NOT vendored separately** — it comes with the engine submodule (`third_party/stb`).
6. **Models are never committed.** `resources/nnmodels/**/*.gguf` gitignored; `download_models.sh`
   is the installer. Keep quant-labeled folder names (`-F32/-Q8/-Q4`) — side-by-side comparison is
   a design constraint of the original service.
7. **No absolute compile-time paths as the primary mechanism.** Runtime env vars
   (`DA3_MODEL_DIR`, `DA3_WWW_DIR`) win; compile-time defaults remain as fallback so the bare
   binary still works on the build host (§6.2). Launch scripts always set the env vars from their
   own location → the repo becomes relocatable without rebuild (removes the current "rebuild if the
   repo moves" limitation, keep it documented for bare-binary use).
8. **API contract is frozen.** `/depth` (binary JPEG + `X-*` headers; `?format=json` lossless PNG
   path; `?variant=`, `?res=`) and `/health` JSON shape must remain byte-compatible with
   `docs/user-guide.md`. The release adds only: backend-only mode and static serving of split
   frontend assets.
9. **Design constraints inherited from the original (do not regress):** no Python anywhere in the
   inference/serving path (tests/tooling may use it); depth PNG/JPEG convention near=bright
   (flip for distance-emitting models); timing fields in responses are the product — keep them stable.
10. **Default port 8090**, bind 127.0.0.1, LAN+TLS opt-in only. (Host-specific note: on the dev
    box, 8080–8087 belong to the LLM fleet — this is a doc note, not code.)

---

## 4. Submodule + patch management (the delicate part — follow exactly)

### 4.1 Ground truth (verified)

| Repo | Upstream URL | Pin (upstream) | Local commits on top |
|---|---|---|---|
| depth-anything.cpp | `https://github.com/mudler/depth-anything.cpp` | `f4e17de` ("feat: Depth Anything V2 support…") | `0cd94f0` *jpeg encode decode* → `6028690` *custom dpt head* → `7385ec3` *tile size optimization* |
| ggml (nested, inside the above) | `https://github.com/ggml-org/ggml` | `3af5f57` ("scripts : fix sed command") — this is what upstream `f4e17de` pins | `80ea2ce` *initial* (33 lines in `src/ggml-cuda/ggml-cuda.cu`: GGML_OP_CUSTOM CUDA dispatch) |

### 4.2 Generate the patch files (from the existing working trees)

```sh
mkdir -p $REL/scripts/patches/depth-anything.cpp $REL/scripts/patches/ggml

cd $GEMMA4/tools/depth-anything.cpp
# CRITICAL: exclude the gitlink — commit 6028690 bumps the third_party/ggml submodule
# pointer to a SHA that will not exist in a fresh clone (git am would fail on it).
git format-patch f4e17de..HEAD -o $REL/scripts/patches/depth-anything.cpp/ \
    -- . ':(exclude)third_party/ggml'

cd third_party/ggml
git format-patch 3af5f57..80ea2ce -o $REL/scripts/patches/ggml/
```

Verify: 3 patch files for the engine, 1 for ggml; grep the engine patches to confirm **no**
`Subproject commit` hunk survived.

### 4.3 Register the submodule in the release repo

```sh
cd $REL
git submodule add https://github.com/mudler/depth-anything.cpp 3rdparty/depth-anything.cpp
git -C 3rdparty/depth-anything.cpp checkout f4e17de
git add .gitmodules 3rdparty/depth-anything.cpp && git commit -m "engine submodule @ upstream f4e17de"
```

If offline, add the submodule from the local clone (`git submodule add $GEMMA4/tools/depth-anything.cpp …`)
**but then rewrite the URL in `.gitmodules` to the public upstream** and `git submodule sync` —
the committed `.gitmodules` must reference the public URL, never a local path.
The nested ggml submodule is *not* registered in the release repo — it belongs to the engine repo
and is initialized recursively.

### 4.4 `scripts/apply_patches.sh` — required behavior

- Init if needed: `git submodule update --init` (release level), then
  `git -C 3rdparty/depth-anything.cpp submodule update --init --recursive` (pulls ggml @ 3af5f57).
- Apply **ggml patch first**, then engine patches (the engine code calls into the patched ggml API).
- Use identity-safe apply: `git -c user.name="da3-release" -c user.email="da3-release@local" am --3way <patch>`.
- **Idempotent**: before applying to a repo, compare `git log --format=%s <pin>..HEAD` against the
  patch `Subject:` lines — if all already present, print "already applied" and exit 0; if *some*
  are present, abort with a clear message (partial state = human needed). `git am` must run on a
  clean tree; abort with instructions if dirty.
- Print final state: `git -C <submodule> log --oneline <pin>..HEAD`.
- Document (in script header + README): running `git submodule update --checkout --force` later
  **discards the patches** — re-run `apply_patches.sh` after any submodule reset.

---

## 5. 1st-party / 3rd-party split — recommendation

**Recommendation: no `1stparty/` directory — delete it from the skeleton.**

Rationale: in this repo everything outside `3rdparty/` *is* first-party. The conventional
top-level `src/ include/ resources/ tests/ scripts/ docs/` split already communicates ownership;
a `1stparty/` wrapper would add a nesting level that contains a single component and duplicate
what `src/` means. The quarantine boundary that matters — "code we must not casually edit and
that carries external licenses" — is exactly `3rdparty/`, which stays.

Reintroduce `1stparty/<component>/` only if this release later hosts **multiple** first-party
components with independent build lives (e.g. the SAM3 service moving in beside DA3, or a client
SDK) — at that point move `src/ include/ resources/` under `1stparty/da3-server/` in one commit.
For now: one service, one `src/`.

---

## 6. First-party code migration & required code changes

### 6.1 File moves (copy from `$GEMMA4`, adapt, commit — the originals stay untouched)

| From | To |
|---|---|
| `tools/depth_ui/server.cpp` | `src/server.cpp` |
| `tools/depth_ui/gpu_preprocess.cu` | `src/gpu_preprocess.cu` |
| `tools/depth_ui/gpu_preprocess.h` | `include/gpu_preprocess.h` |
| `tools/depth_ui/index.html` | `resources/html/index.html` + `resources/js/app.js` (§6.3) |

### 6.2 `server.cpp` changes (keep each change minimal and reviewable)

1. **Model dir**: replace the `GEMMA4_ROOT`-based `MODEL_PATHS` with paths built from a
   `model_dir()` helper: `getenv("DA3_MODEL_DIR")` → else compile-def `DA3_DEFAULT_MODEL_DIR`.
   File names inside stay: `DepthAnything-Base-F32/depth-anything-base-f32.gguf`, etc.
2. **WWW dir**: same pattern — `getenv("DA3_WWW_DIR")` → else `DA3_DEFAULT_WWW_DIR`.
3. **Backend-only mode**: env `DA3_API_ONLY=1` (set by `start.sh --api-only`): register **only**
   `/depth` and `/health`; `/` returns a tiny JSON banner
   (`{"service":"da3","ui":false,"endpoints":["/depth","/health"]}`) instead of the UI. No other
   behavioral change — prewarm, timings, TLS all work identically.
4. **Static serving**: serve `index.html` at `/` from `<www>/html/index.html` and mount `/js` →
   `<www>/js` (httplib `set_mount_point`) so the split-out `app.js` loads. Nothing else mounted.
5. Keep: mutex serialization, per-variant context cache, prewarm (`DEPTH_UI_PREWARM`),
   `DEPTH_UI_HOST`/`DEPTH_UI_CERT`/`DEPTH_UI_KEY`, GPU-preprocess fallbacks, hand-built JSON,
   `X-Timings-Ms` fields — all untouched.

### 6.3 Frontend split

Move the inline `<script>` body of `index.html` (lines ~98–326) verbatim into
`resources/js/app.js`; replace with `<script src="/js/app.js"></script>`. No logic changes, no
reformatting (a clean diff against the original must show only the extraction). Verify the live
camera + render styles still work per §8.5. If anything breaks that can't be fixed within the
task budget, fall back to shipping the single-file `index.html` and record that in the completion
notes — the split is wanted but not worth destabilizing the UI.

### 6.4 Top-level `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.18)
project(da3-release LANGUAGES C CXX CUDA)   # CUDA first-class: gpu_preprocess.cu, nvJPEG
# C++17; Release default.
# Engine (once, with CLI for bench/profiling):
set(DA_BUILD_CLI ON CACHE BOOL "" FORCE)
add_subdirectory(3rdparty/depth-anything.cpp)
# Server:
add_executable(depth-ui-server src/server.cpp src/gpu_preprocess.cu
               3rdparty/cpp-httplib/httplib.cpp)          # split-vendored: .cpp REQUIRED
target_link_libraries(depth-ui-server PRIVATE depthanything CUDA::cudart CUDA::nvjpeg Threads::Threads)
# includes: include/, engine include+src (engine.hpp C++ API), engine third_party/stb, 3rdparty/cpp-httplib
# defs: DA3_DEFAULT_MODEL_DIR="${CMAKE_SOURCE_DIR}/resources/nnmodels"
#       DA3_DEFAULT_WWW_DIR="${CMAKE_SOURCE_DIR}/resources"
# OpenSSL optional → CPPHTTPLIB_OPENSSL_SUPPORT (TLS / phone camera)
```

- `CMAKE_CUDA_ARCHITECTURES`: **do not hardcode 120**. Default `native`; `build.sh --arch N`
  overrides (docs note: RTX 5060/Blackwell = 120, needs CUDA ≥ 12.8).
- Engine flags passed at configure time: `-DDA_GGML_CUDA=ON`. Keep `-DGGML_CUDA_GRAPHS=ON`
  (measured no-op but baked into the tested configuration).
- Preserve the stb symbol isolation (`STB_IMAGE_WRITE_STATIC` etc. — already inside server.cpp).

### 6.5 Docs migration

Copy the six docs + write two new ones as laid out in §2. Required adaptations (do a full-file
pass on each, not blind copy):

- Path rewrites: `tools/depth_ui/…` → `src/…`/`resources/…`; `tools/depth-anything.cpp` →
  `3rdparty/depth-anything.cpp`; `models/` → `resources/nnmodels/`;
  `scripts/launch_depth_ui.sh` → `scripts/start.sh`; `scripts/build_depth.sh` → `scripts/build.sh`;
  `scripts/stop_depth_ui.sh` → `scripts/stop_all.sh`; `<repo>` in deployment-guide = `$REL`.
- deployment-guide: systemd unit points at `scripts/start.sh`; "moving the repo requires rebuild"
  becomes "works via launch scripts; bare-binary invocation falls back to compiled defaults";
  keep the firewalld+ufw double-firewall note (mark host-specific).
- setup-development-guide §2 (replication) is superseded by `setup.sh` — rewrite that section to
  describe the release flow (submodule + patches) instead of the original clone flow; keep §3–§5
  (architecture, tasks, gotchas) with path fixes.
- `docs/plans/gemm-rewrite-plan.md`: copy; fix the two source references
  (`tools/depth-anything.cpp/src/cuda_head_kernels.cu` → `3rdparty/depth-anything.cpp/src/…`).
- **NEW `docs/plans/future-work.md`** — consolidate every pending item so nothing lives only in
  prose: deep-pipelined GEMM (→ link gemm-rewrite-plan.md); device-resident output (skip logits
  readback, ~0.5–1 ms); 1×1/boundary-glue recovery (~2–3 ms); split-K for tiny-spatial shapes;
  nested/metric model support (`da_capi_load_nested` — new code path); batching/multi-stream
  (deliberately not pursued — single-user box); raw f32/u16 output format for SDK callers;
  per-model depth-orientation flag (needed before adding inverse-depth variants).
- `3rdparty/README.md` — table: component, upstream URL, pinned SHA, license (engine MIT,
  ggml MIT, cpp-httplib MIT, stb public domain/MIT), local patches applied (list subjects),
  where patches live.
- Root `README.md` — ≤1 screen: what it is, hardware/software prereqs, 3-command quickstart
  (`scripts/setup.sh` → `scripts/start_all.sh` → `curl localhost:8090/health`), layout table,
  links into docs/, the reference perf table, and the patch/submodule caveat from §4.4.

---

## 7. Deployment & usage scripts

All scripts: `set -euo pipefail`, resolve `REL_ROOT` from `${BASH_SOURCE[0]}` (invokable from any
CWD — exit-127-after-cd was a real bug class), executable bit set, `--help` on every one.
**Never use `pkill -f depth-ui-server`** — it can match and kill the invoking shell; the safe forms
are pidfile-kill and `pkill -x depth-ui-server`. Put this warning in comments wherever relevant.

### 7.1 `start.sh` — the granular entry point

```
Usage: start.sh [options]
  --port N            listen port                     (default 8090)
  --host ADDR         bind address                    (default 127.0.0.1)
  --api-only          backend only: REST endpoints, no web UI   [DA3_API_ONLY=1]
  --tls               HTTPS; auto-generate self-signed cert w/ LAN-IP SAN if resources/certs/ empty
  --lan               shorthand for --host 0.0.0.0 --tls (phone-camera mode; prints security warning)
  --model-dir PATH    override resources/nnmodels     [DA3_MODEL_DIR]
  --conv MODE         cuda_fused (default) | im2col   [DA_CONV] — A/B the custom kernels
  --no-prewarm        skip startup model load + warm-up          [DEPTH_UI_PREWARM=off]
  --daemon            background: nohup, pidfile var/run/da3-<port>.pid, log var/log/da3-<port>.log
  --status            report running instances (pidfile + /health probe) and exit
  --stop              stop the instance for --port (or all pidfiles if no --port) and exit
```

Behavior: validate binary exists (else point at `build.sh`), validate at least one model present
(else point at `download_models.sh`), export env, exec the binary (foreground) or daemonize.
Foreground mode must remain `exec` so systemd `Type=simple` works. TLS block ports the existing
self-signed generation (SAN must include LAN IP — browsers ignore CN; regenerate on DHCP change).

### 7.2 `start_all.sh` / `stop_all.sh` — the "just make it run" pair

- `start_all.sh`: if already running (pidfile + `/health` 200) → print status, exit 0. Else
  `start.sh --daemon` with defaults, poll `/health` up to ~15 s (prewarm ≈ 0.6 s + safety),
  print health JSON + UI URL. Nonzero exit if health never comes up (tail the log).
- `stop_all.sh`: iterate `var/run/*.pid` → TERM, wait, KILL if stubborn; then `pkill -x
  depth-ui-server` as sweep; report what was stopped. Exit 0 when nothing was running.
- `start_server.sh`: `exec "$(dirname …)/start.sh" --api-only "$@"` — satisfies the
  "backend only" convenience the skeleton reserved a file for.

### 7.3 Others

- `setup.sh`: prereq checks (nvcc ≥ 12.8 at `/usr/local/cuda` or PATH, cmake ≥ 3.18, g++ C++17,
  `nvidia-smi` working, OpenSSL headers optional-warn) → `apply_patches.sh` → `download_models.sh
  ${1:-all}` (`--no-models` to skip) → `build.sh`. Idempotent; each stage prints PASS/SKIP.
- `build.sh [--arch native|<num>] [--debug] [--clean]`: configure into `build/`, build `-j$(nproc)`,
  print artifact paths (`build/depth-ui-server`, `build/…/da3-cli` — echo the real ones).
- `download_models.sh [f32|q8|q4|all]`: curl `-L --fail --continue-at -` from
  `https://huggingface.co/mudler/depth-anything.cpp-gguf/resolve/main/…` into the three folders;
  verify each file is within ±5 % of expected size (412/149/104 MB) and starts with GGUF magic
  (`head -c4` = `GGUF`); skip files already present and valid. **Do not** download q5/q6 files —
  they belong to Depth Anything V2, not DA3 (known upstream-repo trap).
- `run_cli.sh <in.jpg> [out.png] [f32|q8|q4]`: wraps `da3-cli` with model path resolution.
- `setup_tls.sh [extra-hosts…]`: port of `setup_depth_ui_tls.sh` — mkcert to `~/.local/bin`,
  local CA, issue into `resources/certs/`, copy `rootCA.pem` (public half only; CA key never
  leaves `~/.local/share/mkcert`).
- `da3.service.in`: `Type=simple`, `ExecStart=<REL>/scripts/start.sh --port 8090`,
  `Restart=on-failure`, placeholder `@USER@`/`@REL_ROOT@` + a sed one-liner in the header to
  instantiate. Referenced from deployment-guide.

---

## 8. Test plan

Python is permitted here (tests/tooling exemption), but prefer `curl`+`jq`+shell where easy.
Tests must run against a server started on an **ephemeral high port** (e.g. 18090) so they never
collide with a production instance. Test image: the engine submodule's
`assets/samples/desk.jpg` — no separate fixture needed.

### 8.1 Reproducibility test (the gate that matters most)

From a scratch directory: `git clone $REL da3-fresh && cd da3-fresh && scripts/setup.sh &&
scripts/start_all.sh && tests/smoke_test.sh && scripts/stop_all.sh`.
This exercises submodule URLs, patch application on virgin checkouts, model download, build, and
run — end to end. (If the sandbox blocks network: clone submodules from the local mirrors with
`git -c protocol.file.allow=always clone --recurse-submodules`, document the substitution, and
flag it for the final inspection to re-run with network.)

### 8.2 `tests/smoke_test.sh` — API contract (automated, must pass)

| # | Check | Expectation |
|---|---|---|
| 1 | `GET /health` | 200; JSON has `ok:true` and `variants.{f32,q8,q4}.{downloaded,loaded}` |
| 2 | `POST /depth?variant=q8` (desk.jpg body) | 200; `Content-Type: image/jpeg`; body starts `FFD8`; headers `X-Variant:q8`, `X-Depth-Width/Height` > 0, `X-Timings-Ms` parses as JSON with keys save/model_load/preprocess/infer/encode/server |
| 3 | Same for `f32`, `q4` | as above; `X-Variant` echoes |
| 4 | `?format=json` | 200 `application/json`; `depth_png` base64-decodes to a PNG (`\x89PNG`); width/height match headers from #2 |
| 5 | `?res=full` | 200; `X-Depth-Width` ≈ input res (~1022) ≫ std (~504) |
| 6 | PNG input (convert desk.jpg once with any tool, or a tiny committed PNG) | 200 |
| 7 | `?variant=zzz` | 400 + JSON `error` naming valid variants |
| 8 | Empty body | 400 |
| 9 | Garbage body (e.g. 1 KB of `/dev/urandom`) | 500 + JSON `error` |
| 10 | Warm-cache: repeat #2, parse `X-Timings-Ms` | `model_load_ms == 0` |
| 11 | **api-only instance** (second server, `--api-only`, another port) | `/depth`+`/health` work; `/` returns the JSON banner, not HTML |
| 12 | UI serving (normal instance) | `GET /` returns HTML containing `app.js` reference; `GET /js/app.js` returns 200 non-empty |
| 13 | Concurrency: 4 parallel `POST`s | all 200 (mutex serializes, none error) |
| 14 | `--tls` instance | `curl -k https://…/health` 200 |

Script exits nonzero on first failure with the failing request printed. Start/stop its own server
instances (foreground + `&`, kill by PID — not pkill).

### 8.3 `tests/parity_test.sh` — custom-kernel numeric parity (automated, must pass)

1. Fetch depth via `?format=json` twice against two server runs (or `da3-cli` runs):
   `DA_CONV=cuda_fused` (default) vs `DA_CONV=im2col` (stock ggml), same image, q8, std res.
2. Decode both PNGs; assert **mean abs diff ≤ 1.0/255 and max abs diff ≤ 3/255** (u8 domain —
   generous vs the measured max-rel 1.03e-3, tight enough to catch a broken kernel).
3. Also assert both runs' `X-Gpu-Preprocess: true` (GPU path actually engaged, not fallback).

### 8.4 Script/option matrix (automated where cheap, else scripted-manual)

| Case | Verify |
|---|---|
| `start.sh --daemon` → `--status` → `--stop` | pidfile lifecycle; log file created; stop removes pidfile |
| `start_all.sh` twice | second call is a no-op "already running" |
| `stop_all.sh` with nothing running | exit 0, clear message |
| `start.sh --no-prewarm` | `/health` immediately after start shows `loaded:false`; first request slower, still 200 |
| `start.sh --conv im2col` | serves correctly (perf checked in §9, not here) |
| `build.sh` second run | incremental, fast, no reconfigure errors |
| `apply_patches.sh` second run | "already applied", exit 0 |
| `download_models.sh` re-run | skips existing files |

### 8.5 `tests/MANUAL_UI_CHECKLIST.md` (human, for final inspection)

Page loads; drag-drop JPG → side-by-side result + timing table; variant dropdown switches;
render styles (Grayscale/Turbo/Viridis/Inferno/Relief 3D) re-render instantly with no new
inference; high-res checkbox works (slower, denser); live camera starts on localhost (or via
`--lan` from a phone: cert accept → camera permission → HUD shows fps/drop count; flip camera).
Record browser + device used.

### 8.6 Upstream engine tests (best-effort, non-blocking)

The engine ships a ctest suite (`tests/test_capi*`, backbone tests…). After the patched build,
run `ctest` in the build tree; record pass/fail/skipped in the completion notes. Many need
model/reference assets — skips are acceptable; **regressions caused by our patches are not**
(the patches were developed against this exact suite, so failures = application problem).

---

## 9. Performance measurement plan

### 9.1 `scripts/bench.sh` — the harness

```
bench.sh [--variant q8] [--n 20] [--warmup 3] [--res std|full] [--port 8090] [--image PATH]
```

- Requires a running server (or `--spawn` to manage its own on an ephemeral port).
- Sends `--warmup` requests (discarded), then `--n` requests; captures `X-Timings-Ms` from each.
- Reports per stage (save/model_load/preprocess/infer/encode/server): **median, min, max, p95**,
  plus derived sequential fps (1000/median-server). jq/awk math is fine.
- `--json` flag emits machine-readable output → used to append dated entries to
  `docs/perf-baselines.md`.

### 9.2 Reference numbers & acceptance gates

On reference-class hardware (RTX 5060 8 GB, CUDA 12.8, desk.jpg, q8, std res, warm):

| Metric | Reference | Gate (fail → investigate before completing) |
|---|---|---|
| server total median | ~36.0 ms | **≤ 40 ms** |
| infer median | ~33.5 ms | ≤ 38 ms |
| preprocess median (JPEG) | ~2.2 ms | ≤ 4 ms and `X-Gpu-Preprocess: true` |
| encode median | ~0.3 ms | ≤ 1 ms |
| first request after start (prewarm on) | ~44 ms | ≤ 80 ms |
| `--conv im2col` server median | ~52.5 ms | ≤ 60 ms (and **slower** than cuda_fused — else kernels aren't engaging) |
| `?res=full` server median | ~800 ms | informational only |
| f32 vs q8 vs q4 warm infer | near-identical | spread ≤ 15 % (quantization must NOT change warm GPU speed — a big spread signals a broken path) |

The cuda_fused-vs-im2col comparison doubles as proof that the ggml patch + custom kernels are
actually live in the release build — treat a "no speedup" result as a packaging bug, not a perf
miss.

On any *other* GPU: no absolute gates; record the full table as the host's baseline in
`docs/perf-baselines.md` (GPU, driver, CUDA, date, git SHA, all stage medians).

### 9.3 GPU-path verification (perf-adjacent sanity, from the original playbook)

- During a 30-request loop: `nvidia-smi dmon -s u -d 1` must show SM% bursts ≥ 30 (the
  long-window `utilization.gpu` counter is misleading for sub-second bursts — don't use it).
- Startup log contains `ggml_cuda_init: found … CUDA devices` and `da::Backend using device: CUDA0`.
- `nvidia-smi` VRAM with all three variants loaded ≤ ~1.5 GB.
- Deep profiling knobs to document in perf-baselines.md preamble (not gates):
  `DA_PROFILE=1` (per-stage engine timings), `DA_CUDA_HEAD_STATS=1` (per-shape kernel table),
  `da3-cli --repeat 20` (server-independent warm medians).

### 9.4 `docs/perf-baselines.md`

Seed it with: the reference table above (marked "gemma4 dev box, 2026-07"), the §9.3 toolbox
notes, and the first `bench.sh --json` run from the release build. Every future entry appends —
this file is the longitudinal record that regressions get caught against.

---

## 10. Final-inspection acceptance checklist (definition of done)

Structure & hygiene:
- [ ] Layout matches §2; `1stparty/` removed; no empty placeholder files remain.
- [ ] `git status` in `$REL` clean except the (documented) patched-submodule gitlink; no `.gguf`,
      no `certs/`, no `build/`, no `var/` tracked; `.gitignore` covers them.
- [ ] `.gitmodules` references the public upstream URL only.
- [ ] Patches in `scripts/patches/` regenerate the exact local state: submodule
      `git log f4e17de..HEAD` shows 3 commits w/ original authorship; ggml shows 1; **no
      Subproject-commit hunks in any patch file**.
- [ ] `3rdparty/README.md` provenance table complete (incl. httplib version + licenses).
- [ ] Meaningful commit history in `$REL` (per phase, not one blob).

Function:
- [ ] §8.1 fresh-clone flow passes end-to-end.
- [ ] `smoke_test.sh` all 14 checks green; `parity_test.sh` green; §8.4 matrix green.
- [ ] Backend-only mode: `start_server.sh` serves API without UI.
- [ ] Relocatability: `mv` the repo (or clone to a second path) → `start_all.sh` works **without rebuild**.
- [ ] systemd unit template instantiates and runs (if the box allows; else dry-run documented).

Performance:
- [ ] `bench.sh` gates of §9.2 met on the dev box; baseline appended to perf-baselines.md.
- [ ] cuda_fused measurably faster than im2col (kernel patches proven live).

Docs:
- [ ] All copied docs path-correct for the release tree (grep for `tools/depth_ui`, `gemma4`,
      `models/` must return nothing unintended).
- [ ] `docs/plans/` = gemm-rewrite-plan.md + future-work.md; README quickstart works as written
      (execute it literally).

## 11. Suggested execution order

1. Scaffolding: dirs, `.gitignore`, delete `1stparty/` → commit.
2. Patches: generate (§4.2), submodule add (§4.3), `apply_patches.sh`, verify → commit.
3. Vendor httplib + `3rdparty/README.md` → commit.
4. Port first-party code + code changes (§6.1–6.3) → commit.
5. Top-level CMake + `build.sh`; build until green → commit.
6. Remaining scripts (§7) → commit.
7. `download_models.sh` + models fetched (files themselves stay untracked) → commit.
8. Docs migration (§6.5) → commit.
9. Tests (§8.2–8.4) written & passing → commit.
10. Bench + baselines (§9) → commit.
11. Fresh-clone run (§8.1); fix fallout → final commit. Leave tagging to the final inspection.

Out of scope for this effort: implementing anything in `docs/plans/` (GEMM rewrite, metric/nested
models), multi-GPU/batching, auth/rate-limiting, packaging beyond the repo itself (no .deb/docker),
and any change to the inference math or API contract.
