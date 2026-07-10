# Depth Anything 3 Release — README

**DA3** is a pure-C++ (no Python in the serving path) local depth-estimation service based on [mudler/depth-anything.cpp](https://github.com/mudler/depth-anything.cpp). It provides a persistent HTTP server with GPU-accelerated inference, GPU JPEG encode/decode, and a self-contained web UI for live camera depth estimation.

## Quickstart

```sh
# 1. Setup (one-time): prereqs → submodules → patches → models → build
./scripts/setup.sh

# 2. Start the server (default: LAN + HTTPS, so phone cameras work;
#    use --local for localhost-only plain HTTP)
./scripts/start_all.sh

# 2b. First time only, to accept connections from other devices (phones):
sudo ./scripts/open_firewall.sh

# 3. Check health
curl -k https://localhost:8090/health

# 4. Open the web UI
# https://<this-pc-lan-ip>:8090   (from a phone; accept the cert warning once)
# https://localhost:8090          (on this machine)
```

## What it does

| Component | Description |
|-----------|-------------|
| Inference engine | `3rdparty/depth-anything.cpp` — GGML/CUDA backend, custom DPT head kernels |
| HTTP server | `src/server.cpp` — Persistent model residency, GPU preprocess (nvJPEG), GPU JPEG encode |
| Web UI | `resources/html/index.html` — Upload, live camera, render styles (Grayscale/Turbo/Viridis/Inferno/Relief 3D) |
| Models | `resources/nnmodels/` — GGUF files (downloaded by `scripts/download_models.sh`) |

## Prerequisites

- **Hardware**: NVIDIA GPU with CUDA 12.8+ (RTX 5060 or better recommended)
- **Software**:
  - CUDA Toolkit 12.8+ (`nvcc`, `nvidia-smi`)
  - CMake 3.18+
  - g++ with C++17 support
  - OpenSSL development headers (optional, for TLS)

## Layout

```
release/da3/
├── CMakeLists.txt              # Top-level build
├── README.md                   # This file
├── 3rdparty/                   # Third-party code (quarantined)
│   ├── depth-anything.cpp/     # Engine submodule @ upstream f4e17de + patches
│   └── cpp-httplib/            # Vendored httplib 0.48.0
├── src/                        # First-party C++
│   ├── server.cpp              # HTTP server
│   └── gpu_preprocess.cu       # CUDA nvJPEG/resize kernels
├── include/
│   └── gpu_preprocess.h
├── resources/
│   ├── html/                   # Web UI markup
│   ├── js/                     # Web UI JavaScript
│   └── nnmodels/               # GGUF models (gitignored)
├── scripts/
│   ├── setup.sh                # One-time setup
│   ├── build.sh                # Build
│   ├── start.sh                # Granular launcher
│   ├── start_all.sh            # Start everything
│   └── ...                     # See scripts/ for more
├── docs/                       # Documentation
└── tests/                      # Test scripts
```

## Usage

### Starting the server

```sh
# Full server with UI, LAN + HTTPS (default; phone camera ready)
./scripts/start_all.sh

# Localhost-only plain HTTP
./scripts/start_all.sh --local

# Backend only (no UI)
./scripts/start_server.sh --port 8090

# Granular control (host/port/TLS/prewarm/kernels)
./scripts/start.sh --help
```

### API endpoints

- `GET /health` — Health check
- `POST /depth?variant=q8` — Depth inference (raw image bytes → grayscale JPEG)
- `GET /` — Web UI (or JSON banner in `--api-only` mode)

See [`docs/user-guide.md`](docs/user-guide.md) for full API documentation.

### Model variants

| Variant | Size | Use case |
|---------|------|----------|
| `f32` | 412 MB | Reference quality |
| `q8` | 149 MB | Near-lossless, faster load |
| `q4` | 104 MB | Smallest, visually indistinguishable |

Download models: `./scripts/download_models.sh [f32|q8|q4|all]`

## Performance

**Reference** (RTX 5060 8 GB, CUDA 12.8, desk.jpg 1024×673, q8, std res):

| Metric | Value |
|--------|-------|
| Server total (median) | ~36.0 ms |
| Inference | ~33.5 ms |
| GPU preprocess | ~2.2 ms |
| GPU JPEG encode | ~0.3 ms |

See [`docs/performance-optimization.md`](docs/performance-optimization.md) for details.

## Submodule & patches

This repo uses the upstream engine as a submodule with local patches applied:

- `3rdparty/depth-anything.cpp` — Pinned to upstream `f4e17de` + 3 local commits
- `scripts/patches/depth-anything.cpp/` — Patch files (0001–0003)
- `scripts/apply_patches.sh` — Apply patches idempotently

**Note**: The submodule gitlink will show as dirty after patching — this is expected.

## License

- **First-party code**: MIT
- **Third-party**: See `3rdparty/README.md` (engine MIT, ggml MIT, cpp-httplib MIT, stb public domain/MIT)

---

For deployment instructions, see [`docs/deployment-guide.md`](docs/deployment-guide.md).  
For setup troubleshooting, see [`docs/setup-development-guide.md`](docs/setup-development-guide.md).
