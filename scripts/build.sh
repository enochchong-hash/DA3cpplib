#!/bin/bash
# build.sh - Configure and build the DA3 depth-estimation service
#
# Usage: ./build.sh [options]
#   --arch native|<num>  CUDA architectures (default: auto-detect host GPU)
#   --jobs <num>          Parallel build jobs (default: min(CPU cores, 4))
#   --debug              Build in Debug mode
#   --clean              Clean build directory before building
#   --help               Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REL_ROOT"

# Prefer /usr/local/cuda* over any PATH nvcc (Ubuntu's apt package is 11.5).
source "$SCRIPT_DIR/cuda_env.sh"

ARCH="native"
BUILD_TYPE="Release"
CLEAN=0
CPU_CORES="$(nproc)"
if [ "$CPU_CORES" -gt 4 ]; then
  JOBS=4
else
  JOBS="$CPU_CORES"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      ARCH="$2"
      shift 2
      ;;
    --debug)
      BUILD_TYPE="Debug"
      shift
      ;;
    --jobs)
      if [ "$#" -lt 2 ] || [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: --jobs requires a positive integer." >&2
        exit 1
      fi
      JOBS="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --help)
      echo "Usage: ./build.sh [options]"
      echo "  --arch native|<num>  CUDA architectures (default: auto-detect host GPU)"
      echo "  --jobs <num>          Parallel build jobs (default: min(CPU cores, 4))"
      echo "  --debug              Build in Debug mode"
      echo "  --clean              Clean build directory before building"
      echo "  --help               Show this help"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ "$ARCH" = "native" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Error: cannot auto-detect the CUDA architecture: nvidia-smi was not found." >&2
    echo "Install a working NVIDIA driver or pass --arch <num> (for example, --arch 75)." >&2
    exit 1
  fi

  COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | sed -n '1p' | tr -d '[:space:].' || true)"
  if [[ ! "$COMPUTE_CAP" =~ ^[0-9]+$ ]]; then
    echo "Error: cannot auto-detect the CUDA architecture from nvidia-smi." >&2
    echo "Check that the NVIDIA driver is working or pass --arch <num> explicitly." >&2
    exit 1
  fi

  ARCH="$COMPUTE_CAP"
  echo "Detected CUDA architecture: $ARCH"
fi

BUILD_DIR="$REL_ROOT/build"

if [ "$CLEAN" -eq 1 ]; then
  echo "Cleaning build directory..."
  rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring with CMake (build type: $BUILD_TYPE, CUDA arch: $ARCH)..."
GGML_MMQ=OFF
if [[ "$ARCH" =~ ^7[5-9]$ ]]; then
  GGML_MMQ=ON
  echo "Enabling GGML_CUDA_FORCE_MMQ for Turing compute capability $ARCH"
fi
cmake .. \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DDA_GGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FORCE_MMQ="$GGML_MMQ"

echo "Building with $JOBS parallel job(s)..."
cmake --build . -j"$JOBS"

echo ""
echo "=== Build complete ==="
echo "Artifacts:"
echo "  Server:   $BUILD_DIR/depth-ui-server"
echo "  CLI tool: $BUILD_DIR/3rdparty/depth-anything.cpp/examples/cli/da3-cli"
echo ""
echo "To run: ./scripts/start.sh"
