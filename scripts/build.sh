#!/bin/bash
# build.sh - Configure and build the DA3 depth-estimation service
#
# Usage: ./build.sh [options]
#   --arch native|<num>  CUDA architectures (default: native)
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
    --clean)
      CLEAN=1
      shift
      ;;
    --help)
      echo "Usage: ./build.sh [options]"
      echo "  --arch native|<num>  CUDA architectures (default: native)"
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

BUILD_DIR="$REL_ROOT/build"

if [ "$CLEAN" -eq 1 ]; then
  echo "Cleaning build directory..."
  rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring with CMake (build type: $BUILD_TYPE, CUDA arch: $ARCH)..."
cmake .. \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DDA_GGML_CUDA=ON \
  -DGGML_CUDA_GRAPHS=ON

echo "Building..."
cmake --build . -j"$(nproc)"

echo ""
echo "=== Build complete ==="
echo "Artifacts:"
echo "  Server:   $BUILD_DIR/depth-ui-server"
echo "  CLI tool: $BUILD_DIR/3rdparty/depth-anything.cpp/examples/cli/da3-cli"
echo ""
echo "To run: ./scripts/start.sh"
