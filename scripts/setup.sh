#!/bin/bash
# setup.sh - One-time setup: prereqs → patches → models → build
#
# Usage: ./setup.sh [model-variant]
#   model-variant: f32|q8|q4|all (default: all)
#   --no-models    Skip model download
#
# Idempotent: safe to run multiple times.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REL_ROOT"

echo "=== DA3 Setup ==="
echo ""

# Prerequisite checks
echo "Checking prerequisites..."

# CUDA
if ! command -v nvcc &>/dev/null; then
  echo "ERROR: nvcc not found. Install CUDA 12.8+ and ensure it's in PATH."
  exit 1
fi
CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9.]+')
if [[ "$(echo "$CUDA_VER < 12.8" | bc -l)" -eq 1 ]]; then
  echo "ERROR: CUDA $CUDA_VER detected. Need CUDA 12.8+."
  exit 1
fi
echo "  ✓ nvcc $CUDA_VER"

# nvidia-smi
if ! command -v nvidia-smi &>/dev/null; then
  echo "WARNING: nvidia-smi not found. NVIDIA driver may not be installed."
else
  echo "  ✓ nvidia-smi available"
fi

# cmake
if ! command -v cmake &>/dev/null; then
  echo "ERROR: cmake not found. Install cmake 3.18+."
  exit 1
fi
CMAKE_VER=$(cmake --version | head -1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+')
if [[ "$(echo "$CMAKE_VER < 3.18" | bc -l)" -eq 1 ]]; then
  echo "ERROR: CMake $CMAKE_VER detected. Need CMake 3.18+."
  exit 1
fi
echo "  ✓ cmake $CMAKE_VER"

# g++ C++17
if ! command -v g++ &>/dev/null; then
  echo "ERROR: g++ not found. Install g++ with C++17 support."
  exit 1
fi
echo "  ✓ g++ available"

echo ""

# Apply patches
echo "Applying patches..."
"$SCRIPT_DIR/apply_patches.sh"
echo ""

# Download models
MODEL_ARG="${1:-all}"
if [ "${2:-}" = "--no-models" ] || [ "$MODEL_ARG" = "--no-models" ]; then
  echo "Skipping model download (--no-models specified)."
else
  echo "Downloading models..."
  "$SCRIPT_DIR/download_models.sh" "$MODEL_ARG"
  echo ""
fi

# Build
echo "Building..."
"$SCRIPT_DIR/build.sh"
echo ""

echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Start the server: ./scripts/start_all.sh"
echo "  2. Check health:     curl http://localhost:8090/health"
echo "  3. Open UI:          http://localhost:8090"
