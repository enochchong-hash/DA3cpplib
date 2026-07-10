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

# ver_ge A B -> true if version A >= B (handles 3-part versions, no bc needed)
ver_ge() { [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]; }

# CUDA - prefer /usr/local/cuda* over whatever nvcc is on PATH (Ubuntu's
# 'nvidia-cuda-toolkit' apt package ships nvcc 11.5 at /usr/bin/nvcc and
# shadows the real toolkit). cuda_env.sh fixes PATH and sets DA3_CUDA_HOME.
source "$SCRIPT_DIR/cuda_env.sh"

cuda_usable() {
  command -v nvcc &>/dev/null || return 1
  CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9.]+')
  ver_ge "$CUDA_VER" 12.8
}

if ! cuda_usable; then
  echo "  WARNING: no CUDA 12.8+ toolkit found (checked /usr/local/cuda* and PATH${CUDA_VER:+; PATH has $CUDA_VER})."
  echo "           Note: Ubuntu's 'nvidia-cuda-toolkit' apt package is CUDA 11.5 - too old."
  echo "  Launching ./scripts/setup_cuda.sh to install CUDA 12.8 (needs sudo)..."
  "$SCRIPT_DIR/setup_cuda.sh"
  source "$SCRIPT_DIR/cuda_env.sh"
  if ! cuda_usable; then
    echo "ERROR: CUDA 12.8+ still not available. Install it, then re-run setup.sh."
    exit 1
  fi
fi
echo "  ✓ nvcc $CUDA_VER ($(command -v nvcc))"

# Warn (non-fatal) if an older nvcc elsewhere on PATH is being shadowed.
if [ -n "${DA3_CUDA_HOME:-}" ]; then
  OTHER_NVCC=$(type -ap nvcc | grep -v "^$DA3_CUDA_HOME/bin/" | head -n1 || true)
  if [ -n "$OTHER_NVCC" ]; then
    OTHER_VER=$("$OTHER_NVCC" --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo '?')
    if [ "$OTHER_VER" != "$CUDA_VER" ]; then
      echo "  WARNING: ignoring $OTHER_NVCC (CUDA $OTHER_VER) - using $DA3_CUDA_HOME."
      echo "           To remove it: ./scripts/setup_cuda.sh --remove-old"
    fi
  fi
fi

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
if ! ver_ge "$CMAKE_VER" 3.18; then
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
