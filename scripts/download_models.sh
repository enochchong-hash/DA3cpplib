#!/bin/bash
# download_models.sh - Download GGUF models from HuggingFace
#
# Usage: ./download_models.sh [f32|q8|q4|all]
#   Downloads models into resources/nnmodels/
#   Verifies file size and GGUF magic bytes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REL_ROOT"

MODEL_DIR="$REL_ROOT/resources/nnmodels"
HF_BASE="https://huggingface.co/mudler/depth-anything.cpp-gguf/resolve/main"

mkdir -p "$MODEL_DIR"

download_and_verify() {
  local variant="$1"
  local filename="$2"
  local expected_size="$3"
  local dest="$MODEL_DIR/$variant/$filename"
  
  mkdir -p "$MODEL_DIR/$variant"
  
  if [ -f "$dest" ]; then
    local actual_size
    actual_size=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    local diff_pct
    diff_pct=$(echo "scale=2; 100 * ($actual_size - $expected_size) / $expected_size" | bc)
    if [[ "${diff_pct#-}" -le 5 ]]; then
      echo "  ✓ $variant/$filename already exists and valid (${actual_size} bytes)"
      return 0
    fi
    echo "  ! $variant/$filename exists but size mismatch, re-downloading..."
    rm -f "$dest"
  fi
  
  echo "  Downloading $variant/$filename..."
  curl -L --fail --continue-at - -o "$dest" "$HF_BASE/$filename"
  
  # Verify size
  local actual_size
  actual_size=$(stat -c%s "$dest")
  local diff_pct
  diff_pct=$(echo "scale=2; 100 * ($actual_size - $expected_size) / $expected_size" | bc)
  if [[ "${diff_pct#-}" -gt 5 ]]; then
    echo "ERROR: $variant/$filename size mismatch: expected ~$expected_size, got $actual_size"
    exit 1
  fi
  
  # Verify GGUF magic
  local magic
  magic=$(head -c4 "$dest" | od -A n -t x1 | tr -d ' \n')
  if [ "$magic" != "47475546" ]; then
    echo "ERROR: $variant/$filename is not a valid GGUF file (magic: $magic)"
    exit 1
  fi
  
  echo "  ✓ $variant/$filename downloaded and verified (${actual_size} bytes)"
}

case "${1:-all}" in
  f32)
    download_and_verify "DepthAnything-Base-F32" "depth-anything-base-f32.gguf" 412000000
    ;;
  q8)
    download_and_verify "DepthAnything-Base-Q8" "depth-anything-base-q8_0.gguf" 149000000
    ;;
  q4)
    download_and_verify "DepthAnything-Base-Q4" "depth-anything-base-q4_k.gguf" 104000000
    ;;
  all)
    download_and_verify "DepthAnything-Base-F32" "depth-anything-base-f32.gguf" 412000000
    download_and_verify "DepthAnything-Base-Q8" "depth-anything-base-q8_0.gguf" 149000000
    download_and_verify "DepthAnything-Base-Q4" "depth-anything-base-q4_k.gguf" 104000000
    ;;
  *)
    echo "Usage: $0 [f32|q8|q4|all]"
    exit 1
    ;;
esac

echo ""
echo "Models installed to $MODEL_DIR/"
