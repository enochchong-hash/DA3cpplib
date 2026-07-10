#!/bin/bash
# run_cli.sh - Run da3-cli for one-shot depth inference
#
# Usage: ./run_cli.sh <input.jpg> [output.png] [variant]
#   variant: f32|q8|q4 (default: q8)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REL_ROOT"

BINARY="$REL_ROOT/build/da3-cli"
MODEL_DIR="$REL_ROOT/resources/nnmodels"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <input.jpg> [output.png] [variant]"
  echo "  variant: f32|q8|q4 (default: q8)"
  exit 1
fi

INPUT="$1"
OUTPUT="${2:-}"
VARIANT="${3:-q8}"

if [ ! -f "$BINARY" ]; then
  echo "ERROR: Binary not found: $BINARY"
  echo "Run: ./scripts/build.sh"
  exit 1
fi

# Resolve model path
MODEL="$MODEL_DIR/DepthAnything-Base-${VARIANT}/depth-anything-base-${VARIANT}.gguf"

# Handle variant naming
case "$VARIANT" in
  q8) MODEL_NAME="depth-anything-base-q8_0.gguf" ;;
  q4) MODEL_NAME="depth-anything-base-q4_k.gguf" ;;
  f32) MODEL_NAME="depth-anything-base-f32.gguf" ;;
  *)
    echo "ERROR: Unknown variant: $VARIANT (use f32|q8|q4)"
    exit 1
    ;;
esac
MODEL="$MODEL_DIR/DepthAnything-Base-${VARIANT}/${MODEL_NAME}"

if [ ! -f "$MODEL" ]; then
  echo "ERROR: Model not found: $MODEL"
  echo "Run: ./scripts/download_models.sh $VARIANT"
  exit 1
fi

if [ -n "$OUTPUT" ]; then
  "$BINARY" -m "$MODEL" -i "$INPUT" -o "$OUTPUT"
else
  "$BINARY" -m "$MODEL" -i "$INPUT"
fi
