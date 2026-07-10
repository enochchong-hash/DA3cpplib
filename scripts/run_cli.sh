#!/bin/bash
# run_cli.sh - Run da3-cli for one-shot depth inference (no HTTP server)
#
# Usage: ./run_cli.sh <input.jpg> [output.png] [variant]
#   variant: f32|q8|q4 (default: q8)
#   output defaults to <input>.depth.png

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BINARY="$REL_ROOT/build/3rdparty/depth-anything.cpp/examples/cli/da3-cli"
MODEL_DIR="${DA3_MODEL_DIR:-$REL_ROOT/resources/nnmodels}"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <input.jpg> [output.png] [variant]"
  echo "  variant: f32|q8|q4 (default: q8)"
  exit 1
fi

INPUT="$1"
OUTPUT="${2:-${1%.*}.depth.png}"
VARIANT="${3:-q8}"

if [ ! -x "$BINARY" ]; then
  echo "ERROR: Binary not found: $BINARY"
  echo "Run: ./scripts/build.sh"
  exit 1
fi

case "$VARIANT" in
  f32) MODEL_NAME="depth-anything-base-f32.gguf";  FOLDER="DepthAnything-Base-F32" ;;
  q8)  MODEL_NAME="depth-anything-base-q8_0.gguf"; FOLDER="DepthAnything-Base-Q8" ;;
  q4)  MODEL_NAME="depth-anything-base-q4_k.gguf"; FOLDER="DepthAnything-Base-Q4" ;;
  *)
    echo "ERROR: Unknown variant: $VARIANT (use f32|q8|q4)"
    exit 1
    ;;
esac
MODEL="$MODEL_DIR/$FOLDER/$MODEL_NAME"

if [ ! -f "$MODEL" ]; then
  echo "ERROR: Model not found: $MODEL"
  echo "Run: ./scripts/download_models.sh $VARIANT"
  exit 1
fi

exec "$BINARY" depth --model "$MODEL" --input "$INPUT" --png "$OUTPUT"
