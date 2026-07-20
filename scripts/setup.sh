#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

git -C "$ROOT_DIR" submodule update --init --recursive

GGML_DIR="$ROOT_DIR/3rdparty/ggml"
PATCH="$SCRIPT_DIR/patches/ggml/0001-da3-custom-cuda-ops.patch"
if git -C "$GGML_DIR" apply --check "$PATCH" >/dev/null 2>&1; then
    git -C "$GGML_DIR" apply "$PATCH"
    echo "Applied da3cpplib's ggml CUDA custom-op extension."
elif git -C "$GGML_DIR" apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    echo "ggml CUDA custom-op extension is already applied."
else
    echo "Error: ggml is neither at the expected pin nor already patched." >&2
    exit 1
fi

echo "Setup complete. Run scripts/build.sh auto|cpu|cuda."
