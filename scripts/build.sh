#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND="${1:-auto}"
BUILD_DIR="${DA3CPP_BUILD_DIR:-$ROOT_DIR/build}"
JOBS="${DA3CPP_BUILD_JOBS:-$(nproc)}"

if [[ "$BACKEND" == auto ]]; then
    if command -v nvcc >/dev/null 2>&1; then BACKEND=cuda; else BACKEND=cpu; fi
fi
if [[ "$BACKEND" != cpu && "$BACKEND" != cuda ]]; then
    echo "usage: $0 [auto|cpu|cuda]" >&2
    exit 2
fi

CUDA=OFF
[[ "$BACKEND" == cuda ]] && CUDA=ON
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DDA3CPP_CUDA="$CUDA"
cmake --build "$BUILD_DIR" --parallel "$JOBS"
echo "Built da3cpplib ($BACKEND) in $BUILD_DIR"
