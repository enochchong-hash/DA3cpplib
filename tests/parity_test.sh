#!/bin/bash
# parity_test.sh - Custom-kernel numeric parity test
#
# Runs two servers (DA_CONV=cuda_fused vs DA_CONV=im2col), fetches the
# lossless depth PNG from each for the same image, and compares pixels.
# Thresholds (u8 levels, 0-255 domain): mean abs diff <= 1.0, max <= 3.
# Python/PIL is used for the comparison (tests are exempt from the
# no-Python-in-serving-path rule).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CUDA_FUSED_PORT=18091
IM2COL_PORT=18092
TEST_IMG="$REL_ROOT/3rdparty/depth-anything.cpp/assets/samples/desk.jpg"
TMPDIR_T=$(mktemp -d)
PID_A=""
PID_B=""

python3 -c "import PIL" 2>/dev/null || {
  echo "ERROR: python3-PIL (pillow) is required for the pixel comparison."
  echo "Install: pip3 install --user pillow   (or apt install python3-pil)"
  exit 1
}

cleanup() {
  [ -n "$PID_A" ] && kill "$PID_A" 2>/dev/null || true
  [ -n "$PID_B" ] && kill "$PID_B" 2>/dev/null || true
  rm -rf "$TMPDIR_T"
}
trap cleanup EXIT

wait_health() {  # wait_health <port> <pid>
  for i in {1..40}; do
    curl -s "http://localhost:$1/health" &>/dev/null && return 0
    kill -0 "$2" 2>/dev/null || { echo "ERROR: server on port $1 died"; exit 1; }
    sleep 0.5
  done
  echo "ERROR: server on port $1 never became healthy"; exit 1
}

fetch_png() {  # fetch_png <port> <out.png>
  curl -s -X POST --data-binary @"$TEST_IMG" -H "Content-Type: image/jpeg" \
    "http://localhost:$1/depth?variant=q8&format=json" \
    | python3 -c "import sys,json,base64; sys.stdout.buffer.write(base64.b64decode(json.load(sys.stdin)['depth_png']))" \
    > "$2"
  [ -s "$2" ] || { echo "ERROR: empty depth PNG from port $1"; exit 1; }
}

echo "Starting cuda_fused server on port $CUDA_FUSED_PORT..."
DA_CONV=cuda_fused DEPTH_UI_PREWARM=off "$REL_ROOT/build/depth-ui-server" \
  "$CUDA_FUSED_PORT" > "$TMPDIR_T/cuda.log" 2>&1 &
PID_A=$!
wait_health "$CUDA_FUSED_PORT" "$PID_A"

echo "Starting im2col server on port $IM2COL_PORT..."
DA_CONV=im2col DEPTH_UI_PREWARM=off "$REL_ROOT/build/depth-ui-server" \
  "$IM2COL_PORT" > "$TMPDIR_T/im2col.log" 2>&1 &
PID_B=$!
wait_health "$IM2COL_PORT" "$PID_B"

echo ""
echo "Fetching depth from cuda_fused..."
fetch_png "$CUDA_FUSED_PORT" "$TMPDIR_T/cuda_depth.png"
echo "Fetching depth from im2col..."
fetch_png "$IM2COL_PORT" "$TMPDIR_T/im2col_depth.png"

echo ""
echo "Comparing outputs..."
python3 - "$TMPDIR_T/cuda_depth.png" "$TMPDIR_T/im2col_depth.png" << 'EOF'
from PIL import Image
import sys

cuda = Image.open(sys.argv[1]).convert('L')
im2col = Image.open(sys.argv[2]).convert('L')

if cuda.size != im2col.size:
    print(f"FAIL: size mismatch {cuda.size} vs {im2col.size}")
    sys.exit(1)

diffs = [abs(a - b) for a, b in zip(cuda.getdata(), im2col.getdata())]
mean_diff = sum(diffs) / len(diffs)
max_diff = max(diffs)

# Thresholds are in u8 levels (0-255 domain)
print(f"Mean abs diff: {mean_diff:.4f} u8 levels (threshold: 1.0)")
print(f"Max abs diff:  {max_diff} u8 levels (threshold: 3)")

if mean_diff > 1.0:
    print(f"FAIL: mean diff {mean_diff:.4f} > 1.0 u8 levels")
    sys.exit(1)
if max_diff > 3:
    print(f"FAIL: max diff {max_diff} > 3 u8 levels")
    sys.exit(1)

print("PASS: outputs are within tolerance")
EOF

echo ""
echo "=== Parity test passed ==="
