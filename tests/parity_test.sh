#!/bin/bash
# parity_test.sh - Custom kernel numeric parity test
#
# Usage: ./parity_test.sh
#   Compares cuda_fused vs im2col depth outputs for numeric parity.
#   Expects both to be within tolerance (mean abs diff ≤ 1.0/255, max ≤ 3/255).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REL_ROOT"

TEST_PORT=18091
TEST_IMG="$REL_ROOT/3rdparty/depth-anything.cpp/assets/samples/desk.jpg"
CUDA_FUSED_PORT=18091
IM2COL_PORT=18092

cleanup() {
  kill $(cat /tmp/parity_cuda.pid 2>/dev/null) 2>/dev/null || true
  kill $(cat /tmp/parity_im2col.pid 2>/dev/null) 2>/dev/null || true
  rm -f /tmp/parity_cuda.pid /tmp/parity_im2col.pid /tmp/cuda_depth.png /tmp/im2col_depth.png
}
trap cleanup EXIT

# Start cuda_fused server
echo "Starting cuda_fused server on port $CUDA_FUSED_PORT..."
DA_CONV=cuda_fused "$REL_ROOT/build/depth-ui-server" "$CUDA_FUSED_PORT" &
echo $! > /tmp/parity_cuda.pid
sleep 2

# Start im2col server
echo "Starting im2col server on port $IM2COL_PORT..."
DA_CONV=im2col "$REL_ROOT/build/depth-ui-server" "$IM2COL_PORT" &
echo $! > /tmp/parity_im2col.pid
sleep 2

# Get depth from each
echo ""
echo "Fetching depth from cuda_fused..."
curl -s -X POST --data-binary @"$TEST_IMG" \
  -H "Content-Type: image/jpeg" \
  "http://localhost:$CUDA_FUSED_PORT/depth?variant=q8&format=json" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['depth_png'])" \
  | base64 -d > /tmp/cuda_depth.png

echo "Fetching depth from im2col..."
curl -s -X POST --data-binary @"$TEST_IMG" \
  -H "Content-Type: image/jpeg" \
  "http://localhost:$IM2COL_PORT/depth?variant=q8&format=json" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['depth_png'])" \
  | base64 -d > /tmp/im2col_depth.png

# Compare
echo ""
echo "Comparing outputs..."

# Use Python for precise comparison
python3 << 'EOF'
from PIL import Image
import sys

cuda = Image.open('/tmp/cuda_depth.png').convert('L')
im2col = Image.open('/tmp/im2col_depth.png').convert('L')

if cuda.size != im2col.size:
    print(f"FAIL: Size mismatch {cuda.size} vs {im2col.size}")
    sys.exit(1)

cuda_data = list(cuda.getdata())
im2col_data = list(im2col.getdata())

if len(cuda_data) != len(im2col_data):
    print("FAIL: Pixel count mismatch")
    sys.exit(1)

diffs = [abs(a - b) for a, b in zip(cuda_data, im2col_data)]
mean_diff = sum(diffs) / len(diffs)
max_diff = max(diffs)

print(f"Mean abs diff: {mean_diff:.3f} (threshold: 1.0/255 = 0.0039)")
print(f"Max abs diff: {max_diff:.3f} (threshold: 3/255 = 0.0118)")

# Convert to 0-255 domain for threshold comparison
mean_diff_255 = mean_diff
max_diff_255 = max_diff

if mean_diff_255 > 1.0:
    print(f"FAIL: Mean diff {mean_diff_255:.3f} > 1.0")
    sys.exit(1)

if max_diff_255 > 3.0:
    print(f"FAIL: Max diff {max_diff_255:.3f} > 3.0")
    sys.exit(1)

print("PASS: Both within tolerance")
EOF

echo ""
echo "=== Parity test passed ==="
