#!/bin/bash
# bench.sh - Performance measurement harness
#
# Usage: ./bench.sh [options]
#   --variant q8       Model variant (default: q8)
#   --n 20             Number of requests (default: 20)
#   --warmup 3         Warmup requests (default: 3)
#   --res std          Resolution: std|full (default: std)
#   --port 8090        Server port (default: 8090)
#   --image PATH       Test image (default: desk.jpg from engine)
#   --json             Output machine-readable JSON
#   --spawn            Spawn server automatically if not running

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REL_ROOT"

VARIANT="q8"
N=20
WARMUP=3
RES="std"
PORT=8090
IMAGE="$REL_ROOT/3rdparty/depth-anything.cpp/assets/samples/desk.jpg"
JSON_OUTPUT=0
SPAWN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    --n)
      N="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --res)
      RES="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --image)
      IMAGE="$2"
      shift 2
      ;;
    --json)
      JSON_OUTPUT=1
      shift
      ;;
    --spawn)
      SPAWN=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check server running
if ! curl -s "http://localhost:$PORT/health" &>/dev/null; then
  if [ "$SPAWN" -eq 1 ]; then
    echo "Server not running, spawning on port $PORT..."
    "$REL_ROOT/build/depth-ui-server" "$PORT" &
    SPAWN_PID=$!
    sleep 3
  else
    echo "ERROR: Server not running on port $PORT"
    echo "Use --spawn to auto-spawn, or start with ./scripts/start.sh"
    exit 1
  fi
fi

# Warmup
echo "Running $WARMUP warmup requests..."
for i in $(seq 1 $WARMUP); do
  curl -s -X POST --data-binary @"$IMAGE" \
    -H "Content-Type: image/jpeg" \
    "http://localhost:$PORT/depth?variant=$VARIANT&res=$RES" &>/dev/null
done

# Benchmark
echo "Running $N benchmark requests..."
TIMINGS=()
for i in $(seq 1 $N); do
  RESPONSE=$(curl -s -w "\n%{http_code}" -X POST --data-binary @"$IMAGE" \
    -H "Content-Type: image/jpeg" \
    "http://localhost:$PORT/depth?variant=$VARIANT&res=$RES")
  HTTP_CODE=$(echo "$RESPONSE" | tail -1)
  BODY=$(echo "$RESPONSE" | head -n -1)
  
  if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: Request $i failed with HTTP $HTTP_CODE"
    exit 1
  fi
  
  # Extract timings from header or JSON
  TIMING=$(echo "$BODY" | grep -o '"server_ms":[0-9.]*' | cut -d: -f2)
  if [ -n "$TIMING" ]; then
    TIMINGS+=("$TIMING")
  fi
done

# Calculate stats
echo ""
echo "=== Results ==="
echo "Variant: $VARIANT, Resolution: $RES, N: $N"
echo ""

# Use awk for statistics
STATS=$(printf '%s\n' "${TIMINGS[@]}" | awk '
{
  vals[NR] = $1
  sum += $1
  if (NR == 1 || $1 < min) min = $1
  if (NR == 1 || $1 > max) max = $1
}
END {
  n = NR
  median_idx = int((n + 1) / 2)
  p95_idx = int(n * 0.95)
  if (p95_idx < 1) p95_idx = 1
  
  # Sort for percentiles
  for (i = 1; i <= n; i++) {
    for (j = i + 1; j <= n; j++) {
      if (vals[i] > vals[j]) {
        tmp = vals[i]
        vals[i] = vals[j]
        vals[j] = tmp
      }
    }
  }
  
  median = vals[median_idx]
  p95 = vals[p95_idx]
  avg = sum / n
  
  printf "%.1f %.1f %.1f %.1f %.1f\n", median, min, max, p95, avg
}')

read MEDIAN MIN MAX P95 AVG <<< "$STATS"

echo "Server total (ms):"
echo "  Median: $MEDIAN"
echo "  Min:    $MIN"
echo "  Max:    $MAX"
echo "  P95:    $P95"
echo "  Avg:    $AVG"
echo ""
echo "Derived fps: $(awk "BEGIN {printf \"%.1f\", 1000 / $MEDIAN}")"

if [ "$JSON_OUTPUT" -eq 1 ]; then
  echo ""
  echo "=== JSON Output ==="
  cat <<EOF
{
  "date": "$(date -Iseconds)",
  "variant": "$VARIANT",
  "resolution": "$RES",
  "n": $N,
  "warmup": $WARMUP,
  "server_ms": {
    "median": $MEDIAN,
    "min": $MIN,
    "max": $MAX,
    "p95": $P95,
    "avg": $AVG
  },
  "fps": $(awk "BEGIN {printf \"%.1f\", 1000 / $MEDIAN}")
}
EOF
fi

# Cleanup spawned server
if [ "${SPAWN_PID:-}" != "" ]; then
  kill "$SPAWN_PID" 2>/dev/null || true
fi
