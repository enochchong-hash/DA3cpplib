#!/bin/bash
# bench.sh - Performance measurement harness
#
# Usage: ./bench.sh [options]
#   --variant q8       Model variant (default: q8)
#   --n 20             Number of measured requests (default: 20)
#   --warmup 3         Warmup requests, discarded (default: 3)
#   --res std          Resolution: std|full (default: std)
#   --port 8090        Server port (default: 8090)
#   --image PATH       Test image (default: desk.jpg from engine assets)
#   --json             Output machine-readable JSON only (for perf-baselines.md)
#   --spawn            Spawn a server if none is running, kill it afterwards
#
# Timings come from the X-Timings-Ms response header (the default /depth
# response is binary JPEG; there is no JSON body to parse).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VARIANT="q8"
N=20
WARMUP=3
RES="std"
PORT=8090
IMAGE="$REL_ROOT/3rdparty/depth-anything.cpp/assets/samples/desk.jpg"
JSON_OUTPUT=0
SPAWN=0
SPAWN_PID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --n)       N="$2"; shift 2 ;;
    --warmup)  WARMUP="$2"; shift 2 ;;
    --res)     RES="$2"; shift 2 ;;
    --port)    PORT="$2"; shift 2 ;;
    --image)   IMAGE="$2"; shift 2 ;;
    --json)    JSON_OUTPUT=1; shift ;;
    --spawn)   SPAWN=1; shift ;;
    --help)    sed -n '2,16p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown option: $1 (see --help)"; exit 1 ;;
  esac
done

URL="http://localhost:$PORT/depth?variant=$VARIANT&res=$RES"

cleanup() {
  if [ -n "$SPAWN_PID" ]; then
    kill "$SPAWN_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

log() { [ "$JSON_OUTPUT" -eq 1 ] || echo "$@"; }

# Check server running
if ! curl -s "http://localhost:$PORT/health" &>/dev/null; then
  if [ "$SPAWN" -eq 1 ]; then
    log "Server not running, spawning on port $PORT..."
    # Bench the shipped configuration: fused kernels unless caller overrides.
    DA_CONV="${DA_CONV:-cuda_fused}" "$REL_ROOT/build/depth-ui-server" "$PORT" \
      > /dev/null 2>&1 &
    SPAWN_PID=$!
    for i in {1..30}; do
      curl -s "http://localhost:$PORT/health" &>/dev/null && break
      kill -0 "$SPAWN_PID" 2>/dev/null || { echo "ERROR: spawned server died"; exit 1; }
      sleep 0.5
    done
  else
    echo "ERROR: Server not running on port $PORT"
    echo "Use --spawn to auto-spawn, or start with ./scripts/start.sh"
    exit 1
  fi
fi

# One request -> one CSV line "save,model_load,preprocess,infer,encode,server"
measure_one() {
  local hdr code timings
  hdr=$(mktemp)
  code=$(curl -s -o /dev/null -D "$hdr" -w '%{http_code}' -X POST \
    --data-binary @"$IMAGE" -H "Content-Type: image/jpeg" "$URL")
  if [ "$code" != "200" ]; then
    rm -f "$hdr"; echo "ERROR: request failed with HTTP $code" >&2; return 1
  fi
  timings=$(grep -i '^x-timings-ms:' "$hdr" | sed 's/^[^:]*: *//' | tr -d '\r')
  rm -f "$hdr"
  if [ -z "$timings" ]; then
    echo "ERROR: no X-Timings-Ms header in response" >&2; return 1
  fi
  echo "$timings" | grep -oE '"(save|model_load|preprocess|infer|encode|server)_ms":[0-9.]+' \
    | cut -d: -f2 | paste -sd, -
}

log "Running $WARMUP warmup requests..."
for i in $(seq 1 "$WARMUP"); do
  curl -s -o /dev/null -X POST --data-binary @"$IMAGE" \
    -H "Content-Type: image/jpeg" "$URL"
done

log "Running $N benchmark requests..."
SAMPLES=$(mktemp)
for i in $(seq 1 "$N"); do
  measure_one >> "$SAMPLES"
done

# Per-stage stats: field order matches measure_one's grep order
# (save, model_load, preprocess, infer, encode, server)
STATS=$(awk -F, '
{
  for (f = 1; f <= NF; f++) vals[f, NR] = $f + 0
  n = NR
}
END {
  split("save model_load preprocess infer encode server", names, " ")
  for (f = 1; f <= 6; f++) {
    # insertion sort of column f
    for (i = 2; i <= n; i++) {
      v = vals[f, i]
      for (j = i - 1; j >= 1 && vals[f, j] > v; j--) vals[f, j + 1] = vals[f, j]
      vals[f, j + 1] = v
    }
    med = vals[f, int((n + 1) / 2)]
    p95i = int(n * 0.95); if (p95i < 1) p95i = 1
    printf "%s %.1f %.1f %.1f %.1f\n", names[f], med, vals[f, 1], vals[f, n], vals[f, p95i]
  }
}' "$SAMPLES")
rm -f "$SAMPLES"

SERVER_MEDIAN=$(echo "$STATS" | awk '$1 == "server" {print $2}')
FPS=$(awk "BEGIN {printf \"%.1f\", 1000 / $SERVER_MEDIAN}")

if [ "$JSON_OUTPUT" -eq 1 ]; then
  {
    echo "{"
    echo "  \"date\": \"$(date -Iseconds)\","
    echo "  \"variant\": \"$VARIANT\", \"resolution\": \"$RES\", \"n\": $N, \"warmup\": $WARMUP,"
    echo "  \"stages_ms\": {"
    echo "$STATS" | awk '{printf "    \"%s\": {\"median\": %s, \"min\": %s, \"max\": %s, \"p95\": %s},\n", $1, $2, $3, $4, $5}' | sed '$ s/,$//'
    echo "  },"
    echo "  \"fps\": $FPS"
    echo "}"
  }
else
  echo ""
  echo "=== Results (variant=$VARIANT res=$RES n=$N) ==="
  printf "%-12s %8s %8s %8s %8s\n" "stage" "median" "min" "max" "p95"
  echo "$STATS" | awk '{printf "%-12s %8s %8s %8s %8s\n", $1, $2, $3, $4, $5}'
  echo ""
  echo "Sequential fps (1000/server-median): $FPS"
fi
