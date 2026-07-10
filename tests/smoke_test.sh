#!/bin/bash
# smoke_test.sh - API contract tests
#
# Usage: ./smoke_test.sh
#   Starts test servers on ephemeral ports (18090/18093), runs all tests,
#   stops them. Exit 0 = all pass; exits 1 at the first failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TEST_PORT=18090
API_PORT=18093
TEST_IMG="$REL_ROOT/3rdparty/depth-anything.cpp/assets/samples/desk.jpg"
PNG_IMG="$REL_ROOT/3rdparty/depth-anything.cpp/assets/localai_logo.png"
TMPDIR_T=$(mktemp -d)
TEST_PID=""
API_PID=""

cleanup() {
  [ -n "$TEST_PID" ] && kill "$TEST_PID" 2>/dev/null || true
  [ -n "$API_PID" ]  && kill "$API_PID"  2>/dev/null || true
  rm -rf "$TMPDIR_T"
}
trap cleanup EXIT

fail() { echo "FAIL: $1"; exit 1; }
pass() { echo "PASS: $1"; }

wait_health() {  # wait_health <port> <pid>
  for i in {1..40}; do
    curl -s "http://localhost:$1/health" &>/dev/null && return 0
    kill -0 "$2" 2>/dev/null || fail "server on port $1 died during startup"
    sleep 0.5
  done
  fail "server on port $1 never became healthy"
}

# post_depth <url-suffix> <body-file|-> -> writes $TMPDIR_T/{body,hdrs}, echoes http code
post_depth() {
  local data_arg=(--data-binary @"$2")
  [ "$2" = "-" ] && data_arg=()
  curl -s -o "$TMPDIR_T/body" -D "$TMPDIR_T/hdrs" -w '%{http_code}' -X POST \
    "${data_arg[@]}" -H "Content-Type: image/jpeg" \
    "http://localhost:$TEST_PORT/depth$1"
}

echo "Starting test server on port $TEST_PORT..."
DA_CONV="${DA_CONV:-cuda_fused}" "$REL_ROOT/build/depth-ui-server" "$TEST_PORT" \
  > "$TMPDIR_T/server.log" 2>&1 &
TEST_PID=$!
wait_health "$TEST_PORT" "$TEST_PID"

echo ""
echo "=== Test 1: GET /health ==="
HEALTH=$(curl -s "http://localhost:$TEST_PORT/health")
echo "$HEALTH" | grep -q '"ok":true' || fail "/health ok:true"
for v in f32 q8 q4; do
  echo "$HEALTH" | grep -q "\"$v\"" || fail "/health missing variant $v"
done
pass "GET /health"

echo ""
echo "=== Test 2: POST /depth?variant=q8 (binary JPEG contract) ==="
CODE=$(post_depth "?variant=q8" "$TEST_IMG")
[ "$CODE" = "200" ] || fail "HTTP status: expected 200, got $CODE"
head -c 2 "$TMPDIR_T/body" | od -A n -t x1 | grep -q "ff d8" || fail "body is not JPEG (FFD8)"
grep -qi '^x-variant: *q8' "$TMPDIR_T/hdrs" || fail "X-Variant header"
grep -qi '^x-timings-ms:' "$TMPDIR_T/hdrs" || fail "X-Timings-Ms header"
for k in save_ms model_load_ms infer_ms encode_ms server_ms; do
  grep -i '^x-timings-ms:' "$TMPDIR_T/hdrs" | grep -q "\"$k\"" || fail "timings key $k"
done
pass "POST /depth?variant=q8"

echo ""
echo "=== Test 3: All variants ==="
for v in f32 q8 q4; do
  CODE=$(post_depth "?variant=$v" "$TEST_IMG")
  [ "$CODE" = "200" ] || fail "variant $v: HTTP $CODE"
done
pass "All variants (f32, q8, q4)"

echo ""
echo "=== Test 4: warm cache (model_load_ms == 0 on repeat) ==="
CODE=$(post_depth "?variant=q8" "$TEST_IMG")
grep -i '^x-timings-ms:' "$TMPDIR_T/hdrs" | grep -q '"model_load_ms":0' \
  || fail "expected model_load_ms:0 on warm request"
pass "Warm cache"

echo ""
echo "=== Test 5: format=json (lossless PNG contract) ==="
JSON_RESP=$(curl -s -X POST --data-binary @"$TEST_IMG" \
  -H "Content-Type: image/jpeg" "http://localhost:$TEST_PORT/depth?variant=q8&format=json")
echo "$JSON_RESP" | grep -q '"depth_png"' || fail "JSON: depth_png field"
echo "$JSON_RESP" | grep -q '"timings_ms"' || fail "JSON: timings_ms field"
echo "$JSON_RESP" | python3 -c '
import sys, json, base64
d = json.load(sys.stdin)
png = base64.b64decode(d["depth_png"])
assert png[:4] == b"\x89PNG", "not a PNG"
assert d["width"] > 0 and d["height"] > 0, "bad dimensions"
' || fail "JSON: depth_png does not decode to a PNG"
pass "format=json"

echo ""
echo "=== Test 6: res=full (near-input resolution) ==="
CODE=$(post_depth "?variant=q8&res=full" "$TEST_IMG")
[ "$CODE" = "200" ] || fail "res=full: HTTP $CODE"
W=$(grep -i '^x-depth-width:' "$TMPDIR_T/hdrs" | tr -dc '0-9')
[ "${W:-0}" -gt 700 ] || fail "res=full width $W not > 700 (std would be ~504)"
pass "res=full (width $W)"

echo ""
echo "=== Test 7: PNG input ==="
CODE=$(curl -s -o "$TMPDIR_T/body" -D "$TMPDIR_T/hdrs" -w '%{http_code}' -X POST \
  --data-binary @"$PNG_IMG" -H "Content-Type: image/png" \
  "http://localhost:$TEST_PORT/depth?variant=q8")
[ "$CODE" = "200" ] || fail "PNG input: HTTP $CODE"
pass "PNG input"

echo ""
echo "=== Test 8: invalid variant -> 400 ==="
CODE=$(post_depth "?variant=zzz" "$TEST_IMG")
[ "$CODE" = "400" ] || fail "invalid variant: expected 400, got $CODE"
grep -q '"error"' "$TMPDIR_T/body" || fail "invalid variant: no error field"
pass "Invalid variant returns 400"

echo ""
echo "=== Test 9: empty body -> 400 ==="
CODE=$(post_depth "?variant=q8" "-")
[ "$CODE" = "400" ] || fail "empty body: expected 400, got $CODE"
pass "Empty body returns 400"

echo ""
echo "=== Test 10: garbage body -> 4xx/5xx with error ==="
head -c 1024 /dev/urandom > "$TMPDIR_T/garbage.bin"
CODE=$(post_depth "?variant=q8" "$TMPDIR_T/garbage.bin")
[ "$CODE" -ge 400 ] || fail "garbage body: expected error status, got $CODE"
grep -q '"error"' "$TMPDIR_T/body" || fail "garbage body: no error field"
pass "Garbage body returns $CODE"

echo ""
echo "=== Test 11: UI serving (normal mode) ==="
UI_HTML=$(curl -s "http://localhost:$TEST_PORT/")
echo "$UI_HTML" | grep -q "app.js" || fail "UI: no app.js reference in /"
UI_JS_CODE=$(curl -s -o "$TMPDIR_T/appjs" -w '%{http_code}' "http://localhost:$TEST_PORT/js/app.js")
[ "$UI_JS_CODE" = "200" ] || fail "UI: /js/app.js HTTP $UI_JS_CODE"
[ -s "$TMPDIR_T/appjs" ] || fail "UI: /js/app.js empty"
pass "UI serving"

echo ""
echo "=== Test 12: API-only mode ==="
DA3_API_ONLY=1 DEPTH_UI_PREWARM=off "$REL_ROOT/build/depth-ui-server" "$API_PORT" \
  > "$TMPDIR_T/api_server.log" 2>&1 &
API_PID=$!
wait_health "$API_PORT" "$API_PID"
API_ROOT=$(curl -s "http://localhost:$API_PORT/")
echo "$API_ROOT" | grep -q '"service":"da3"' || fail "API-only: root should be JSON banner, got: $(echo "$API_ROOT" | head -c 80)"
API_CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST --data-binary @"$TEST_IMG" \
  -H "Content-Type: image/jpeg" "http://localhost:$API_PORT/depth?variant=q8")
[ "$API_CODE" = "200" ] || fail "API-only: /depth HTTP $API_CODE"
kill "$API_PID" 2>/dev/null; wait "$API_PID" 2>/dev/null || true; API_PID=""
pass "API-only mode"

echo ""
echo "=== Test 13: concurrency (4 parallel requests) ==="
CURL_PIDS=()
for i in 1 2 3 4; do
  curl -s -o /dev/null -w '%{http_code}\n' -X POST --data-binary @"$TEST_IMG" \
    -H "Content-Type: image/jpeg" \
    "http://localhost:$TEST_PORT/depth?variant=q8" >> "$TMPDIR_T/codes" &
  CURL_PIDS+=($!)
done
wait "${CURL_PIDS[@]}"
[ "$(grep -c '^200$' "$TMPDIR_T/codes")" -eq 4 ] || fail "concurrency: $(cat "$TMPDIR_T/codes" | tr '\n' ' ')"
pass "4 parallel requests all 200"

kill "$TEST_PID" 2>/dev/null; wait "$TEST_PID" 2>/dev/null || true; TEST_PID=""

echo ""
echo "=== All smoke tests passed ==="
