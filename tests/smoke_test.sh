#!/bin/bash
# smoke_test.sh - API contract tests
#
# Usage: ./smoke_test.sh
#   Starts a test server on ephemeral port, runs all tests, stops server.
#   Exit 0 = all tests pass, exit 1 = first failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REL_ROOT"

TEST_PORT=18090
TEST_PID=""
TEST_IMG="$REL_ROOT/3rdparty/depth-anything.cpp/assets/samples/desk.jpg"

cleanup() {
  if [ -n "$TEST_PID" ] && kill -0 "$TEST_PID" 2>/dev/null; then
    kill "$TEST_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Start test server
echo "Starting test server on port $TEST_PORT..."
"$REL_ROOT/build/depth-ui-server" "$TEST_PORT" &
TEST_PID=$!
sleep 2

# Wait for health
for i in {1..20}; do
  if curl -s "http://localhost:$TEST_PORT/health" &>/dev/null; then
    break
  fi
  sleep 0.5
done

fail() {
  echo "FAIL: $1"
  exit 1
}

pass() {
  echo "PASS: $1"
}

# Test 1: GET /health
echo ""
echo "=== Test 1: GET /health ==="
HEALTH=$(curl -s "http://localhost:$TEST_PORT/health")
echo "$HEALTH" | grep -q '"ok":true' || fail "/health ok:true"
echo "$HEALTH" | grep -q '"variants"' || fail "/health variants"
pass "GET /health"

# Test 2: POST /depth?variant=q8
echo ""
echo "=== Test 2: POST /depth?variant=q8 ==="
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST --data-binary @"$TEST_IMG" \
  -H "Content-Type: image/jpeg" "http://localhost:$TEST_PORT/depth?variant=q8")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)
[ "$HTTP_CODE" = "200" ] || fail "HTTP status: expected 200, got $HTTP_CODE"
echo "$BODY" | head -c 2 | od -A n -t x1 | grep -q "ffd8" || fail "Content: not JPEG (FFD8)"
pass "POST /depth?variant=q8"

# Test 3: All variants
echo ""
echo "=== Test 3: All variants ==="
for var in f32 q8 q4; do
  RESP=$(curl -s -w "\n%{http_code}" -X POST --data-binary @"$TEST_IMG" \
    -H "Content-Type: image/jpeg" "http://localhost:$TEST_PORT/depth?variant=$var")
  CODE=$(echo "$RESP" | tail -1)
  [ "$CODE" = "200" ] || fail "variant $var: HTTP $CODE"
done
pass "All variants (f32, q8, q4)"

# Test 4: format=json
echo ""
echo "=== Test 4: format=json ==="
JSON_RESP=$(curl -s -X POST --data-binary @"$TEST_IMG" \
  -H "Content-Type: image/jpeg" "http://localhost:$TEST_PORT/depth?variant=q8&format=json")
echo "$JSON_RESP" | grep -q '"depth_png"' || fail "JSON: depth_png field"
echo "$JSON_RESP" | grep -q '"timings_ms"' || fail "JSON: timings_ms field"
pass "format=json"

# Test 5: res=full
echo ""
echo "=== Test 5: res=full ==="
FULL_RESP=$(curl -s -w "\n%{http_code}" -X POST --data-binary @"$TEST_IMG" \
  -H "Content-Type: image/jpeg" "http://localhost:$TEST_PORT/depth?variant=q8&res=full")
FULL_CODE=$(echo "$FULL_RESP" | tail -1)
[ "$FULL_CODE" = "200" ] || fail "res=full: HTTP $FULL_CODE"
pass "res=full"

# Test 6: Invalid variant
echo ""
echo "=== Test 6: Invalid variant ==="
INVALID_RESP=$(curl -s -w "\n%{http_code}" -X POST --data-binary @"$TEST_IMG" \
  -H "Content-Type: image/jpeg" "http://localhost:$TEST_PORT/depth?variant=zzz")
INVALID_CODE=$(echo "$INVALID_RESP" | tail -1)
[ "$INVALID_CODE" = "400" ] || fail "Invalid variant: expected 400, got $INVALID_CODE"
echo "$INVALID_RESP" | grep -q '"error"' || fail "Invalid variant: no error field"
pass "Invalid variant returns 400"

# Test 7: Empty body
echo ""
echo "=== Test 7: Empty body ==="
EMPTY_RESP=$(curl -s -w "\n%{http_code}" -X POST -H "Content-Type: image/jpeg" \
  "http://localhost:$TEST_PORT/depth?variant=q8")
EMPTY_CODE=$(echo "$EMPTY_RESP" | tail -1)
[ "$EMPTY_CODE" = "400" ] || fail "Empty body: expected 400, got $EMPTY_CODE"
pass "Empty body returns 400"

# Test 8: API-only mode
echo ""
echo "=== Test 8: API-only mode ==="
"$REL_ROOT/build/depth-ui-server" "$((TEST_PORT + 1))" &
API_PID=$!
sleep 2
API_ROOT=$(curl -s "http://localhost:$((TEST_PORT + 1))/")
echo "$API_ROOT" | grep -q '"service":"da3"' || fail "API-only: root JSON banner"
kill "$API_PID" 2>/dev/null || true
pass "API-only mode"

# Test 9: UI serving (normal mode)
echo ""
echo "=== Test 9: UI serving ==="
UI_HTML=$(curl -s "http://localhost:$TEST_PORT/")
echo "$UI_HTML" | grep -q "app.js" || fail "UI: contains app.js reference"
UI_JS=$(curl -s "http://localhost:$TEST_PORT/js/app.js")
[ -n "$UI_JS" ] || fail "UI: /js/app.js returns content"
pass "UI serving"

# Test 10: Concurrency
echo ""
echo "=== Test 10: Concurrency ==="
for i in {1..4}; do
  curl -s -X POST --data-binary @"$TEST_IMG" \
    -H "Content-Type: image/jpeg" "http://localhost:$TEST_PORT/depth?variant=q8" &
done
wait
pass "4 parallel requests"

kill "$TEST_PID"
TEST_PID=""

echo ""
echo "=== All smoke tests passed ==="
