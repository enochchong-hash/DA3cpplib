#!/bin/bash
# start_all.sh - Start everything with sensible defaults
#
# DEFAULT = LAN + HTTPS (start.sh --lan): phone/tablet browsers only allow
# camera access on secure origins, so the UI is served over https and bound
# to all interfaces. Anyone on your LAN can reach the server while it runs.
#
# Usage: ./start_all.sh [--local] [extra start.sh options]
#   --local    localhost-only plain HTTP instead (no LAN, no TLS)
#
# Idempotent: reports "already running" if a healthy instance exists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE_ARGS=(--lan)
PORT=8090
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) MODE_ARGS=(); shift ;;
    --port)  PORT="$2"; ARGS+=(--port "$2"); shift 2 ;;
    *)       ARGS+=("$1"); shift ;;
  esac
done

"$SCRIPT_DIR/start.sh" --daemon ${MODE_ARGS[@]+"${MODE_ARGS[@]}"} ${ARGS[@]+"${ARGS[@]}"}

# Determine which scheme actually answers (an already-running instance may
# be either) and report usable URLs.
echo "Waiting for server to be ready..."
SCHEME=""
for i in {1..30}; do
  if curl -sk "https://127.0.0.1:${PORT}/health" &>/dev/null; then SCHEME="https"; break; fi
  if curl -s  "http://127.0.0.1:${PORT}/health"  &>/dev/null; then SCHEME="http";  break; fi
  sleep 0.5
done

if [ -z "$SCHEME" ]; then
  echo "ERROR: Server failed to become ready. Check var/log/da3-${PORT}.log"
  tail -20 "$REL_ROOT/var/log/da3-${PORT}.log" 2>/dev/null || true
  exit 1
fi

LAN_IP=$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' || true)

echo ""
echo "=== Server ready ==="
if [ "$SCHEME" = "https" ] && [ -n "$LAN_IP" ]; then
  echo "  Phone/LAN UI: https://${LAN_IP}:${PORT}   (accept the certificate warning once)"
fi
echo "  Local UI:     ${SCHEME}://127.0.0.1:${PORT}"
echo "  Health:       ${SCHEME}://127.0.0.1:${PORT}/health"
if [ "$SCHEME" = "https" ]; then
  echo ""
  echo "  NOTE: the server is reachable from your LAN (no auth) while it runs."
  echo "  Phone times out? Open the firewall once: ./scripts/open_firewall.sh (needs sudo)"
  echo "  Warning-free certs for repeated phone use: ./scripts/setup_tls.sh, then restart."
  echo "  Localhost-only plain HTTP instead: ./scripts/start_all.sh --local"
fi
echo ""
curl -sk "${SCHEME}://127.0.0.1:${PORT}/health" | head -c 500
echo ""
