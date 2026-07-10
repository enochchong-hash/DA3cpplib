#!/bin/bash
# start_all.sh - Start everything with sensible defaults
#
# Usage: ./start_all.sh
#   Starts the server on port 8090, daemonized, health-checked.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REL_ROOT"

"$SCRIPT_DIR/start.sh" --daemon

# Health check
echo "Waiting for server to be ready..."
for i in {1..30}; do
  if curl -s http://localhost:8090/health &>/dev/null; then
    echo ""
    echo "=== Server ready ==="
    echo "  Health: http://localhost:8090/health"
    echo "  UI:     http://localhost:8090"
    echo ""
    curl -s http://localhost:8090/health | head -c 500
    echo ""
    exit 0
  fi
  sleep 0.5
done

echo "ERROR: Server failed to become ready. Check var/log/da3-8090.log"
tail -20 "$REL_ROOT/var/log/da3-8090.log" 2>/dev/null || true
exit 1
