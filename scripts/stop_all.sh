#!/bin/bash
# stop_all.sh - Stop all DA3 instances
#
# Usage: ./stop_all.sh
#   Stops all running instances using pidfiles and pkill -x (safe).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REL_ROOT"

STOPPED=0

# Stop via pidfiles
for pidfile in "$REL_ROOT/var/run"/da3-*.pid; do
  [ -f "$pidfile" ] || continue
  
  PORT=$(basename "$pidfile" | sed 's/da3-\([0-9]*\).pid/\1/')
  PID=$(cat "$pidfile")
  
  if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping instance on port $PORT (PID $PID)..."
    kill "$PID"
    for i in {1..10}; do
      if ! kill -0 "$PID" 2>/dev/null; then break; fi
      sleep 0.5
    done
    if kill -0 "$PID" 2>/dev/null; then
      echo "  Force killing..."
      kill -9 "$PID"
    fi
    STOPPED=1
  else
    echo "Removing stale pidfile: $pidfile"
  fi
  rm -f "$pidfile"
done

# Safe sweep (pkill -x matches exact binary name, not cmdline)
if pkill -x depth-ui-server 2>/dev/null; then
  echo "Stopped additional instances via pkill -x"
  STOPPED=1
fi

if [ "$STOPPED" -eq 0 ]; then
  echo "No instances running"
else
  echo "All instances stopped"
fi
