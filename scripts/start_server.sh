#!/bin/bash
# start_server.sh - Thin alias for backend-only mode
#
# Usage: ./start_server.sh [start.sh options]
#   Starts the server in API-only mode (no web UI).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/start.sh" --api-only "$@"
