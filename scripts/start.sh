#!/bin/bash
# start.sh - Granular launcher for the DA3 server
#
# Usage: ./start.sh [options]
#   --port N            Listen port (default: 8090)
#   --host ADDR         Bind address (default: 127.0.0.1)
#   --api-only          Backend only: REST endpoints, no web UI
#   --tls               HTTPS; auto-generate self-signed cert if resources/certs/ empty
#   --lan               Shorthand for --host 0.0.0.0 --tls (phone-camera mode)
#   --model-dir PATH    Override model directory (default: resources/nnmodels)
#   --conv MODE         cuda_fused (default) | im2col — A/B custom kernels
#   --no-prewarm        Skip startup model load + warm-up
#   --daemon            Background: nohup, pidfile, log
#   --status            Report running instances and exit
#   --stop              Stop the instance and exit
#   --help              Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REL_ROOT"

PORT=8090
HOST="127.0.0.1"
API_ONLY=0
TLS=0
LAN=0
MODEL_DIR=""
CONV_MODE=""
NO_PREWARM=0
DAEMON=0
STATUS=0
STOP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --api-only)
      API_ONLY=1
      shift
      ;;
    --tls)
      TLS=1
      shift
      ;;
    --lan)
      LAN=1
      shift
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --conv)
      CONV_MODE="$2"
      shift 2
      ;;
    --no-prewarm)
      NO_PREWARM=1
      shift
      ;;
    --daemon)
      DAEMON=1
      shift
      ;;
    --status)
      STATUS=1
      shift
      ;;
    --stop)
      STOP=1
      shift
      ;;
    --help)
      echo "Usage: ./start.sh [options]"
      echo "  --port N            Listen port (default: 8090)"
      echo "  --host ADDR         Bind address (default: 127.0.0.1)"
      echo "  --api-only          Backend only: REST endpoints, no web UI"
      echo "  --tls               HTTPS; auto-generate self-signed cert"
      echo "  --lan               Shorthand for --host 0.0.0.0 --tls"
      echo "  --model-dir PATH    Override model directory"
      echo "  --conv MODE         cuda_fused (default) | im2col"
      echo "  --no-prewarm        Skip startup model load + warm-up"
      echo "  --daemon            Background: nohup, pidfile, log"
      echo "  --status            Report running instances and exit"
      echo "  --stop              Stop the instance and exit"
      echo "  --help              Show this help"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Handle --lan shorthand
if [ "$LAN" -eq 1 ]; then
  HOST="0.0.0.0"
  TLS=1
fi

PIDFILE="$REL_ROOT/var/run/da3-${PORT}.pid"
LOGFILE="$REL_ROOT/var/log/da3-${PORT}.log"
BINARY="$REL_ROOT/build/depth-ui-server"
mkdir -p "$REL_ROOT/var/run" "$REL_ROOT/var/log"

# Probe /health over http, then https (instance may be running with --tls).
health_ok() {
  curl -s "http://127.0.0.1:${1}/health" &>/dev/null \
    || curl -sk "https://127.0.0.1:${1}/health" &>/dev/null
}

# Status check
if [ "$STATUS" -eq 1 ]; then
  echo "Checking instance on port $PORT..."
  if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
      echo "  Running (PID $PID)"
      if health_ok "$PORT"; then
        echo "  Health: OK"
      else
        echo "  Health: FAILED"
      fi
    else
      echo "  Stale pidfile (process not running)"
      rm -f "$PIDFILE"
    fi
  else
    echo "  Not running (no pidfile)"
  fi
  exit 0
fi

# Stop
if [ "$STOP" -eq 1 ]; then
  echo "Stopping instance on port $PORT..."
  if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
      kill "$PID"
      for i in {1..10}; do
        if ! kill -0 "$PID" 2>/dev/null; then break; fi
        sleep 0.5
      done
      if kill -0 "$PID" 2>/dev/null; then
        echo "  Force killing..."
        kill -9 "$PID"
      fi
      echo "  Stopped (PID $PID)"
    else
      echo "  Process not running"
    fi
    rm -f "$PIDFILE"
  else
    echo "  No pidfile found"
  fi
  # Deliberately NO pkill sweep here: --stop is scoped to this port only.
  # To stop every instance (any port), use scripts/stop_all.sh.
  exit 0
fi

# Validate binary exists
if [ ! -f "$BINARY" ]; then
  echo "ERROR: Binary not found: $BINARY"
  echo "Run: ./scripts/build.sh"
  exit 1
fi

# Validate at least one model exists
MODEL_DIR="${MODEL_DIR:-$REL_ROOT/resources/nnmodels}"
if ! find "$MODEL_DIR" -name "*.gguf" -type f | grep -q .; then
  echo "ERROR: No models found in $MODEL_DIR"
  echo "Run: ./scripts/download_models.sh"
  exit 1
fi

# TLS setup
if [ "$TLS" -eq 1 ]; then
  CERT_DIR="$REL_ROOT/resources/certs"
  CERT_FILE="$CERT_DIR/server.crt"
  KEY_FILE="$CERT_DIR/server.key"
  
  if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
    echo "Generating self-signed TLS certificate..."
    mkdir -p "$CERT_DIR"
    
    # Browsers require the SAN (not CN) to match the address being visited,
    # so the cert must carry this host's LAN IP or phones reject it outright.
    LAN_IP=$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' || true)
    SAN="DNS:localhost,IP:127.0.0.1${LAN_IP:+,IP:$LAN_IP}"

    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
      -keyout "$KEY_FILE" \
      -out "$CERT_FILE" \
      -subj "/CN=depth-ui" \
      -addext "subjectAltName=$SAN" 2>/dev/null
    echo "  SAN: $SAN  (if the LAN IP changes, delete resources/certs/ and relaunch)"
    
    echo "  Certificate: $CERT_FILE"
    echo "  Key: $KEY_FILE"
  fi
  
  export DEPTH_UI_CERT="$CERT_FILE"
  export DEPTH_UI_KEY="$KEY_FILE"
fi

# Export environment
export DA3_MODEL_DIR="$MODEL_DIR"
export DA3_WWW_DIR="$REL_ROOT/resources"
export DEPTH_UI_HOST="$HOST"
export DEPTH_UI_PORT="$PORT"

if [ "$API_ONLY" -eq 1 ]; then
  export DA3_API_ONLY=1
fi

if [ "$NO_PREWARM" -eq 1 ]; then
  export DEPTH_UI_PREWARM=off
fi

# Fused DPT-head CUDA kernels are the default (~36 vs ~52 ms server total);
# --conv im2col A/Bs against the stock ggml path. Precedence: --conv flag,
# then pre-set DA_CONV env, then cuda_fused.
if [ -n "$CONV_MODE" ]; then
  export DA_CONV="$CONV_MODE"
else
  export DA_CONV="${DA_CONV:-cuda_fused}"
fi

if [ "$DAEMON" -eq 1 ]; then
  # Idempotent: if this port already has a live, healthy instance, do nothing.
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    if health_ok "$PORT"; then
      echo "Already running on port $PORT (PID $(cat "$PIDFILE")) - nothing to do."
      exit 0
    fi
    echo "ERROR: PID $(cat "$PIDFILE") is alive but /health fails on port $PORT."
    echo "Inspect it, or stop it first: ./scripts/start.sh --port $PORT --stop"
    exit 1
  fi

  echo "Starting daemon on ${HOST}:${PORT}..."
  nohup "$BINARY" "$PORT" > "$LOGFILE" 2>&1 &
  PID=$!
  echo "$PID" > "$PIDFILE"

  # Wait for health; bail out early if the process died (port taken, bad model dir...)
  for i in {1..30}; do
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "ERROR: Server exited during startup. Last log lines:"
      tail -5 "$LOGFILE" 2>/dev/null || true
      rm -f "$PIDFILE"
      exit 1
    fi
    if health_ok "$PORT"; then
      echo "  Started (PID $PID)"
      echo "  Log: $LOGFILE"
      exit 0
    fi
    sleep 0.5
  done

  echo "ERROR: Server failed to become healthy. Check $LOGFILE"
  rm -f "$PIDFILE"
  exit 1
else
  echo "Starting server on ${HOST}:${PORT}..."
  echo "  Press Ctrl+C to stop"
  exec "$BINARY" "$PORT"
fi
