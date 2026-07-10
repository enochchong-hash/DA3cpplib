#!/bin/bash
# setup_tls.sh - Set up TLS certificates using mkcert
#
# Usage: ./setup_tls.sh [extra-hosts...]
#   Generates a local CA and server certificate with SANs for localhost
#   and any additional hosts/IPs provided.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REL_ROOT"

CERT_DIR="$REL_ROOT/resources/certs"
MKCERT_DIR="$HOME/.local/share/mkcert"

# Install mkcert if needed
if ! command -v mkcert &>/dev/null; then
  echo "Installing mkcert..."
  mkdir -p "$HOME/.local/bin"
  MKCERT_URL="https://github.com/FiloSottile/mkcert/releases/download/v1.4.4/mkcert-v1.4.4-linux-amd64"
  curl -L -o "$HOME/.local/bin/mkcert" "$MKCERT_URL"
  chmod +x "$HOME/.local/bin/mkcert"
  export PATH="$HOME/.local/bin:$PATH"
  echo "  ✓ mkcert installed to $HOME/.local/bin/mkcert"
fi

# Initialize CA if needed
if [ ! -d "$MKCERT_DIR" ]; then
  echo "Initializing local CA..."
  mkdir -p "$MKCERT_DIR"
  mkcert -install
  echo "  ✓ CA installed"
fi

# Generate server certificate
mkdir -p "$CERT_DIR"

HOSTS=("localhost" "127.0.0.1")

# Add LAN IP
LAN_IP=$(ip route | grep default | awk '{print $9}' | head -1)
if [ -n "$LAN_IP" ]; then
  HOSTS+=("$LAN_IP")
fi

# Add extra hosts from args
HOSTS+=("${@:-}")

echo "Generating certificate for: ${HOSTS[*]}"

# Remove old cert if exists
rm -f "$CERT_DIR/server.crt" "$CERT_DIR/server.key"

mkcert -cert-file "$CERT_DIR/server.crt" -key-file "$CERT_DIR/server.key" "${HOSTS[@]}"

echo ""
echo "✓ TLS certificates generated:"
echo "  Certificate: $CERT_DIR/server.crt"
echo "  Key:         $CERT_DIR/server.key"
echo ""
echo "To use with the server:"
echo "  export DEPTH_UI_CERT=$CERT_DIR/server.crt"
echo "  export DEPTH_UI_KEY=$CERT_DIR/server.key"
echo "  ./scripts/start.sh --tls"
