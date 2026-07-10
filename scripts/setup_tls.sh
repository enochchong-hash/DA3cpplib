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

# Initialize CA if needed (key on the CA key file, not the directory - the
# directory existing without a CA inside would skip -install forever)
CAROOT="$(mkcert -CAROOT)"
if [ ! -f "$CAROOT/rootCA-key.pem" ]; then
  echo "Initializing local CA..."
  mkcert -install
  echo "  ✓ CA installed"
fi

# Generate server certificate
mkdir -p "$CERT_DIR"

HOSTS=("localhost" "127.0.0.1")

# Add LAN IP (SAN must match the address the client visits; CN is ignored)
LAN_IP=$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' || true)
if [ -n "$LAN_IP" ]; then
  HOSTS+=("$LAN_IP")
fi

# Add extra hosts from args (if any)
if [ $# -gt 0 ]; then
  HOSTS+=("$@")
fi

echo "Generating certificate for: ${HOSTS[*]}"

# Remove old cert if exists
rm -f "$CERT_DIR/server.crt" "$CERT_DIR/server.key"

mkcert -cert-file "$CERT_DIR/server.crt" -key-file "$CERT_DIR/server.key" "${HOSTS[@]}"

# Public half of the CA, for import on phones/other devices. The CA PRIVATE
# key stays in $CAROOT and must never be copied or committed.
cp "$CAROOT/rootCA.pem" "$CERT_DIR/rootCA.pem"

echo ""
echo "✓ TLS certificates generated:"
echo "  Certificate: $CERT_DIR/server.crt"
echo "  Key:         $CERT_DIR/server.key"
echo "  CA (public): $CERT_DIR/rootCA.pem  <- import this on each browsing device"
echo ""
echo "Then start the server with:  ./scripts/start.sh --lan"
echo "(--tls picks up server.crt/server.key automatically; certs are loaded"
echo " at startup only - restart the server after regenerating them)"
