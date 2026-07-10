#!/bin/bash
# open_firewall.sh - Allow incoming connections to the DA3 port
#
# Usage: sudo-capable user runs:  ./scripts/open_firewall.sh [port]
#   (default port: 8090)
#
# Opens the port in every ACTIVE firewall. Some hosts (like the original
# reference box) run BOTH firewalld and ufw simultaneously - the port must be
# open in both or LAN devices time out. From the host itself everything works
# even with the port closed, so "works locally, phone times out" = firewall.

set -euo pipefail

PORT="${1:-8090}"
OPENED=0

if systemctl is-active --quiet firewalld 2>/dev/null; then
  echo "firewalld active - opening ${PORT}/tcp..."
  sudo firewall-cmd --add-port="${PORT}/tcp" --permanent
  sudo firewall-cmd --reload
  OPENED=1
fi

if systemctl is-active --quiet ufw 2>/dev/null; then
  echo "ufw active - opening ${PORT}/tcp..."
  sudo ufw allow "${PORT}/tcp"
  OPENED=1
fi

if [ "$OPENED" -eq 0 ]; then
  echo "No active firewall (firewalld/ufw) detected - nothing to do."
  echo "If LAN devices still can't connect, check your router/AP client isolation."
  exit 0
fi

echo ""
echo "Done. Verify from another device on the LAN:"
LAN_IP=$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' || echo "<this-pc-ip>")
echo "  curl -sk https://${LAN_IP}:${PORT}/health"
