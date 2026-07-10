#!/bin/bash
# setup_cuda.sh - Install the CUDA 12.8 toolkit from NVIDIA's apt repository.
#
# Usage: ./scripts/setup_cuda.sh [--remove-old] [--yes]
#   --remove-old   Also remove Ubuntu's 'nvidia-cuda-toolkit' packages (CUDA
#                  11.5 at /usr/bin/nvcc) which shadow the real toolkit.
#   --yes          Non-interactive: assume yes for our prompts.
#
# Installs the TOOLKIT ONLY (compiler + libraries into /usr/local/cuda-12.8).
# It does NOT touch the NVIDIA driver - 'nvidia-smi' must already work.
# Needs sudo for apt operations. Idempotent: skips anything already in place.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REMOVE_OLD=0
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --remove-old) REMOVE_OLD=1 ;;
    --yes)        ASSUME_YES=1 ;;
    --help)       sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown option: $arg (see --help)"; exit 1 ;;
  esac
done

confirm() {  # confirm <question> -> 0 if yes
  [ "$ASSUME_YES" -eq 1 ] && return 0
  if [ ! -t 0 ]; then
    echo "  (non-interactive shell: skipping. Re-run with --yes to force.)"
    return 1
  fi
  read -r -p "$1 [Y/n] " reply
  [[ -z "$reply" || "$reply" =~ ^[Yy] ]]
}

echo "=== CUDA 12.8 toolkit setup ==="

# 1. Deal with Ubuntu's ancient apt toolkit (nvcc 11.5 at /usr/bin/nvcc).
#    Its -dev sibling drops CUDA 11.5 headers into /usr/include, which can
#    shadow the real toolkit's headers at build time - removal is recommended.
OLD_PKGS=$(dpkg -l 2>/dev/null | awk '/^ii  nvidia-cuda-(toolkit|toolkit-doc|dev|gdb)/ {print $2}' || true)
if [ -n "$OLD_PKGS" ]; then
  echo ""
  echo "WARNING: Ubuntu's CUDA 11.5 packages are installed and shadow /usr/local/cuda:"
  echo "$OLD_PKGS" | sed 's/^/    /'
  if [ "$REMOVE_OLD" -eq 1 ] || confirm "Remove them now (apt will show the full removal list)?"; then
    # No -y on purpose: let apt display exactly what gets removed and ask.
    sudo apt-get remove --purge $OLD_PKGS
    sudo apt-get autoremove
  else
    echo "  Keeping them. Builds still work: our scripts prefer /usr/local/cuda."
  fi
fi

# 2. Already have a usable toolkit? (cuda_env.sh prefers /usr/local/cuda*)
source "$SCRIPT_DIR/cuda_env.sh"
if command -v nvcc &>/dev/null; then
  CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9.]+')
  if [ "$(printf '%s\n' "$CUDA_VER" 12.8 | sort -V | head -n1)" = "12.8" ]; then
    echo ""
    echo "CUDA $CUDA_VER already installed at ${DA3_CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")} - nothing to install."
    exit 0
  fi
fi

# 3. Install from NVIDIA's apt repo (the documented "deb (network)" flow).
if [ "$(uname -m)" != "x86_64" ]; then
  echo "ERROR: this script only handles x86_64. For other platforms see:"
  echo "  https://developer.nvidia.com/cuda-downloads"
  exit 1
fi
. /etc/os-release
case "${ID}-${VERSION_ID}" in
  ubuntu-22.04) REPO=ubuntu2204 ;;
  ubuntu-24.04) REPO=ubuntu2404 ;;
  ubuntu-20.04) REPO=ubuntu2004 ;;
  *)
    echo "ERROR: unsupported distro '${ID} ${VERSION_ID}'. Install manually from:"
    echo "  https://developer.nvidia.com/cuda-downloads"
    exit 1
    ;;
esac

echo ""
echo "Installing CUDA 12.8 toolkit (repo: $REPO). This downloads ~3 GB and needs sudo."
confirm "Proceed?" || { echo "Aborted."; exit 1; }

if ! dpkg -s cuda-keyring &>/dev/null; then
  KEYRING_DEB="$(mktemp -d)/cuda-keyring_1.1-1_all.deb"
  curl -fL -o "$KEYRING_DEB" \
    "https://developer.download.nvidia.com/compute/cuda/repos/${REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
  sudo dpkg -i "$KEYRING_DEB"
fi
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8

# 4. Verify.
source "$SCRIPT_DIR/cuda_env.sh"
if ! command -v nvcc &>/dev/null; then
  echo "ERROR: install finished but nvcc not found under /usr/local/cuda*." ; exit 1
fi
echo ""
echo "Installed: $(nvcc --version | grep release) at ${DA3_CUDA_HOME:-?}"
echo "Done. Re-run ./scripts/setup.sh to continue."
