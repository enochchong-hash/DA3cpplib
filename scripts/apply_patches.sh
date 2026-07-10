#!/bin/bash
# apply_patches.sh - Apply local patches to depth-anything.cpp and ggml submodules
#
# Usage: ./apply_patches.sh
#
# This script:
# 1. Initializes submodules if needed
# 2. Applies ggml patch first, then engine patches (engine code needs patched ggml)
# 3. Is idempotent - safe to run multiple times
#
# IMPORTANT: Running `git submodule update --checkout --force` will discard patches.
# Re-run this script after any submodule reset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PATCH_DIR="$REL_ROOT/scripts/patches"

cd "$REL_ROOT"

echo "=== Applying patches to submodules ==="

# Initialize submodules if needed
if [ ! -e "3rdparty/depth-anything.cpp/.git" ]; then
    echo "Initializing submodules..."
    git submodule update --init
fi
if [ ! -e "3rdparty/depth-anything.cpp/third_party/ggml/.git" ]; then
    echo "Initializing nested ggml submodule..."
    git -C 3rdparty/depth-anything.cpp submodule update --init --recursive
fi

# apply_patch_set <name> <repo_dir> <upstream_pin> <patch_dir>
#
# Counts how many of the patch subjects already exist as commits above the
# upstream pin: all -> skip (idempotent), none -> apply, some -> abort
# (partial state needs a human). No HEAD-vs-pin comparison needed.
apply_patch_set() {
    local name="$1" repo_dir="$2" pin="$3" patch_dir="$4"
    local patches=("$patch_dir"/*.patch)

    if [ ! -f "${patches[0]}" ]; then
        echo "ERROR: no patch files in $patch_dir"
        return 1
    fi

    local applied_subjects found=0 total=0 subject
    applied_subjects=$(git -C "$repo_dir" log --format='%s' "$pin..HEAD")
    for patch in "${patches[@]}"; do
        total=$((total + 1))
        subject=$(grep -m1 '^Subject:' "$patch" | sed 's/^Subject: \(\[PATCH[^]]*\] \)\?//')
        if grep -qxF "$subject" <<< "$applied_subjects"; then
            found=$((found + 1))
        fi
    done

    if [ "$found" -eq "$total" ]; then
        echo "  $name: all $total patch(es) already applied"
        return 0
    fi
    if [ "$found" -gt 0 ]; then
        echo "ERROR: $name has $found of $total patches applied - partial state."
        echo "  Inspect: git -C $repo_dir log --oneline $pin..HEAD"
        echo "  To reset to upstream: git -C $repo_dir reset --hard $pin  (then re-run)"
        return 1
    fi

    # --ignore-submodules: a patched nested submodule (ggml) legitimately moves
    # its gitlink; only real file modifications should block git am.
    if ! git -C "$repo_dir" diff --quiet --ignore-submodules=all \
       || ! git -C "$repo_dir" diff --cached --quiet --ignore-submodules=all; then
        echo "ERROR: $name working tree is dirty. Commit or stash changes first."
        return 1
    fi

    echo "  $name: applying $total patch(es)..."
    if ! git -C "$repo_dir" -c user.name="da3-release" -c user.email="da3-release@local" \
            am --3way "${patches[@]}"; then
        echo "ERROR: git am failed for $name (see message above)."
        echo "  Clean up with: git -C $repo_dir am --abort"
        return 1
    fi
    echo "  ✓ $name patches applied"
}

echo ""
GGML_DIR="$REL_ROOT/3rdparty/depth-anything.cpp/third_party/ggml"
ENGINE_DIR="$REL_ROOT/3rdparty/depth-anything.cpp"

apply_patch_set "ggml"   "$GGML_DIR"   3af5f57 "$PATCH_DIR/ggml"
apply_patch_set "engine" "$ENGINE_DIR" f4e17de "$PATCH_DIR/depth-anything.cpp"

echo ""
echo "=== Patch application complete ==="
echo ""
echo "Final state:"
echo "  ggml:   $(git -C "$GGML_DIR" log --oneline --format='%h %s' -1)"
echo "  engine: $(git -C "$ENGINE_DIR" log --oneline --format='%h %s' -1)"
echo ""
echo "Note: the submodule gitlink shows as modified in the release repo after"
echo "patching - expected; the patches sit on top of the upstream-pinned SHAs."
