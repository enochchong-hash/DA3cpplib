#!/bin/bash
# apply_patches.sh - Apply local patches to depth-anything.cpp and ggml submodules
#
# Usage: ./apply_patches.sh
#
# This script:
# 1. Initializes submodules if needed
# 2. Applies ggml patch first, then engine patches
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
if [ ! -d "3rdparty/depth-anything.cpp/.git" ]; then
    echo "Initializing submodules..."
    git submodule update --init
    git -C 3rdparty/depth-anything.cpp submodule update --init --recursive
fi

# Function to check if patches are already applied
check_patches_applied() {
    local repo_dir="$1"
    local pin="$2"
    local patch_dir="$3"
    
    cd "$repo_dir"
    
    # Get list of patch subjects
    local applied_commits
    applied_commits=$(git log --format="%s" "$pin..HEAD" 2>/dev/null || echo "")
    
    for patch in "$patch_dir"/*.patch; do
        [ -f "$patch" ] || continue
        local subject
        subject=$(grep "^Subject:" "$patch" | sed 's/Subject: \[PATCH\] //')
        
        if echo "$applied_commits" | grep -qF "$subject"; then
            echo "  ✓ Already applied: $subject"
        else
            return 1
        fi
    done
    
    cd "$REL_ROOT"
    return 0
}

# Apply ggml patch first
echo ""
echo "Applying ggml patch..."
GGML_DIR="3rdparty/depth-anything.cpp/third_party/ggml"
GGML_PATCH_DIR="$PATCH_DIR/ggml"
GGML_PIN="3af5f57"

cd "$GGML_DIR"
current_commit=$(git rev-parse HEAD)
if [ "$current_commit" = "$GGML_PIN" ]; then
    echo "  ggml at upstream pin $GGML_PIN"
elif ! check_patches_applied "$GGML_DIR" "$GGML_PIN" "$GGML_PATCH_DIR" 2>/dev/null; then
    # Clean tree check
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "ERROR: ggml working tree is dirty. Commit or stash changes first."
        exit 1
    fi
    
    echo "  Applying ggml patches..."
    git -c user.name="da3-release" -c user.email="da3-release@local" am --3way "$GGML_DIR"/*.patch 2>/dev/null || {
        echo "ERROR: Failed to apply ggml patches. Manual intervention required."
        exit 1
    }
    echo "  ✓ ggml patches applied"
else
    echo "  ggml patches already applied"
fi
echo "  Current ggml state: $(git log --oneline -1)"
cd "$REL_ROOT"

# Apply engine patches
echo ""
echo "Applying depth-anything.cpp patches..."
ENGINE_DIR="3rdparty/depth-anything.cpp"
ENGINE_PATCH_DIR="$PATCH_DIR/depth-anything.cpp"
ENGINE_PIN="f4e17de"

cd "$ENGINE_DIR"
current_commit=$(git rev-parse HEAD)
if [ "$current_commit" = "$ENGINE_PIN" ]; then
    echo "  Engine at upstream pin $ENGINE_PIN"
elif ! check_patches_applied "$ENGINE_DIR" "$ENGINE_PIN" "$ENGINE_PATCH_DIR" 2>/dev/null; then
    # Clean tree check
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "ERROR: Engine working tree is dirty. Commit or stash changes first."
        exit 1
    fi
    
    echo "  Applying engine patches..."
    git -c user.name="da3-release" -c user.email="da3-release@local" am --3way "$ENGINE_PATCH_DIR"/*.patch 2>/dev/null || {
        echo "ERROR: Failed to apply engine patches. Manual intervention required."
        exit 1
    }
    echo "  ✓ Engine patches applied"
else
    echo "  Engine patches already applied"
fi
echo "  Current engine state: $(git log --oneline -1)"
cd "$REL_ROOT"

echo ""
echo "=== Patch application complete ==="
echo ""
echo "Final state:"
echo "  ggml:     $(git -C $GGML_DIR log --oneline -1 --format='%H %s')"
echo "  engine:   $(git -C $ENGINE_DIR log --oneline -1 --format='%H %s')"
echo ""
echo "Note: The submodule gitlink will show as dirty in the release repo."
echo "This is expected - the patches modify the upstream-pinned submodules."
