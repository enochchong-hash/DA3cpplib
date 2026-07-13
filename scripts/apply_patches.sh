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
ENGINE_UPSTREAM="${DA3_DEPTH_ANYTHING_UPSTREAM:-f4e17dea695dd12ae76bea98ba58030996b98118}"
GGML_UPSTREAM="3af5f5760e19a96427f5f7a93b79cbdf3d4b265b"

cd "$REL_ROOT"

echo "=== Applying patches to submodules ==="

# Keep both levels' URLs in sync with their .gitmodules files.  A plain
# `git submodule update --init` only guarantees the first level; --recursive
# is required for depth-anything.cpp's third_party/ggml checkout.
echo "Synchronizing submodule URLs (including nested submodules)..."
git submodule sync --recursive

ENGINE_DIR="$REL_ROOT/3rdparty/depth-anything.cpp"
GGML_DIR="$ENGINE_DIR/third_party/ggml"

is_git_worktree() {
    # The .git marker is essential here: `git -C empty/submodule-dir` otherwise
    # walks up to the parent repository and incorrectly reports success.
    [ -e "$1/.git" ] &&
        git -C "$1" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

# Initialize each level from the repository that owns its .gitmodules entry.
# Doing this in two stages also avoids global `submodule.*.update=none` settings
# silently skipping the nested checkout during one top-level recursive update.
if ! is_git_worktree "$ENGINE_DIR"; then
    echo "Initializing depth-anything.cpp submodule..."
    git -c submodule.3rdparty/depth-anything.cpp.update=checkout \
        submodule update --init --checkout -- 3rdparty/depth-anything.cpp
fi

if ! is_git_worktree "$ENGINE_DIR"; then
    echo "ERROR: depth-anything.cpp submodule was not initialized by Git."
    exit 1
fi

# Validate the engine checkout before asking it to initialize children.  This
# gives a useful error when an old release still pins a different engine tree.
git -C "$ENGINE_DIR" cat-file -e "$ENGINE_UPSTREAM^{commit}" || {
    echo "ERROR: depth-anything.cpp base $ENGINE_UPSTREAM is unavailable."
    exit 1
}

echo "Synchronizing depth-anything.cpp nested submodule URLs..."
git -C "$ENGINE_DIR" submodule sync --recursive

if ! is_git_worktree "$GGML_DIR"; then
    echo "Initializing nested ggml submodule..."
    # Do not pass a pathspec here.  Older Git versions/configurations can fail
    # with "pathspec ... did not match" before reading the nested .gitmodules.
    # The engine has only the dependencies declared in that file, so updating
    # all of them recursively is both safer and equivalent for this release.
    git -C "$ENGINE_DIR" -c submodule.third_party/ggml.update=checkout \
        submodule update --init --checkout --recursive
fi

if ! is_git_worktree "$GGML_DIR"; then
    echo "ERROR: Git did not initialize nested ggml at: $GGML_DIR"
    echo "Check the command output above and the nested configuration with:"
    echo "  git -C $ENGINE_DIR submodule status"
    echo "  git -C $ENGINE_DIR config --get-regexp '^submodule\\.'"
    exit 1
fi

# Fail with a useful message before git-log/git-am if a shallow or incomplete
# checkout does not contain the public nested patch base.
git -C "$GGML_DIR" cat-file -e "$GGML_UPSTREAM^{commit}" || {
    echo "ERROR: nested ggml base $GGML_UPSTREAM is unavailable."
    exit 1
}

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
apply_patch_set "ggml"   "$GGML_DIR"   "$GGML_UPSTREAM" "$PATCH_DIR/ggml"
apply_patch_set "engine" "$ENGINE_DIR" "$ENGINE_UPSTREAM" "$PATCH_DIR/depth-anything.cpp"

echo ""
echo "=== Patch application complete ==="
echo ""
echo "Final state:"
echo "  ggml:   $(git -C "$GGML_DIR" log --oneline --format='%h %s' -1)"
echo "  engine: $(git -C "$ENGINE_DIR" log --oneline --format='%h %s' -1)"
echo ""
echo "Note: the submodule gitlink shows as modified in the release repo after"
echo "patching - expected; the patches sit on top of the upstream-pinned SHAs."
