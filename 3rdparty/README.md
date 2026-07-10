# Third-Party Dependencies

This directory contains all third-party components used by the DA3 depth-estimation service.

## Components

| Component | Upstream URL | Pinned SHA | License | Local Patches |
|-----------|--------------|------------|---------|---------------|
| depth-anything.cpp | https://github.com/mudler/depth-anything.cpp | f4e17de | MIT | 3 patches (see `scripts/patches/depth-anything.cpp/`) |
| ggml (nested in depth-anything.cpp) | https://github.com/ggml-org/ggml | 3af5f57 | MIT | 1 patch (see `scripts/patches/ggml/`) |
| cpp-httplib | https://github.com/yhirose/cpp-httplib | 0.48.0 | MIT | None |
| stb (nested in depth-anything.cpp) | https://github.com/nothings/stb | (upstream) | Public Domain/MIT | None |

## Patch Application

Local modifications to third-party code are shipped as patch files in `scripts/patches/`.
Apply them by running:

```sh
scripts/apply_patches.sh
```

This script:
1. Initializes the submodules if needed
2. Applies patches with `git am --3way` to preserve authorship
3. Is idempotent (safe to run multiple times)

**Note:** Running `git submodule update --checkout --force` will discard patches.
Re-run `apply_patches.sh` after any submodule reset.

## Licensing

- **depth-anything.cpp**: MIT License
- **ggml**: MIT License  
- **cpp-httplib**: MIT License
- **stb**: Public Domain / MIT (see individual headers)

All licenses are included in their respective directories.
