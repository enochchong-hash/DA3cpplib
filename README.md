# da3cpplib

Standalone C++17 library for Depth Anything 3 and Depth Anything V2 GGUF
inference. It is intended to be embedded in applications as a Git submodule,
using the same `add_subdirectory` workflow as `sam3cpplib`.

The inference implementation and the DA3 release optimizations live directly
in this repository. There is no `depth-anything.cpp` dependency and no patching
of an engine submodule. `ggml` remains the portable compute dependency; an
idempotent one-file ggml CUDA extension is retained for the optional fused DPT
head.

## Build

```bash
git clone --recurse-submodules <da3cpplib-url>
cd da3cpplib
./scripts/setup.sh
./scripts/build.sh auto
```

CPU-only configuration:

```bash
cmake -S . -B build -DDA3CPP_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Embed in another application

```cmake
set(DA3CPP_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(DA3CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(3rdparty/da3cpplib)
target_link_libraries(my_app PRIVATE da3cpp::da3cpp)
```

```cpp
#include "da3cpp/da3.h"

da3::Params params;
params.model_path = "depth-anything-base-q8_0.gguf";
auto model = da3::load_model(params);

da3::Result result;
if (model && da3::infer_file(*model, "photo.jpg", result)) {
    // result.depth is row-major result.width * result.height floats.
}
```

The public API accepts raw interleaved RGB8 through `da3::Image`, so host
applications do not need file I/O. The file helper, compatibility C API,
camera pose output, nested metric loading, and an advanced zero-copy prepared
input hook are also included. See [docs/architecture.md](docs/architecture.md)
and [docs/porting.md](docs/porting.md).

## CMake options

| Option | Default | Meaning |
|---|---:|---|
| `DA3CPP_CUDA` | ON | ggml CUDA backend |
| `DA3CPP_METAL` | OFF | ggml Metal backend |
| `DA3CPP_VULKAN` | OFF | ggml Vulkan backend |
| `DA3CPP_CUSTOM_CUDA_KERNELS` | ON | Compile optional fused DPT kernels |
| `DA3CPP_AVX512` | OFF | Build an AVX-512-only Winograd fast path |
| `DA3CPP_BUILD_CLI` | top-level only | Build CLI and API example |
| `DA3CPP_BUILD_TESTS` | top-level only | Build tests |

Model weights are intentionally not part of the library repository. Existing
DA3 GGUF files are compatible and can be supplied by the consuming app.
