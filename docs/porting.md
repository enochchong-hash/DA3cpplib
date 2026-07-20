# Port status

The initial extraction uses `depth-anything.cpp` commit
`f4e17dea695dd12ae76bea98ba58030996b98118` as the behavioral baseline.

Integrated into first-party `src/`:

- GGUF model loading and CPU/GPU backend selection
- DA3/DA2 preprocessing, DINO backbone, DPT and camera heads
- mono, nested metric, multiview, ray pose, reconstruction, and exporters
- the release DA3 prepared-input patch
- the release fused DPT CUDA head and tile-size optimization patches
- the former C ABI, now published as `da3cpp/da3_c.h`

New library boundary:

- `da3cpp::da3cpp` CMake target with a diamond-inclusion guard
- opaque C++ API with raw RGB8 inputs and owning results
- top-level-only examples/tests and a standalone consumer build
- no HTTP, TLS, web resources, server lifecycle, or UI dependencies

The UI/server migration belongs in the separate `release/da3cppui` repository.
Its GPU preprocessing code can migrate through `da3::model_info` and
`da3::infer_preprocessed`, without including internal engine or ggml headers.
