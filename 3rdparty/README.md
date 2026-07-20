# Third-party dependencies

- `ggml` is pinned as a Git submodule. CPU, CUDA, Metal, and Vulkan execution
  are selected through da3cpplib's CMake options.
- `stb_image` and `stb_image_write` provide the optional file-oriented image
  helpers and are included under their upstream license blocks.
- TensorRT headers and runtime libraries are optional, ignored developer
  artifacts populated by `scripts/setup/setup_tensorrt.sh`. They are not
  required by the default ggml build and are not committed to this repository.

The optional fused CUDA DPT head uses one small ggml extension. The patch is
kept in `scripts/patches/ggml` and applied idempotently by `scripts/setup.sh`.
The normal ggml CUDA path does not require enabling the fused kernels at
runtime.
