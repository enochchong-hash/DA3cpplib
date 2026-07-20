# Architecture

`include/da3cpp/da3.h` is the stable application-facing C++ surface. Its
opaque `da3::Model` owns the first-party engine in `src/`, which loads GGUF
metadata and weights, selects a ggml backend, preprocesses RGB8 input, builds
the ViT/DPT graphs, and returns owning C++ result vectors.

```text
application RGB8 / prepared F32
            |
       da3cpp API
            |
  preprocess -> DINO backbone -> DPT head -> depth/confidence/sky
                         |          |
                         +----------+-> pose / nested metric / reconstruction
            |
       ggml CPU, CUDA, Metal, or Vulkan
```

The library recognizes ordinary DA3 DualDPT, DA3 mono, two-model DA3 nested,
DA2 relative, and DA2 metric checkpoints from GGUF metadata. Applications can
query the selected route with `da3::model_info`.

The `da3cpp/da3_c.h` ABI is retained for C and FFI consumers. New C++ code
should prefer `da3cpp/da3.h` because it uses owning containers and avoids
manual allocation/free pairs.
