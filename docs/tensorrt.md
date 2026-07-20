# TensorRT backend

DA3cpplib has two distinct NVIDIA execution options:

- `DA3CPP_CUDA=ON` runs the native GGUF graph through ggml-CUDA.
- `DA3CPP_TENSORRT=ON` additionally compiles an opt-in TensorRT backend for
  standard, single-view DA3 DualDPT depth and confidence inference.

The TensorRT design follows sam3cpplib: an offline ONNX export, an engine built
on first use, and a serialized plan cache keyed by ONNX contents, precision,
workspace size, GPU name, and TensorRT version. TensorRT types are private and
do not leak through the public library headers.

## Setup and build

```bash
scripts/setup/setup_tensorrt.sh --copy-from ../sam3cpplib/3rdparty
scripts/build.sh cuda --trt
```

Without `--copy-from`, the setup script downloads the pinned TensorRT 10.13.3
headers and CUDA 12 runtime libraries. The paths can instead be supplied to
CMake with `DA3CPP_TRT_INCLUDE_DIR` and `DA3CPP_TRT_LIB_DIR`.

## Export a graph

The official PyTorch checkpoint is an export-time input only; it is not a
runtime dependency of da3cpplib.

```bash
python3 scripts/convert/export_da3_depth_onnx.py \
  /models/DA3-BASE deploy/da3-base-504x504.onnx \
  --source /src/Depth-Anything-3 --height 504 --width 504
```

Large checkpoints may also produce `deploy/da3-base-504x504.onnx.data`; keep
that sidecar beside the ONNX file. It is parsed by TensorRT and included in the
engine cache key.

Height and width must be multiples of patch size 14. Graphs are fixed-shape,
as in sam3cpplib's image-encoder path. DA3 preserves image aspect ratio during
preprocessing, so a deployment can export the shapes it expects to receive;
an input producing a different shape uses GGUF automatically when fallback is
enabled.

The ONNX contract is intentionally small and validated before an engine is
built:

- input `image`: F32 `[1,3,H,W]`, already resized and ImageNet-normalized;
- output `depth`: F32 with `H*W` elements;
- output `confidence`: F32 with `H*W` elements.

## Programmatic use

```cpp
da3::Params params;
params.model_path = "depth-anything-base-q8_0.gguf";
params.tensorrt.enabled = true;
params.tensorrt.onnx_path = "deploy/da3-base-504x504.onnx";
params.tensorrt.cache_dir = "cache/da3";
params.tensorrt.fp16 = true;
params.tensorrt.fallback_to_ggml = true;

auto model = da3::load_model(params);
```

`model_info(*model).tensorrt_active` becomes true after the most recent depth
inference successfully used TensorRT. Pose/ray-pose, multi-view, Gaussian
reconstruction, nested metric, DA3 mono, DA2, legacy resize, and the prepared
device-input API remain on the existing GGUF backend.

The compatibility C API and CLI can opt in without a new ABI by setting:

```bash
export DA3_TRT=1
export DA3_TRT_ONNX_PATH=deploy/da3-base-504x504.onnx
export DA3_TRT_CACHE_DIR=cache/da3
```

Debug controls are `DA3_TRT_FP32=1`, `DA3_TRT_REBUILD=1`, and
`DA3_TRT_NO_FALLBACK=1`.

## GPU contract test

On a TensorRT-enabled build host with the Python `onnx` package:

```bash
python3 tests/generate_tensorrt_contract.py /tmp/da3-trt-test.onnx
DA_TEST_TENSORRT_ONNX=/tmp/da3-trt-test.onnx \
  ctest --test-dir build -R test_tensorrt --output-on-failure
```

This builds/loads a real plan and verifies named inputs, named outputs, FP16
execution, host/device transfers, and numerical results without downloading a
DA3 checkpoint.
