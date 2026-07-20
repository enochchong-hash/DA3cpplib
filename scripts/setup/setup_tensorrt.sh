#!/usr/bin/env bash
# Vendor the optional TensorRT 10.13.3 headers and CUDA 12 runtime libraries.
# This mirrors sam3cpplib's standalone setup and is only needed when building
# with DA3CPP_TENSORRT=ON.
set -euo pipefail

TRT_TAG="v10.13.3"
TRT_PIP_VERSION="10.13.3.9"
ONNX_TRT_COMMIT="9a9f7883dd7b8cb0a718395bac2075fab6f97da8"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
INCLUDE_DIR="$ROOT_DIR/3rdparty/tensorrt-include"
LIB_DIR="$ROOT_DIR/3rdparty/tensorrt-libs"

usage() {
    echo "Usage: $0 [--copy-from PATH]" >&2
    echo "PATH must contain tensorrt-include/ and tensorrt-libs/." >&2
}

if [[ "${1:-}" == "--copy-from" ]]; then
    [[ $# -eq 2 ]] || { usage; exit 2; }
    SOURCE_DIR="$(cd "$2" && pwd)"
    [[ -d "$SOURCE_DIR/tensorrt-include" && -d "$SOURCE_DIR/tensorrt-libs" ]] || {
        echo "Error: invalid TensorRT SDK directory: $SOURCE_DIR" >&2
        exit 1
    }
    rm -rf "$INCLUDE_DIR" "$LIB_DIR"
    cp -al "$SOURCE_DIR/tensorrt-include" "$INCLUDE_DIR" 2>/dev/null ||
        cp -a "$SOURCE_DIR/tensorrt-include" "$INCLUDE_DIR"
    cp -al "$SOURCE_DIR/tensorrt-libs" "$LIB_DIR" 2>/dev/null ||
        cp -a "$SOURCE_DIR/tensorrt-libs" "$LIB_DIR"
    echo "TensorRT SDK imported. Build with: scripts/build.sh cuda --trt"
    exit 0
elif [[ $# -ne 0 ]]; then
    usage
    exit 2
fi

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
rm -rf "$INCLUDE_DIR" "$LIB_DIR"
mkdir -p "$INCLUDE_DIR" "$LIB_DIR"

git clone --depth 1 --branch "$TRT_TAG" https://github.com/NVIDIA/TensorRT.git \
    "$TEMP_DIR/TensorRT"
cp "$TEMP_DIR/TensorRT"/include/*.h "$INCLUDE_DIR/"
git clone https://github.com/onnx/onnx-tensorrt.git "$TEMP_DIR/onnx-tensorrt"
git -C "$TEMP_DIR/onnx-tensorrt" checkout "$ONNX_TRT_COMMIT"
cp "$TEMP_DIR/onnx-tensorrt/NvOnnxParser.h" "$INCLUDE_DIR/"

python3 -m venv "$TEMP_DIR/venv"
"$TEMP_DIR/venv/bin/pip" install --quiet --upgrade pip
"$TEMP_DIR/venv/bin/pip" install --quiet "tensorrt-cu12==$TRT_PIP_VERSION"
SITE_PACKAGES="$("$TEMP_DIR/venv/bin/python" -c \
    'import os, tensorrt_libs; print(os.path.dirname(tensorrt_libs.__file__))')"
cp -P "$SITE_PACKAGES"/libnvinfer.so* "$LIB_DIR/"
cp -P "$SITE_PACKAGES"/libnvinfer_plugin.so* "$LIB_DIR/"
cp -P "$SITE_PACKAGES"/libnvonnxparser.so* "$LIB_DIR/"
cp "$SITE_PACKAGES"/libnvinfer_builder_resource.so.* "$LIB_DIR/"
for library in nvinfer nvinfer_plugin nvonnxparser; do
    versioned="$(find "$LIB_DIR" -maxdepth 1 -name "lib${library}.so.*" | head -1)"
    [[ -z "$versioned" ]] || ln -sf "$(basename "$versioned")" "$LIB_DIR/lib${library}.so"
done

echo "TensorRT SDK installed. Build with: scripts/build.sh cuda --trt"
