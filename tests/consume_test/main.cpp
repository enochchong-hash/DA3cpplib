#include "da3cpp/da3.h"

#include <cstdio>

int main() {
    da3::Params params;
    params.model_path = "/nonexistent.gguf";
    params.tensorrt.enabled = false;
    params.tensorrt.onnx_path = "/nonexistent.onnx";
    params.tensorrt.fallback_to_ggml = true;
    if (da3::load_model(params)) {
        std::fprintf(stderr, "unexpectedly loaded a nonexistent model\n");
        return 1;
    }
    da3::Image invalid;
    da3::Result result;
    std::printf("consume_test: da3cpplib %s public header is usable\n", DA3CPP_VERSION);
    return invalid.data.empty() && result.depth.empty() ? 0 : 1;
}
