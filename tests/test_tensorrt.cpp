#include "trt/trt_depth.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

int main() {
    const char* graph_path = std::getenv("DA_TEST_TENSORRT_ONNX");
    if (!graph_path || !std::ifstream(graph_path).good()) return 77;

    da::TrtOptions options;
    options.enabled = true;
    options.onnx_path = graph_path;
    options.cache_dir = std::string(graph_path) + ".cache";
    options.fallback_to_ggml = false;
    auto backend = da::make_trt_depth(options);
    if (!backend || !backend->ready()) return 1;

    constexpr int height = 14;
    constexpr int width = 28;
    const std::size_t pixels = std::size_t{height} * width;
    std::vector<float> input(3 * pixels);
    for (std::size_t i = 0; i < pixels; ++i) {
        input[i] = 1.0f;
        input[pixels + i] = 2.0f;
        input[2 * pixels + i] = 3.0f;
    }
    std::vector<float> depth, confidence;
    if (!backend->infer(input, height, width, depth, confidence)) return 1;
    if (depth.size() != pixels || confidence.size() != pixels) return 1;
    for (std::size_t i = 0; i < pixels; ++i) {
        if (std::fabs(depth[i] - 2.0f) > 1e-3f ||
            std::fabs(confidence[i] - 3.0f) > 1e-3f) {
            std::fprintf(stderr, "TensorRT mismatch at %zu: depth=%g confidence=%g\n",
                         i, depth[i], confidence[i]);
            return 1;
        }
    }
    return 0;
}
