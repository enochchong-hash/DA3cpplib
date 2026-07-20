#include "da3cpp/da3.h"

#include <algorithm>
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf IMAGE\n", argv[0]);
        return 2;
    }

    da3::Params params;
    params.model_path = argv[1];
    auto model = da3::load_model(params);
    if (!model) {
        std::fprintf(stderr, "failed to load model: %s\n", argv[1]);
        return 1;
    }

    da3::Result result;
    if (!da3::infer_file(*model, argv[2], result)) {
        std::fprintf(stderr, "inference failed\n");
        return 1;
    }
    const auto bounds = std::minmax_element(result.depth.begin(), result.depth.end());
    std::printf("depth %dx%d min=%.6f max=%.6f metric=%s\n",
                result.width, result.height, *bounds.first, *bounds.second,
                result.is_metric ? "yes" : "no");
    return 0;
}
