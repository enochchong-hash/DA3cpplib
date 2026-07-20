#include "da3cpp/da3.h"

#include "engine.hpp"
#include "image_io.hpp"

#include <algorithm>
#include <cctype>
#include <utility>

namespace da3 {

struct Model {
    std::unique_ptr<da::Engine> engine;
};

namespace {

bool contains_case_insensitive(std::string value, const std::string& needle) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value.find(needle) != std::string::npos;
}

bool config_is_metric(const da::Config& config) {
    return config.head_max_depth > 0.0f ||
           contains_case_insensitive(config.checkpoint_name, "metric") ||
           contains_case_insensitive(config.checkpoint_name, "nested") ||
           contains_case_insensitive(config.checkpoint_name, "mono");
}

da::Image to_internal_image(const Image& image) {
    da::Image converted;
    converted.w = image.width;
    converted.h = image.height;
    converted.rgb = image.data;
    return converted;
}

bool valid_image(const Image& image) {
    return image.width > 0 && image.height > 0 &&
           image.data.size() == static_cast<std::size_t>(image.width) * image.height * 3;
}

void finish_result(Result& result, int height, int width, bool metric) {
    result.height = height;
    result.width = width;
    result.is_metric = metric;
}

} // namespace

std::shared_ptr<Model> load_model(const Params& params) {
    if (params.model_path.empty()) return nullptr;
    da::TrtOptions trt;
    trt.enabled = params.tensorrt.enabled;
    trt.onnx_path = params.tensorrt.onnx_path;
    trt.cache_dir = params.tensorrt.cache_dir;
    trt.fp16 = params.tensorrt.fp16;
    trt.fallback_to_ggml = params.tensorrt.fallback_to_ggml;
    trt.workspace_bytes = params.tensorrt.workspace_bytes;
    auto engine = da::Engine::load(params.model_path, params.n_threads, trt);
    if (!engine) return nullptr;
    auto model = std::make_shared<Model>();
    model->engine = std::move(engine);
    return model;
}

std::shared_ptr<Model> load_nested_model(const NestedParams& params) {
    if (params.anyview_model_path.empty() || params.metric_model_path.empty()) return nullptr;
    auto engine = da::Engine::load_nested(params.anyview_model_path,
                                          params.metric_model_path,
                                          params.n_threads);
    if (!engine) return nullptr;
    auto model = std::make_shared<Model>();
    model->engine = std::move(engine);
    return model;
}

ModelInfo model_info(const Model& model) {
    ModelInfo info;
    if (!model.engine) return info;
    const auto& engine = *model.engine;
    const auto& config = engine.config();
    info.checkpoint_name = config.checkpoint_name;
    info.patch_size = static_cast<int>(config.patch_size);
    info.resize_target = static_cast<int>(config.img_resize_target);
    for (std::size_t i = 0; i < 3 && i < config.img_mean.size(); ++i) info.mean[i] = config.img_mean[i];
    for (std::size_t i = 0; i < 3 && i < config.img_std.size(); ++i) info.std[i] = config.img_std[i];
    info.is_metric = config_is_metric(config) || engine.is_nested();
    info.has_aux_ray_head = engine.has_aux();
    info.tensorrt_enabled = engine.tensorrt_enabled();
    info.tensorrt_active = engine.tensorrt_active();

    if (engine.is_nested()) {
        info.kind = ModelKind::da3_nested;
        info.has_camera_pose = true;
    } else if (engine.is_da2()) {
        info.kind = config.head_max_depth > 0.0f ? ModelKind::da2_metric
                                                : ModelKind::da2_relative;
    } else if (engine.is_mono()) {
        info.kind = ModelKind::da3_mono;
    } else {
        info.kind = ModelKind::da3;
        info.has_camera_pose = true;
    }
    return info;
}

bool infer(Model& model, const Image& image, Result& result,
           const InferOptions& options) {
    result = {};
    if (!model.engine || !valid_image(image)) return false;
    auto& engine = *model.engine;
    engine.clear_tensorrt_active();
    const auto info = model_info(model);
    auto internal = to_internal_image(image);
    int height = 0;
    int width = 0;

    if (engine.is_nested()) {
        da::NestedOut nested;
        if (!engine.depth_metric(internal, nested, height, width)) return false;
        result.depth = std::move(nested.depth);
        result.extrinsics = nested.extrinsics;
        result.intrinsics = nested.intrinsics;
        result.has_pose = true;
    } else if (engine.is_da2()) {
        if (options.with_pose || options.ray_pose) return false;
        if (!engine.depth_relative(internal, result.depth, height, width)) return false;
    } else if (engine.is_mono()) {
        if (options.with_pose || options.ray_pose) return false;
        if (!engine.depth_mono(internal, result.depth, result.sky, height, width)) return false;
    } else if (options.with_pose) {
        if (options.legacy_resize && options.ray_pose) return false;
        const bool ok = options.ray_pose
            ? engine.depth_pose_rays_native(internal, result.depth, result.confidence,
                                            result.extrinsics, result.intrinsics,
                                            height, width)
            : options.legacy_resize
                ? engine.depth_pose(internal, result.depth, result.confidence,
                                    result.extrinsics, result.intrinsics,
                                    height, width)
                : engine.depth_pose_native(internal, result.depth, result.confidence,
                                           result.extrinsics, result.intrinsics,
                                           height, width);
        if (!ok) return false;
        result.has_pose = true;
    } else {
        if (options.ray_pose) return false;
        const bool ok = options.legacy_resize
            ? engine.depth_image(internal, result.depth, result.confidence, height, width)
            : engine.depth_native_image(internal, result.depth, result.confidence,
                                        height, width);
        if (!ok) return false;
    }

    finish_result(result, height, width, info.is_metric);
    return !result.depth.empty();
}

bool infer_file(Model& model, const std::string& image_path, Result& result,
                const InferOptions& options) {
    da::Image image;
    if (!da::load_image_rgb(image_path, image)) {
        result = {};
        return false;
    }
    Image public_image;
    public_image.width = image.w;
    public_image.height = image.h;
    public_image.data = std::move(image.rgb);
    return infer(model, public_image, result, options);
}

bool infer_preprocessed(Model& model, int height, int width,
                        const UploadCallback& upload, Result& result) {
    result = {};
    if (!model.engine || !upload || height <= 0 || width <= 0) return false;
    auto& engine = *model.engine;
    engine.clear_tensorrt_active();
    if (engine.is_nested() || engine.is_da2() || engine.is_mono()) return false;

    const bool ok = engine.depth_native_prepared([&](ggml_tensor* tensor) {
        const bool host = tensor->buffer && ggml_backend_buffer_is_host(tensor->buffer);
        upload({tensor->data, ggml_nbytes(tensor), !host});
    }, height, width, result.depth, result.confidence);
    if (!ok) return false;
    finish_result(result, height, width, model_info(model).is_metric);
    return !result.depth.empty();
}

} // namespace da3
