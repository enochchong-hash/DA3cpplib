#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#define DA3CPP_VERSION_MAJOR 0
#define DA3CPP_VERSION_MINOR 1
#define DA3CPP_VERSION_PATCH 0
#define DA3CPP_VERSION "0.1.0"

namespace da3 {

struct Model;

enum class ModelKind {
    da3,
    da3_mono,
    da3_nested,
    da2_relative,
    da2_metric,
};

struct Params {
    std::string model_path;
    int n_threads = 4;
};

struct NestedParams {
    std::string anyview_model_path;
    std::string metric_model_path;
    int n_threads = 4;
};

// Interleaved RGB8 image (width * height * 3 bytes).
struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> data;
};

struct ModelInfo {
    ModelKind kind = ModelKind::da3;
    std::string checkpoint_name;
    int patch_size = 0;
    int resize_target = 0;
    std::array<float, 3> mean{};
    std::array<float, 3> std{};
    bool is_metric = false;
    bool has_camera_pose = false;
    bool has_aux_ray_head = false;
};

struct Result {
    int width = 0;
    int height = 0;
    std::vector<float> depth;
    std::vector<float> confidence;
    std::vector<float> sky;
    std::array<float, 12> extrinsics{}; // 3x4 row-major
    std::array<float, 9> intrinsics{};  // 3x3 row-major
    bool is_metric = false;
    bool has_pose = false;
};

struct InferOptions {
    // DA3 DualDPT and nested models can produce camera pose. Keeping this off
    // avoids running the pose head when an application only needs depth.
    bool with_pose = false;
    // Select the auxiliary ray-based pose path. Requires an aux-bearing model
    // and with_pose=true.
    bool ray_pose = false;
    // Use the legacy floor-to-patch resize path, which keeps the output near
    // the source resolution. This is substantially slower and is primarily
    // retained for the UI's explicit high-resolution mode.
    bool legacy_resize = false;
};

std::shared_ptr<Model> load_model(const Params& params);
std::shared_ptr<Model> load_nested_model(const NestedParams& params);
ModelInfo model_info(const Model& model);

bool infer(Model& model, const Image& image, Result& result,
           const InferOptions& options = {});
bool infer_file(Model& model, const std::string& image_path, Result& result,
                const InferOptions& options = {});

// Advanced zero-copy input path used by GPU-native applications. The caller
// supplies an already resized and normalized planar F32 image [3,H,W]. The
// callback is invoked synchronously with the allocated graph input. `device`
// is true when destination points to device memory. Only DA3 cat-token models
// support this depth-only path.
struct UploadTarget {
    void* destination = nullptr;
    std::size_t bytes = 0;
    bool device = false;
};
using UploadCallback = std::function<void(const UploadTarget&)>;
bool infer_preprocessed(Model& model, int height, int width,
                        const UploadCallback& upload, Result& result);

} // namespace da3
