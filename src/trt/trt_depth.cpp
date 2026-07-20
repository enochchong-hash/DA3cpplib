#include "trt_depth.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace da {
namespace {

struct TrtLogger final : nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::fprintf(stderr, "[DA3 TensorRT] %s\n", message);
    }
};

template <typename T>
using TrtPtr = std::unique_ptr<T>;

bool cuda_ok(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "DA3 TensorRT: %s failed: %s\n", operation,
                 cudaGetErrorString(status));
    return false;
}

bool read_file(const std::string& path, std::vector<char>& bytes) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const auto size = file.tellg();
    if (size < 0) return false;
    file.seekg(0, std::ios::beg);
    bytes.resize(static_cast<std::size_t>(size));
    return bytes.empty() || static_cast<bool>(file.read(bytes.data(), size));
}

bool write_file(const std::string& path, const void* data, std::size_t size) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    return file.good();
}

std::uint64_t fnv1a(const void* data, std::size_t size) {
    const auto* p = static_cast<const unsigned char*>(data);
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= p[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::int64_t volume(const nvinfer1::Dims& dims) {
    std::int64_t value = 1;
    for (int i = 0; i < dims.nbDims; ++i) value *= dims.d[i];
    return value;
}

std::string safe_gpu_name() {
    int device = 0;
    if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice")) return "unknown-gpu";
    cudaDeviceProp properties{};
    if (!cuda_ok(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties"))
        return "unknown-gpu";
    std::string name = properties.name;
    for (char& c : name) if (c == ' ' || c == '/' || c == '\\') c = '_';
    return name;
}

bool build_or_load(const TrtOptions& options, TrtLogger& logger,
                   std::vector<char>& plan) {
    std::vector<char> onnx;
    if (!read_file(options.onnx_path, onnx)) {
        std::fprintf(stderr, "DA3 TensorRT: cannot read ONNX graph '%s'\n",
                     options.onnx_path.c_str());
        return false;
    }
    bool fp16 = options.fp16 && std::getenv("DA3_TRT_FP32") == nullptr;
    std::uint64_t hash = fnv1a(onnx.data(), onnx.size());
    // torch.onnx may externalize large checkpoints next to model.onnx as
    // model.onnx.data. Include that payload in the plan key as well.
    std::vector<char> external_data;
    if (read_file(options.onnx_path + ".data", external_data))
        hash ^= fnv1a(external_data.data(), external_data.size());
    hash ^= fp16 ? 0x9e3779b97f4a7c15ULL : 0;
    hash ^= static_cast<std::uint64_t>(options.workspace_bytes);

    const std::string cache_dir = options.cache_dir.empty()
        ? options.onnx_path + ".cache" : options.cache_dir;
    char filename[1024];
    std::snprintf(filename, sizeof(filename),
                  "%s/da3-depth-%s-trt%d.%d.%d.%d-%016llx.engine",
                  cache_dir.c_str(), safe_gpu_name().c_str(), NV_TENSORRT_MAJOR,
                  NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, NV_TENSORRT_BUILD,
                  static_cast<unsigned long long>(hash));

    if (!std::getenv("DA3_TRT_REBUILD") && read_file(filename, plan)) {
        std::fprintf(stderr, "DA3 TensorRT: loaded cached engine '%s'\n", filename);
        return true;
    }

    std::fprintf(stderr, "DA3 TensorRT: building %s engine from '%s'\n",
                 fp16 ? "FP16" : "FP32", options.onnx_path.c_str());
    TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) return false;
    TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    TrtPtr<nvonnxparser::IParser> parser(
        network ? nvonnxparser::createParser(*network, logger) : nullptr);
    if (!network || !parser) return false;
    // parseFromFile is required for ONNX external-data tensors; parsing the
    // already-read protobuf bytes would lose the sidecar's relative path.
    if (!parser->parseFromFile(options.onnx_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::fprintf(stderr, "DA3 TensorRT: ONNX parse failed (%d errors)\n",
                     parser->getNbErrors());
        for (int i = 0; i < parser->getNbErrors(); ++i)
            std::fprintf(stderr, "  %s\n", parser->getError(i)->desc());
        return false;
    }
    if (network->getNbInputs() != 1 ||
        std::string(network->getInput(0)->getName()) != "image") {
        std::fprintf(stderr, "DA3 TensorRT: graph must have exactly one input named 'image'\n");
        return false;
    }
    const nvinfer1::Dims input_dims = network->getInput(0)->getDimensions();
    if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3 ||
        input_dims.d[2] <= 0 || input_dims.d[3] <= 0) {
        std::fprintf(stderr, "DA3 TensorRT: 'image' must have fixed shape [1,3,H,W]\n");
        return false;
    }
    bool have_depth = false, have_confidence = false;
    for (int i = 0; i < network->getNbOutputs(); ++i) {
        const std::string name = network->getOutput(i)->getName();
        have_depth |= name == "depth";
        have_confidence |= name == "confidence";
    }
    if (!have_depth || !have_confidence) {
        std::fprintf(stderr, "DA3 TensorRT: graph outputs must be named 'depth' and 'confidence'\n");
        return false;
    }

    TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) return false;
    if (fp16) config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               options.workspace_bytes);
    TrtPtr<nvinfer1::IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
        std::fprintf(stderr, "DA3 TensorRT: engine build failed\n");
        return false;
    }
    plan.assign(static_cast<const char*>(serialized->data()),
                static_cast<const char*>(serialized->data()) + serialized->size());
    std::error_code ec;
    std::filesystem::create_directories(cache_dir, ec);
    if (!write_file(filename, plan.data(), plan.size()))
        std::fprintf(stderr, "DA3 TensorRT: warning: could not cache '%s'\n", filename);
    else
        std::fprintf(stderr, "DA3 TensorRT: cached engine '%s'\n", filename);
    return true;
}

class TrtDepthImpl final : public TrtDepth {
public:
    explicit TrtDepthImpl(TrtOptions options) : options_(std::move(options)) {}
    ~TrtDepthImpl() override {
        for (auto& binding : bindings_) if (binding.device) cudaFree(binding.device);
        if (stream_) cudaStreamDestroy(stream_);
    }

    bool ready() override {
        std::lock_guard<std::mutex> lock(mutex_);
        return initialize_locked();
    }

    bool infer(const std::vector<float>& chw, int height, int width,
               std::vector<float>& depth, std::vector<float>& confidence) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialize_locked()) return false;
        if (height != input_height_ || width != input_width_) {
            std::fprintf(stderr,
                         "DA3 TensorRT: graph is %dx%d but this image preprocesses to %dx%d\n",
                         input_width_, input_height_, width, height);
            return false;
        }
        const std::size_t input_elements = std::size_t{3} * height * width;
        if (chw.size() != input_elements) return false;
        Binding* image = find("image");
        Binding* depth_out = find("depth");
        Binding* confidence_out = find("confidence");
        const std::size_t pixels = static_cast<std::size_t>(height) * width;
        if (!image || !depth_out || !confidence_out || image->elements != input_elements ||
            depth_out->elements != pixels || confidence_out->elements != pixels) {
            std::fprintf(stderr, "DA3 TensorRT: unexpected engine tensor shape\n");
            return false;
        }
        if (!cuda_ok(cudaMemcpyAsync(image->device, chw.data(), chw.size() * sizeof(float),
                                     cudaMemcpyHostToDevice, stream_), "copy image H2D"))
            return false;
        if (!context_->enqueueV3(stream_)) {
            std::fprintf(stderr, "DA3 TensorRT: enqueueV3 failed\n");
            return false;
        }
        depth.resize(pixels);
        confidence.resize(pixels);
        if (!cuda_ok(cudaMemcpyAsync(depth.data(), depth_out->device, pixels * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream_), "copy depth D2H") ||
            !cuda_ok(cudaMemcpyAsync(confidence.data(), confidence_out->device,
                                     pixels * sizeof(float), cudaMemcpyDeviceToHost, stream_),
                     "copy confidence D2H") ||
            !cuda_ok(cudaStreamSynchronize(stream_), "cudaStreamSynchronize")) {
            depth.clear();
            confidence.clear();
            return false;
        }
        return true;
    }

private:
    struct Binding {
        std::string name;
        std::size_t elements = 0;
        void* device = nullptr;
    };

    Binding* find(const std::string& name) {
        for (auto& binding : bindings_) if (binding.name == name) return &binding;
        return nullptr;
    }

    bool initialize_locked() {
        if (initialized_) return true;
        if (attempted_) return false;
        attempted_ = true;
        std::vector<char> plan;
        if (!build_or_load(options_, logger_, plan)) return false;
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) return false;
        engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
        if (!engine_) return false;
        context_.reset(engine_->createExecutionContext());
        if (!context_ || !cuda_ok(cudaStreamCreate(&stream_), "cudaStreamCreate")) return false;

        const int count = engine_->getNbIOTensors();
        for (int i = 0; i < count; ++i) {
            const char* name = engine_->getIOTensorName(i);
            const nvinfer1::Dims dims = engine_->getTensorShape(name);
            for (int d = 0; d < dims.nbDims; ++d) {
                if (dims.d[d] <= 0) {
                    std::fprintf(stderr, "DA3 TensorRT: dynamic engine tensors are unsupported\n");
                    return false;
                }
            }
            if (engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
                std::fprintf(stderr, "DA3 TensorRT: tensor '%s' is not FLOAT\n", name);
                return false;
            }
            Binding binding;
            binding.name = name;
            binding.elements = static_cast<std::size_t>(volume(dims));
            if (!cuda_ok(cudaMalloc(&binding.device, binding.elements * sizeof(float)),
                         "cudaMalloc(binding)")) return false;
            context_->setTensorAddress(name, binding.device);
            bindings_.push_back(binding);
            if (binding.name == "image") {
                if (dims.nbDims != 4) return false;
                input_height_ = dims.d[2];
                input_width_ = dims.d[3];
            }
        }
        if (!find("image") || !find("depth") || !find("confidence")) return false;
        initialized_ = true;
        return true;
    }

    TrtOptions options_;
    TrtLogger logger_;
    TrtPtr<nvinfer1::IRuntime> runtime_;
    TrtPtr<nvinfer1::ICudaEngine> engine_;
    TrtPtr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_ = nullptr;
    std::vector<Binding> bindings_;
    int input_height_ = 0;
    int input_width_ = 0;
    bool attempted_ = false;
    bool initialized_ = false;
    std::mutex mutex_;
};

} // namespace

std::unique_ptr<TrtDepth> make_trt_depth(const TrtOptions& options) {
    return std::unique_ptr<TrtDepth>(new TrtDepthImpl(options));
}

} // namespace da
