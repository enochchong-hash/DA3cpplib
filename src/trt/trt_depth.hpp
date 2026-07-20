#pragma once

#include "engine.hpp"

#include <memory>

namespace da {

// Implemented only when DA3CPP_TENSORRT=ON. Kept behind Engine's abstract
// TrtDepth interface so public/library headers never expose TensorRT types.
std::unique_ptr<TrtDepth> make_trt_depth(const TrtOptions& options);

} // namespace da
