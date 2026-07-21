# da3cpplib -- standalone C++ library for Depth Anything 3/2 inference.
cmake_minimum_required(VERSION 3.18)

if(TARGET da3cpp)
    return()
endif()

project(da3cpplib VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(DA3CPP_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(DA3CPP_TOP_LEVEL ON)
endif()

option(DA3CPP_CUDA "Enable the ggml CUDA backend" ON)
option(DA3CPP_TENSORRT "Build the optional TensorRT depth backend" OFF)
option(DA3CPP_METAL "Enable the ggml Metal backend" OFF)
option(DA3CPP_VULKAN "Enable the ggml Vulkan backend" OFF)
option(DA3CPP_CUSTOM_CUDA_KERNELS "Compile the optional fused DA3 DPT kernels" ON)
option(DA3CPP_AVX512 "Compile the Winograd AVX-512 fast path (target CPU must support it)" OFF)
option(DA3CPP_BUILD_CLI "Build da3cpp_cli" ${DA3CPP_TOP_LEVEL})
option(DA3CPP_BUILD_TESTS "Build tests" ${DA3CPP_TOP_LEVEL})

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA ${DA3CPP_CUDA} CACHE BOOL "" FORCE)
set(GGML_METAL ${DA3CPP_METAL} CACHE BOOL "" FORCE)
set(GGML_VULKAN ${DA3CPP_VULKAN} CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)

if(NOT TARGET ggml)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ggml ggml EXCLUDE_FROM_ALL)
endif()

set(DA3CPP_SOURCES
    src/api.cpp
    src/model_loader.cpp
    src/backend.cpp
    src/image_io.cpp
    src/preprocess.cpp
    src/rope2d.cpp
    src/uv_posembed.cpp
    src/attention.cpp
    src/vit_block.cpp
    src/dino_backbone.cpp
    src/dpt_blocks.cpp
    src/winograd.cpp
    src/dpt_head.cpp
    src/gs_head.cpp
    src/gs_adapter.cpp
    src/ply_export.cpp
    src/reconstruct.cpp
    src/glb_export.cpp
    src/colmap_export.cpp
    src/cam_pose.cpp
    src/linalg.cpp
    src/ray_pose.cpp
    src/nested.cpp
    src/depth_export.cpp
    src/pose_export.cpp
    src/engine.cpp
    src/da_capi.cpp
    src/cli.cpp
    src/quantize.cpp
)

if(DA3CPP_CUDA AND DA3CPP_CUSTOM_CUDA_KERNELS)
    enable_language(CUDA)
    list(APPEND DA3CPP_SOURCES src/cuda_head_kernels.cu)
endif()

if(DA3CPP_TENSORRT)
    list(APPEND DA3CPP_SOURCES src/trt/trt_depth.cpp)
endif()

add_library(da3cpp STATIC ${DA3CPP_SOURCES})
add_library(da3cpp::da3cpp ALIAS da3cpp)
# Temporary compatibility for the current release/da3 consumer.
add_library(depthanything ALIAS da3cpp)

target_include_directories(da3cpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/da3cpp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/stb
)
target_link_libraries(da3cpp PUBLIC ggml)
target_compile_features(da3cpp PUBLIC cxx_std_17)

if(DA3CPP_AVX512)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" DA3CPP_HAS_AVX512F)
    if(NOT DA3CPP_HAS_AVX512F)
        message(FATAL_ERROR "DA3CPP_AVX512=ON but the compiler does not support -mavx512f")
    endif()
    set_source_files_properties(src/winograd.cpp PROPERTIES
        COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mfma")
endif()

if(DA3CPP_CUDA AND DA3CPP_CUSTOM_CUDA_KERNELS)
    target_compile_definitions(da3cpp PRIVATE DA_CUDA_HEAD)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(da3cpp PUBLIC CUDA::cudart)
    # da3cpp compiles its own CUDA source (src/cuda_head_kernels.cu). ggml picks
    # a CMAKE_CUDA_ARCHITECTURES default, but only via a plain scope-local set()
    # inside its ggml-cuda subdirectory (no CACHE / no PARENT_SCOPE), so that
    # value never reaches this scope. Without an explicit architecture, da3cpp's
    # CUDA_ARCHITECTURES would be empty and CMP0104 aborts at generate time
    # ("CUDA_ARCHITECTURES is empty for target da3cpp"). Honor a cache/-D
    # override when present, otherwise pick a sensible default.
    if(CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(da3cpp PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    elseif(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        # Build for the GPUs present on the build machine.
        set_target_properties(da3cpp PROPERTIES CUDA_ARCHITECTURES "native")
    else()
        # Broad fallback for older CMake without "native" support.
        set_target_properties(da3cpp PROPERTIES CUDA_ARCHITECTURES "75-virtual;80-virtual;86-real;89-real")
    endif()
endif()

if(DA3CPP_TENSORRT)
    target_compile_definitions(da3cpp PRIVATE DA3_TENSORRT)
    set(DA3CPP_TRT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tensorrt-include"
        CACHE PATH "TensorRT public headers")
    set(DA3CPP_TRT_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tensorrt-libs"
        CACHE PATH "TensorRT runtime library directory")
    find_path(DA3CPP_TRT_INCLUDE_PATH NvInfer.h PATHS ${DA3CPP_TRT_INCLUDE_DIR} REQUIRED)
    find_library(DA3CPP_TRT_NVINFER nvinfer PATHS ${DA3CPP_TRT_LIB_DIR} REQUIRED)
    find_library(DA3CPP_TRT_NVONNXPARSER nvonnxparser PATHS ${DA3CPP_TRT_LIB_DIR} REQUIRED)
    find_package(CUDAToolkit REQUIRED)
    target_include_directories(da3cpp PRIVATE ${DA3CPP_TRT_INCLUDE_PATH})
    target_link_libraries(da3cpp PUBLIC
        ${DA3CPP_TRT_NVINFER} ${DA3CPP_TRT_NVONNXPARSER} CUDA::cudart)
    set_target_properties(da3cpp PROPERTIES BUILD_RPATH "${DA3CPP_TRT_LIB_DIR}")
endif()

if(DA3CPP_BUILD_CLI)
    add_subdirectory(examples)
endif()
if(DA3CPP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
