cmake_minimum_required(VERSION 3.18)
project(da3-release LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CUDA architecture: default to native, can be overridden via -DCMAKE_CUDA_ARCHITECTURES
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
endif()

# Force engine to build CLI (for bench/profiling)
set(DA_BUILD_CLI ON CACHE BOOL "Build CLI tools" FORCE)
set(DA_GGML_CUDA ON CACHE BOOL "Enable ggml CUDA" FORCE)
set(GGML_CUDA_GRAPHS ON CACHE BOOL "Enable ggml CUDA graphs" FORCE)

# Add engine subdirectory (once)
add_subdirectory(3rdparty/depth-anything.cpp)

# Imported targets found inside the engine subproject are directory-scoped;
# the server target needs its own find_package for CUDA::cudart/nvjpeg + Threads.
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# Find OpenSSL for httplib TLS support
find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
  add_definitions(-DCPPHTTPLIB_OPENSSL_SUPPORT)
endif()

# Server executable
add_executable(depth-ui-server
  src/server.cpp
  src/gpu_preprocess.cu
  3rdparty/cpp-httplib/httplib.cpp
)

target_link_libraries(depth-ui-server PRIVATE
  depthanything
  CUDA::cudart
  CUDA::nvjpeg
  Threads::Threads
)

if(OpenSSL_FOUND)
  target_link_libraries(depth-ui-server PRIVATE OpenSSL::SSL OpenSSL::Crypto)
endif()

# Includes
target_include_directories(depth-ui-server PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/depth-anything.cpp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/depth-anything.cpp/src
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/depth-anything.cpp/third_party/stb
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cpp-httplib
)

# Compile-time defaults: LAST-RESORT fallback only. At runtime the server
# prefers $DA3_MODEL_DIR/$DA3_WWW_DIR env vars, then <exe>/../resources
# (relocatable). CMAKE_CURRENT_SOURCE_DIR (not CMAKE_SOURCE_DIR) so the paths
# stay correct when this project is consumed via add_subdirectory().
target_compile_definitions(depth-ui-server PRIVATE
  DA3_DEFAULT_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources/nnmodels"
  DA3_DEFAULT_WWW_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources"
)
