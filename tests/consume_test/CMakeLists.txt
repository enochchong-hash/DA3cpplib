cmake_minimum_required(VERSION 3.18)
project(da3cpp_consume_test LANGUAGES C CXX)

set(DA3CPP_CUDA OFF CACHE BOOL "" FORCE)
set(DA3CPP_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(DA3CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. da3cpplib)

add_executable(consume_main main.cpp)
target_link_libraries(consume_main PRIVATE da3cpp::da3cpp)

add_library(c_header_check OBJECT c_header.c)
target_link_libraries(c_header_check PRIVATE da3cpp::da3cpp)
