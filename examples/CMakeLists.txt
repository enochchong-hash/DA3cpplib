add_executable(da3cpp_cli da3_cli.cpp)
target_link_libraries(da3cpp_cli PRIVATE da3cpp::da3cpp)
target_include_directories(da3cpp_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

add_executable(da3cpp_depth_image depth_image.cpp)
target_link_libraries(da3cpp_depth_image PRIVATE da3cpp::da3cpp)
