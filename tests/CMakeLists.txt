function(da3cpp_add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE da3cpp::da3cpp)
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
        ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

da3cpp_add_test(test_linalg)
da3cpp_add_test(test_ray_pose)
da3cpp_add_test(test_nested_align)
if(DA3CPP_TENSORRT)
    da3cpp_add_test(test_tensorrt)
endif()
