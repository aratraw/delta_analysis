# tests/geometry/CMakeLists.txt
# CMake configuration for geometry unit tests.
# This suite tests simplicial complexes, dual complexes, hat basis functions,
# discrete forms, connections, curvature, and simplicial paths.

add_executable(delta_tests_geometry
    main_tests_geometry.cpp
    simplicial_complex_test.cpp
    product_regulative_test.cpp
 constructive_core_test.cpp
 "tensor_field_test.cpp" 
 #"matrix_field_test.cpp" 
 "discrete_operators_test.cpp"
 "discrete_operators_3d_4d_test.cpp")

target_include_directories(delta_tests_geometry PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(delta_tests_geometry
    PRIVATE
    delta_core
    gtest_main
)

if(MSVC)
    target_compile_options(delta_tests_geometry PRIVATE /EHsc)
endif()

include(GoogleTest)
gtest_discover_tests(delta_tests_geometry
    PROPERTIES
    ENVIRONMENT "PATH=${MSVC_BIN_DIR};$ENV{PATH}"
)

target_precompile_headers(delta_tests_geometry PRIVATE
    <Eigen/Core>
    <Eigen/Sparse>
    <Eigen/Dense>
    <gtest/gtest.h>
    <fmt/format.h>
    <vector>
    <functional>
    <iostream>
    <omp.h>
)