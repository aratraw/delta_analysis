# tests/numerical/CMakeLists.txt
# CMake configuration for numerical unit tests.
# This suite tests finite differences, gradient, Laplacian, discrete operators,
# FEM assemblers, cotangent Laplacian, gauge theory, Green's identities,
# and Cartesian grid operators.

add_executable(delta_tests_numerical
    main_tests_numerical.cpp
    test_boundary_conditions.cpp
    test_cartesian_operators.cpp
    test_cotangent_laplacian.cpp
    test_discrete_operators.cpp
    test_fem_assemblers.cpp
    test_finite_differences.cpp
    test_gauge_theory.cpp
    test_integrals_green.cpp
    test_simplicial_complex.cpp
    test_tensor_arithmetic.cpp
    test_tensor_field.cpp
)

target_include_directories(delta_tests_numerical PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(delta_tests_numerical
    PRIVATE
    delta_core
    gtest_main
)

if(MSVC)
    target_compile_options(delta_tests_numerical PRIVATE /EHsc)
endif()

include(GoogleTest)
gtest_discover_tests(delta_tests_numerical
    PROPERTIES
    ENVIRONMENT "PATH=${MSVC_BIN_DIR};$ENV{PATH}"
)