# tests/numerical/CMakeLists.txt
# CMake configuration for numerical tests (e.g., tensor fields).
# This suite focuses on matrix‑valued functions (Eigen::MatrixXd) and their
# behaviour with uniform grids and operational functions.

# 1. Create the test executable for numerical tests.
add_executable(delta_tests_numerical
    main_tests_numerical.cpp    # Test main with OpenMP initialisation
    test_tensor_field.cpp       # Tests for tensor fields (Eigen matrices) on uniform grids
 "test_tensor_arithmetic.cpp" 
 "test_finite_differences.cpp" 
 "test_simplicial_complex.cpp"
test_cotangent_laplacian.cpp)

# 2. Link against required libraries.
#    delta_core is an INTERFACE library that brings in Boost, Eigen, OpenMP, fmt,
#    and the necessary include directories. gtest_main provides the Google Test
#    entry point (and automatically pulls in gtest).
target_link_libraries(delta_tests_numerical
    PRIVATE
    delta_core
    gtest_main
)

# 3. Integration with Visual Studio Test Explorer and CTest.
include(GoogleTest)

# Discover all tests defined in the executable and register them with CTest.
# The ENVIRONMENT property ensures that the directory containing the MSVC compiler
# (and thus the OpenMP DLLs) is in PATH, preventing test discovery failures.
# MSVC_BIN_DIR is set in the root CMakeLists.txt.
gtest_discover_tests(delta_tests_numerical
    PROPERTIES 
    ENVIRONMENT "PATH=${MSVC_BIN_DIR};$ENV{PATH}"
)