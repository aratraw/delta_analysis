# tests/rational/CMakeLists.txt
# CMake configuration for rational unit tests.

add_executable(delta_tests_rational
    main_tests_rational.cpp
    "rational_test.cpp"
#    "lazy_test.cpp"
#    "depth_test.cpp"
#    "eager_test.cpp"
#    "batch_test.cpp"
#    "interval_test.cpp"
#    "transcendentals_test.cpp"
#   "eigen_integration_test.cpp"
#    "performance_test.cpp"
    )

target_include_directories(delta_tests_rational PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(delta_tests_rational
    PRIVATE
    delta_core
    gtest_main
)

if(MSVC)
    target_compile_options(delta_tests_rational PRIVATE /EHsc)
endif()

include(GoogleTest)
gtest_discover_tests(delta_tests_rational
    PROPERTIES
    ENVIRONMENT "PATH=${MSVC_BIN_DIR};$ENV{PATH}"
)