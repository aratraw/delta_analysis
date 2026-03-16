# tests/solvers/CMakeLists.txt
# CMake configuration for PDE solver unit tests.
# This suite tests Poisson, heat, wave, advection, and variational solvers.

add_executable(delta_tests_solvers
    main_tests_solvers.cpp
)

target_include_directories(delta_tests_solvers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(delta_tests_solvers
    PRIVATE
    delta_core
    gtest_main
)

if(MSVC)
    target_compile_options(delta_tests_solvers PRIVATE /EHsc)
endif()

include(GoogleTest)
gtest_discover_tests(delta_tests_solvers
    PROPERTIES
    ENVIRONMENT "PATH=${MSVC_BIN_DIR};$ENV{PATH}"
)