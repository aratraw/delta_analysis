# tests/numerical/CMakeLists.txt
# CMake configuration for numerical unit tests.
# This suite tests finite differences, gradient, Laplacian, discrete operators,
# FEM assemblers, cotangent Laplacian, gauge theory, Green's identities,
# and Cartesian grid operators.

add_executable(delta_tests_numerical
    main_tests_numerical.cpp
#    integrals_test.cpp
#    cotangent_laplacian_test.cpp
# "boundary_conditions_test.cpp" 
 #"solvers/poisson_solver_test.cpp"
 #"solvers/heat_solver_test.cpp" 
 #"solvers/wave_solver_test.cpp" 
 #"solvers/advection_solver_test.cpp"
 "variational_solver_tests.cpp")

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