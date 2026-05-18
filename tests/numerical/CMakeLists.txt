# tests/numerical/CMakeLists.txt
include(GoogleTest)

# Базовые численные тесты и интегралы
set(NUMERICAL_CORE_SOURCES
    main_tests_numerical.cpp
    integrals_test.cpp
    cotangent_laplacian_test.cpp
    gauge_groups_test.cpp
    gauge_theory_test.cpp
)

# Тесты, связанные с солверами
set(NUMERICAL_SOLVER_SOURCES
    main_tests_numerical.cpp
    "variational_solver_tests.cpp"
    "boundary_conditions_test.cpp"
    "solvers/poisson_solver_test.cpp"
    "solvers/heat_solver_test.cpp"
    "solvers/wave_solver_test.cpp"
    "solvers/advection_solver_test.cpp"
)

# Создание таргетов
add_executable(delta_tests_numerical ${NUMERICAL_CORE_SOURCES})
add_executable(delta_tests_numerical_solvers ${NUMERICAL_SOLVER_SOURCES})

# Общая настройка
set(ALL_NUMERICAL_TESTS delta_tests_numerical delta_tests_numerical_solvers)

foreach(test_target ${ALL_NUMERICAL_TESTS})
    # Директории включения
    target_include_directories(${test_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

    # Линковка библиотек
    target_link_libraries(${test_target}
        PRIVATE
        delta_core
        gtest_main
    )

    # Специфичные флаги MSVC
    if(MSVC)
        target_compile_options(${test_target} PRIVATE /EHsc)
    endif()

    # Регистрация тестов в GoogleTest
    gtest_discover_tests(${test_target}
        PROPERTIES
        ENVIRONMENT "PATH=${MSVC_BIN_DIR};$ENV{PATH}"
    )

    # Предварительно откомпилированные заголовки для ускорения сборки
    target_precompile_headers(${test_target} PRIVATE
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
endforeach()
