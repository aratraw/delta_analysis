# tests/geometry/CMakeLists.txt
include(GoogleTest)

# Списки исходных файлов
set(GEOMETRY_CORE_SOURCES
    main_tests_geometry.cpp
    product_regulative_test.cpp
    constructive_core_test.cpp
    simplicial_complex_test.cpp
    tensor_field_test.cpp
    matrix_field_test.cpp
    discrete_operators_test.cpp
    discrete_operators_3d_4d_test.cpp
)

set(GEOMETRY_ADVANCED_SOURCES
    main_tests_geometry.cpp
    dual_complex_test.cpp
    hat_basis_test.cpp
 discrete_forms_test.cpp
)

# Создание таргетов
add_executable(delta_tests_geometry ${GEOMETRY_CORE_SOURCES})
add_executable(delta_tests_geometry_advanced ${GEOMETRY_ADVANCED_SOURCES})

# Общая настройка для обоих бинарников
set(ALL_GEOMETRY_TESTS delta_tests_geometry delta_tests_geometry_advanced)

foreach(test_target ${ALL_GEOMETRY_TESTS})
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

    # Предварительно откомпилированные заголовки
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
