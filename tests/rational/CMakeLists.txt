# tests/rational/CMakeLists.txt

# ============================================================================
# Основной бинарник: функциональные тесты
# ============================================================================
add_executable(delta_tests_rational
    main_tests_rational.cpp
    rational_gauss_qi_test.cpp
    rational_test.cpp
    rational_test_2.cpp
#    interval_test.cpp
#    batch_test.cpp
#    lazy_rational_contract_tests.cpp
#    lazy_test.cpp
#    lazy_simplification_tests.cpp
#    gc_test.cpp
#    gc_reset_pool_edge_cases_tests.cpp
#    pow_test.cpp
    transcendentals_correctness.cpp
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

target_compile_definitions(delta_tests_rational PRIVATE DELTA_TESTING)

# PCH для основного бинарника (STL + GTest)
target_precompile_headers(delta_tests_rational PRIVATE
    <gtest/gtest.h>
    <vector>
    <functional>
    <iostream>
    <string>
    <memory>
    <algorithm>
)

include(GoogleTest)
gtest_discover_tests(delta_tests_rational
    PROPERTIES
    ENVIRONMENT "PATH=${MSVC_BIN_DIR};$ENV{PATH}"
)

# ============================================================================
# Отдельный бинарник: бенчмарки ядра (производительность)
# ============================================================================
add_executable(delta_tests_rational_kernel_benchmarks
    main_tests_rational.cpp
    "transcendentals_canonicalization_benchmark.cpp"
    "transcendentals_comparative.cpp"
    "performance_compare_test.cpp"
    "performance_test.cpp"
)

target_include_directories(delta_tests_rational_kernel_benchmarks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(delta_tests_rational_kernel_benchmarks
    PRIVATE
    delta_core
    gtest_main
)

if(MSVC)
    target_compile_options(delta_tests_rational_kernel_benchmarks PRIVATE /EHsc)
endif()

target_compile_definitions(delta_tests_rational_kernel_benchmarks PRIVATE DELTA_TESTING)

# PCH для бенчмарков (STL + GTest + хроно)
target_precompile_headers(delta_tests_rational_kernel_benchmarks PRIVATE
    <gtest/gtest.h>
    <vector>
    <functional>
    <iostream>
    <string>
    <memory>
    <algorithm>
    <chrono>
    <ratio>
)

gtest_discover_tests(delta_tests_rational_kernel_benchmarks
    PROPERTIES
    ENVIRONMENT "PATH=${MSVC_BIN_DIR};$ENV{PATH}"
)

#===== EIGEN TRANSCENDENTAL TESTS ======
# Отдельный бинарник: тесты интеграции с Eigen (матричные трансценденты)
add_executable(delta_tests_rational_eigen
    main_tests_rational.cpp
    eigen_integration_test.cpp
    eigen_integration_gauss_qi.cpp
)

target_include_directories(delta_tests_rational_eigen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(delta_tests_rational_eigen
    PRIVATE
    delta_core
    gtest_main
)

if(MSVC)
    # Флаг /bigobj увеличивает лимит COFF-секций с 2^16 до 2^32
    target_compile_options(delta_tests_rational_eigen PRIVATE /EHsc /bigobj)
endif()

target_compile_definitions(delta_tests_rational_eigen PRIVATE DELTA_TESTING)

# PCH для Eigen-тестов (тяжёлые библиотеки + Eigen)
# ВНИМАНИЕ: eigen_transcendentals.h не включаем в PCH, так как он может часто меняться
target_precompile_headers(delta_tests_rational_eigen PRIVATE
    <Eigen/Core>
    <Eigen/Dense>
    <gtest/gtest.h>
    <vector>
    <functional>
    <iostream>
    <string>
    <memory>
    <algorithm>
    <cmath>
    <cstdint>
)

gtest_discover_tests(delta_tests_rational_eigen
    PROPERTIES
    ENVIRONMENT "PATH=${MSVC_BIN_DIR};$ENV{PATH}"
)