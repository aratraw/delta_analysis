// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

//tests/solvers/main_tests_solvers.cpp
#include <gtest/gtest.h>
#include <omp.h>
#include <iostream>

int main(int argc, char** argv) {
    // ПРИНУДИТЕЛЬНАЯ инициализация OpenMP до запуска тестов
    // Это "прогревает" рантайм и предотвращает Access Violation
    omp_set_num_threads(1);

    std::cout << "[OpenMP] Initialized with LLVM backend. Threads: "
        << omp_get_max_threads() << std::endl;

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}