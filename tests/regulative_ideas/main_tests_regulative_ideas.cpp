//tests/numerical/main_tests_numerical.cpp
#include <gtest/gtest.h>
#include <omp.h>
#include <iostream>

int main(int argc, char** argv) {
    // FORCED OpenMP initialization before running tests
    // This "warms up" the runtime and prevents Access Violation
    // "Warm-up" call: force OMP to create thread pool right now
#pragma omp parallel
    {
#pragma omp master
        std::cout << "[OpenMP] Warmup. Total threads: " << omp_get_num_threads() << std::endl;
    }
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}