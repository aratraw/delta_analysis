// tests/rational/performance_test.cpp
#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <cmath>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {

    // Helper to compute approximate harmonic sum: 1/1 + 1/2 + ... + 1/n
    static double harmonic_approx(int n) {
        double sum = 0.0;
        for (int i = 1; i <= n; ++i) sum += 1.0 / i;
        return sum;
    }

    // -------------------------------------------------------------------------
    // 1. Harmonic series (eager mode) – measure time and check result
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, HarmonicSeries) {
        const int N = 10000;
        std::vector<Rational> terms;
        terms.reserve(N);
        for (int i = 1; i <= N; ++i) {
            terms.push_back(Rational(1, i));
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Use eager mode for fair timing (no lazy overhead)
        ScopedEagerEval eager;
        Rational sum = 0_r;
        for (const auto& t : terms) {
            sum = sum + t;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        // Check that the sum is approximately ln(N) + gamma
        double expected = std::log(N) + 0.5772156649015328606065120900824024310421;
        double actual = static_cast<double>(sum);
        EXPECT_NEAR(actual, expected, 1e-6);
        EXPECT_LT(elapsed, 1.0); // should be less than 1 second
    }

    // -------------------------------------------------------------------------
    // 2. Deep tree (lazy mode) – build a chain of additions, then evaluate
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, DeepTree) {
        const int DEPTH = 10000;
        set_eager_mode(false);

        auto start = std::chrono::high_resolution_clock::now();

        // Build left-associative tree: (((0+1)+1)+...)
        Rational tree = 0_r;
        for (int i = 0; i < DEPTH; ++i) {
            tree = tree + 1_r;
        }

        auto build_end = std::chrono::high_resolution_clock::now();
        double build_time = std::chrono::duration<double>(build_end - start).count();

        // Now evaluate (should collapse the tree)
        Rational result = tree.evaluate();

        auto eval_end = std::chrono::high_resolution_clock::now();
        double eval_time = std::chrono::duration<double>(eval_end - build_end).count();

        // The result should be exactly DEPTH
        EXPECT_EQ(result, Rational(DEPTH));

        // The tree should have been collapsed (non‑lazy) due to depth limit
        EXPECT_FALSE(tree.is_lazy());

        // Times should be reasonable (adjust thresholds as needed)
        EXPECT_LT(build_time, 0.5);
        EXPECT_LT(eval_time, 0.5);
    }

    // -------------------------------------------------------------------------
    // 3. Matrix exponential – (commented out until MatrixField is ready)
    //    This test would be moved to the geometry module once the matrix field
    //    is integrated with the new rational core.
    // -------------------------------------------------------------------------
    /*
    TEST_F(RationalTest, MatrixExp) {
        // Placeholder – to be implemented when MatrixField is available
    }
    */

} // namespace delta::testing