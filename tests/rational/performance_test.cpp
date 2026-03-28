// tests/rational/performance_test.cpp
#include <gtest/gtest.h>
#include <vector>
#include "delta/core/rational.h"
#include "test_utils.h"
#include <boost/multiprecision/cpp_dec_float.hpp>

namespace delta::testing {

    class RationalPerformanceTest : public RationalTest {};

    // -------------------------------------------------------------------------
    // 1. Harmonic series (eager mode) – compare with high‑precision reference
    // -------------------------------------------------------------------------
    TEST_F(RationalPerformanceTest, HarmonicSeries10000EagerMode) {
        std::cout << "--- IT'S GONNA TAKE A WHILE (around 40s in debug) - TIME TO THINK ABOUT ETERNITY. ---" << std::endl;
        std::cout << "--- For example: what did you have for breakfast today? ---" << std::endl;
        const int N = 10000;
        ScopedEagerEval eager;
        Rational sum = 0_r;
        for (int i = 1; i <= N; ++i) {
            sum = sum + Rational(1, i);
        }

        // Compute reference value using high‑precision float (no doubles in user code)
        using boost::multiprecision::cpp_dec_float_100;
        cpp_dec_float_100 ref = 0;
        for (int i = 1; i <= N; ++i) {
            ref += cpp_dec_float_100(1) / i;
        }
        // Convert reference to Rational with enough precision
        std::string ref_str = ref.str(60, std::ios_base::fixed);
        Rational expected(ref_str);

        // Tolerance: 1e‑30
        Rational eps = Rational("1/1000000000000000000000000000000");
        EXPECT_RATIONAL_NEAR(sum.eval(), expected, eps);
    }

    // -------------------------------------------------------------------------
    // 2. Deep tree (lazy mode) – build a chain of additions, then evaluate
    // -------------------------------------------------------------------------
    TEST_F(RationalPerformanceTest, DeepTree10000LazyMode) {
        const int DEPTH = 10000;
        set_eager_mode(false);
        Rational tree = 0_r;
        for (int i = 0; i < DEPTH; ++i) {
            tree = tree + 1_r;
        }
        Rational result = tree.eval();
        EXPECT_EQ(result, Rational(DEPTH));
    }

    // -------------------------------------------------------------------------
    // 3. Large product (factorial) – just ensure it doesn't blow up
    // -------------------------------------------------------------------------
    TEST_F(RationalPerformanceTest, LargeProduct500) {
        const int N = 500;
        Rational prod = 1_r;
        for (int i = 1; i <= N; ++i) {
            prod = prod * Rational(i);
        }
        EXPECT_GT(prod, 0_r);
        prod.eval(); // ensure evaluation works
    }

    // -------------------------------------------------------------------------
    // 4. Nested transcendental functions with simplification
    // -------------------------------------------------------------------------
    TEST_F(RationalPerformanceTest, NestedTranscendentals) {
        set_eager_mode(false);
        Rational x = "2"_r;
        Rational expr = delta::sin(delta::cos(delta::exp(delta::log(x))));
        Rational simplified = expr.simplify();
        Rational expected = delta::sin(delta::cos("2"_r));
        EXPECT_EQ(simplified.eval(), expected.eval());
    }

    // -------------------------------------------------------------------------
    // 5. Deep mixed tree – harmonic series in lazy mode, compare with eager
    // -------------------------------------------------------------------------
    TEST_F(RationalPerformanceTest, DeepMixedTree5000LazyEager) {
        const int DEPTH = 5000;
        set_eager_mode(false);
        Rational sum_lazy = 0_r;
        for (int i = 1; i <= DEPTH; ++i) {
            sum_lazy = sum_lazy + Rational(1, i);
        }
        Rational result_lazy = sum_lazy.eval();

        ScopedEagerEval eager;
        Rational sum_eager = 0_r;
        for (int i = 1; i <= DEPTH; ++i) {
            sum_eager = sum_eager + Rational(1, i);
        }
        EXPECT_EQ(result_lazy, sum_eager);
    }

} // namespace delta::testing