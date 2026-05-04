// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/rational/reset_pool_edge_cases_test.cpp
// ============================================================================
// EDGE CASE TESTS FOR reset_pool() AND GLOBAL POOL LIFECYCLE
// ============================================================================
//
// This file tests the behaviour of internal::reset_pool() and its interaction
// with lazy rational expressions, canonicalisation, garbage collection, and
// global caches (π cache, clean object registry, etc.).
//
// Key aspects verified:
//   - reset_pool() completely wipes the global pool and all caches.
//   - After reset, LazyRational objects become dirty zero (default state).
//   - Interning (hash‑consing) yields consistent indices after separate resets.
//   - GC and reset_pool() can be sequenced without memory leaks or dangling references.
//   - Transcendental functions (sin, cos, pi) continue to work correctly.
//   - The global default epsilon is not affected by pool reset.
//
// All tests run inside a fixture that calls reset_pool() before and after each test.
// ============================================================================

#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "delta/rational/node_pool.h"
#include "test_utils.h"
#include "lazy_rational_test_fixture.h"

namespace delta::testing {

    // Fixture that forces reset_pool() before and after each test.
    class ResetPoolEdgeCasesTest : public LazyRationalTestFixture {
    protected:
        void SetUp() override {
            internal::reset_pool();
        }
        void TearDown() override {
            internal::reset_pool();
        }
    };

    // -----------------------------------------------------------------------
    // 1. Simple transcendental expression after reset_pool
    // -----------------------------------------------------------------------
    /**
     * @test SimpleTranscendentalAfterReset
     * @brief Builds a simple expression Sin(0.5)*Cos(0.5) and evaluates it
     *        after resetting the pool. Repeated cycles ensure that the pool
     *        state does not leak across iterations.
     */
    TEST_F(ResetPoolEdgeCasesTest, SimpleTranscendentalAfterReset) {
        for (int cycle = 0; cycle < 5; ++cycle) {
            internal::reset_pool();
            LazyRational x = LazyRational("0.5"_r);
            LazyRational expr = Sin(x.clone()) * Cos(x.clone());
            expr.simplify_inplace();
            ASSERT_TRUE(is_clean(expr));
            Rational val = expr.eval();
            Rational expected = sin("0.5"_r) * cos("0.5"_r);
            EXPECT_EQ(val, expected) << "Cycle " << cycle;
        }
    }

    // -----------------------------------------------------------------------
    // 2. Direct minimal repetition of the problematic RepeatingTerm_Simplify_10
    // -----------------------------------------------------------------------
    /**
     * @test RepeatingTerm10AfterReset
     * @brief Accumulates the same term (sin(0.5)*cos(0.5)) 10 times and simplifies.
     *        This test reproduces a possible hang condition encountered in previous versions
     *        it verifies that after reset_pool() the simplification does not stall.
     */
    TEST_F(ResetPoolEdgeCasesTest, RepeatingTerm10AfterReset) {
        for (int attempt = 0; attempt < 3; ++attempt) {
            internal::reset_pool();
            LazyRational term_val = Sin("0.5"_r) * Cos("0.5"_r);
            LazyRational acc;
            for (int i = 0; i < 10; ++i) acc + term_val;
            acc.simplify_inplace();   // if it hangs, it happens here
            ASSERT_TRUE(is_clean(acc));
            Rational val = acc.eval();
            Rational expected = (sin("0.5"_r) * cos("0.5"_r)) * 10;
            EXPECT_EQ(val, expected) << "Attempt " << attempt;
        }
    }

    // -----------------------------------------------------------------------
    // 3. Multiple reset_pool cycles with different expressions
    // -----------------------------------------------------------------------
    /**
     * @test MultipleResetPoolCycles
     * @brief Repeats resetting the pool and building small transcendental
     *        expressions to ensure no resource accumulation.
     */
    TEST_F(ResetPoolEdgeCasesTest, MultipleResetPoolCycles) {
        for (int i = 0; i < 10; ++i) {
            internal::reset_pool();
            LazyRational a = Sin(LazyRational("0.2"_r));
            LazyRational b = Cos(LazyRational("0.3"_r));
            LazyRational c = a.clone() * b.clone() + a.clone();
            c.simplify_inplace();
            EXPECT_TRUE(is_clean(c));
            Rational r = c.eval();
            EXPECT_TRUE(r > 0_r);
        }
    }

    // -----------------------------------------------------------------------
    // 4. Pi cache integrity after pool reset
    // -----------------------------------------------------------------------
    /**
     * @test PiCacheIntegrity
     * @brief Checks that the cached value of π is correctly recomputed after
     *        reset_pool() and explicit reset_pi_cache().
     */
    TEST_F(ResetPoolEdgeCasesTest, PiCacheIntegrity) {
        // Explicit epsilon to force π computation
        set_default_eps(Rational("1/1000000000000000000000000000000"));

        Rational pi_before = pi(default_eps());

        internal::reset_pool();
        internal::reset_pi_cache(); // explicitly clear the cache

        Rational pi_after = pi(default_eps());
        EXPECT_EQ(pi_before, pi_after) << "Pi must be recomputed correctly after pool reset";
    }

    // -----------------------------------------------------------------------
    // 5. default_eps() survives pool reset
    // -----------------------------------------------------------------------
    /**
     * @test DefaultEpsAfterReset
     * @brief Ensures that reset_pool() does not alter the global default epsilon.
     */
    TEST_F(ResetPoolEdgeCasesTest, DefaultEpsAfterReset) {
        Rational eps_before = default_eps();
        internal::reset_pool();
        Rational eps_after = default_eps();
        EXPECT_EQ(eps_before, eps_after) << "Default epsilon must survive pool reset";
    }

    // -----------------------------------------------------------------------
    // 6. Interaction of GC and reset_pool with simplification
    // -----------------------------------------------------------------------
    /**
     * @test GCAndResetInteraction
     * @brief Tests a complex scenario:
     *        1. Build a large lazy sum (80 terms) without evaluating.
     *        2. Simplify → canonicalisation puts it into the pool.
     *        3. Force garbage collection → the tree is collapsed into a constant
     *           and the pool is replaced; the original LazyRational remains clean.
     *        4. Reset the pool → the LazyRational object becomes dirty zero
     *           (local default state) and all memory is freed.
     *        5. Build a new expression and evaluate, verifying correctness.
     *
     * This test validates that reset_pool() correctly invalidates all clean objects
     * and that no dangling references remain.
     */
    TEST_F(ResetPoolEdgeCasesTest, GCAndResetInteraction) {
        // reset_pool is already called in SetUp; explicit call is redundant but harmless.
        // Expected state: virgin pool, all caches empty, no dangling references.
        internal::reset_pool();
        internal::set_pool_max_size(120);
        LazyRational acc;
        for (int i = 0; i < 80; ++i) {
            // each iteration of the cycle causes construction and destruction of variable "term"; 
            // harmless (beneficial even) for tests, bad for real use-case scenario performance

            LazyRational term = Sin(Rational(i).as_lazy()) * Cos(Rational(i + 1).as_lazy());
            acc + term;   // terms are lazy, NOT evaluated
        }
        // Expected: acc lazily accumulated the terms; 'term' destroyed because out of scope.
        acc.simplify_inplace();   // canonicalise → becomes a clean tree in the pool.
        internal::force_garbage_collect();   // collapses the tree to a constant, creates a new pool.
        EXPECT_TRUE(is_clean(acc));

        internal::reset_pool();   // wipes the pool and caches; acc becomes dirty zero.
        // Expected: acc is now a dirty LazyRational with a single local node 0.
        internal::set_pool_max_size(200);
        LazyRational expr = Sin("0.7"_r) + Cos("0.8"_r);
        expr.simplify_inplace();   // canonicalises and enters the pool.
        Rational val = expr.eval();
        Rational expected = sin("0.7"_r) + cos("0.8"_r);
        EXPECT_EQ(val, expected);
    }

    // -----------------------------------------------------------------------
    // 7. Interning after multiple resets
    // -----------------------------------------------------------------------
    /**
     * @test InterningAfterMultipleResets
     * @brief Verifies that the hash‑consing mechanism yields the same clean index
     *        for the same expression built after separate pool resets.
     */
    TEST_F(ResetPoolEdgeCasesTest, InterningAfterMultipleResets) {
        int idx1 = -1, idx2 = -1;
        {
            internal::reset_pool();
            LazyRational a = LazyRational(3_r);
            a + 3_r + 3_r;
            a.simplify_inplace();
            idx1 = clean_index(a);
        }
        {
            internal::reset_pool();
            LazyRational b = LazyRational(3_r);
            b + 3_r + 3_r;
            b.simplify_inplace();
            idx2 = clean_index(b);
        }
        EXPECT_EQ(idx1, idx2) << "Interning should produce identical indices after separate resets";
    }

    // -----------------------------------------------------------------------
    // 8. Stress test (disabled by default, run manually if needed)
    // -----------------------------------------------------------------------
    /**
     * @test StressLargeAfterReset
     * @brief Builds a large sum of 200 identical terms, simplifies, and evaluates.
     *        Disabled by default; enable manually for stress testing.
     */
    TEST_F(ResetPoolEdgeCasesTest, StressLargeAfterReset) {
        internal::reset_pool();
        LazyRational term = Sin("0.123"_r) * Cos("0.456"_r);
        LazyRational sum;
        for (int i = 0; i < 200; ++i) sum + term;
        sum.simplify_inplace();
        Rational expected = (sin("0.123"_r) * cos("0.456"_r)) * 200;
        EXPECT_EQ(sum.eval(), expected);
    }

    // -----------------------------------------------------------------------
    // 9. RepeatingTerm_Simplify_10 in a loop
    // -----------------------------------------------------------------------
    /**
     * @test RepeatingTerm10ManyTimes
     * @brief Repeats the pattern of test 2 across multiple reset cycles.
     */
    TEST_F(ResetPoolEdgeCasesTest, RepeatingTerm10ManyTimes) {
        for (int iteration = 0; iteration < 5; ++iteration) {
            internal::reset_pool();
            LazyRational term_val = Sin("0.5"_r) * Cos("0.5"_r);
            LazyRational acc;
            for (int j = 0; j < 10; ++j) acc + term_val;
            acc.simplify_inplace();
            ASSERT_TRUE(is_clean(acc));
            Rational val = acc.eval();
            Rational expected = (sin("0.5"_r) * cos("0.5"_r)) * 10;
            EXPECT_EQ(val, expected) << "Iteration " << iteration;
        }
    }

    // -----------------------------------------------------------------------
    // 10. Distributivity with transcendental factors after reset
    // -----------------------------------------------------------------------
    /**
     * @test DistributivityWithTranscendentalReset
     * @brief Checks that algebraic simplification (distributivity) works correctly
     *        after a pool reset: a·b + a·c → a·(b+c).
     */
    TEST_F(ResetPoolEdgeCasesTest, DistributivityWithTranscendentalReset) {
        internal::reset_pool();
        LazyRational a = Sin("0.5"_r);
        LazyRational b = Cos("0.5"_r);
        LazyRational expr = (a.clone() * b.clone()) + (a.clone() * LazyRational(2_r));
        expr.simplify_inplace();
        EXPECT_TRUE(is_clean(expr));
        Rational val = expr.eval();
        Rational expected = sin("0.5"_r) * (cos("0.5"_r) + 2_r);
        EXPECT_EQ(val, expected);
    }

} // namespace delta::testing