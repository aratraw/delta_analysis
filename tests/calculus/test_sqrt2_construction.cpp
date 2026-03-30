// tests/calculus/test_sqrt2_construction.cpp
#include <gtest/gtest.h>
#include "test_fixtures.h"

namespace delta::testing {

    /**
     * @class Sqrt2ConstructionTest
     * @brief Tests the construction of √2 as a fundamental sequence via dyadic refinement.
     *
     * This test demonstrates how a dyadic path on the interval [0,2] generates
     * a nested sequence of intervals containing √2. The left endpoints of these
     * intervals form a fundamental sequence converging to √2. Two different
     * representations (left endpoints and right endpoints) should be equivalent
     * as real numbers.
     */
    class Sqrt2ConstructionTest : public DeltaTest {};

    /**
     * @brief Generates the sequence of left endpoints of intervals containing √2
     *        for dyadic refinement on [0,2] starting with grid {0,2}.
     *
     * At each step, the interval [left, right] is bisected. If the midpoint
     * is ≤ √2, the left half is kept; otherwise the right half is kept.
     * The left endpoint of the current interval is recorded.
     *
     * @param levels Number of refinement steps.
     * @return Vector of left endpoints at each level.
     */
    std::vector<Addr> generate_sqrt2_left_endpoints(std::size_t levels) {
        std::vector<Addr> result;
        result.reserve(levels);
        double target = std::sqrt(2.0);
        Rational left = 0;
        Rational right = 2;
        for (std::size_t n = 0; n < levels; ++n) {
            result.push_back(left);
            Rational mid = (left + right) / 2;
            if (mid.to_double() <= target) {
                left = mid;
            }
            else {
                right = mid;
            }
        }
        return result;
    }

    /**
     * @test Verify that the left‑endpoint sequence from dyadic refinement
     *       is a fundamental sequence with rate 1/2 and bound 2,
     *       and that it is equivalent to the right‑endpoint sequence.
     *
     * Mathematical background:
     *   A fundamental sequence {x_n} satisfies |x_m - x_n| ≤ C·r^{min(m,n)}
     *   for some C>0 and 0<r<1. For the dyadic construction of √2,
     *   the left endpoints satisfy |x_{n+1} - x_n| ≤ 2/2^{n+1}, hence
     *   the whole sequence is fundamental with C=2, r=1/2.
     *   Two fundamental sequences are equivalent if their difference decays
     *   exponentially with the same rate. Here the left and right endpoints
     *   differ by the interval length, which also decays like 2/2^{n+1},
     *   so they are equivalent.
     */
    TEST_F(Sqrt2ConstructionTest, DyadicPathGeneratesFundamentalSequence) {
        const std::size_t N_LEVELS = 40; // enough to verify equivalence
        auto seq_vals = generate_sqrt2_left_endpoints(N_LEVELS);

        // Create a fundamental sequence from the left endpoints
        auto gen = [seq_vals](std::size_t n) { return seq_vals[n]; };
        FundamentalSequence seq(gen, Rational(2), Rational(1, 2), 0);

        // Check that differences decay exponentially
        for (std::size_t i = 1; i < seq_vals.size(); ++i) {
            Rational diff = seq_vals[i] - seq_vals[i - 1];
            if (diff < 0) diff = -diff;
            double expected_max = 2.0 / pow2(i).to_double();
            EXPECT_LE(diff.to_double(), expected_max + 1e-12);
        }

        // Create a sequence of right endpoints (left + interval length)
        auto right_gen = [seq_vals](std::size_t n) {
            Rational len = Rational(2) / pow2(n + 1);
            return seq_vals[n] + len;
            };
        FundamentalSequence right_seq(right_gen, Rational(2), Rational(1, 2), 0);

        // The two sequences should be equivalent (both converge to √2)
        EXPECT_TRUE(are_equivalent(seq, right_seq));
    }

} // namespace delta::testing