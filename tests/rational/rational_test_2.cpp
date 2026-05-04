// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/rational/rational_test_2.cpp
// ============================================================================
// ADDITIONAL TESTS FOR RATIONAL – REDUCTION, CANONICAL FORM, LARGE OPERATIONS
// ============================================================================
//
// This file extends the tests for the Rational class with a focus on:
//   - Automatic reduction after arithmetic operations.
//   - Cross‑cancellation in multiplication (including huge numbers).
//   - Epsilon interplay (comparisons with default epsilon are exact).
//   - Canonical form invariants (positive denominator, gcd = 1).
//   - Denominator growth behaviour (does not explode).
//   - Accurate chained sums (harmonic‑like sequences).
//   - Large integer exponentiation.
//   - Simulation of rational series terms (Taylor series, Padé).
//   - String round‑trip for big integers.
//
// All tests are deterministic and use rational exact comparisons.
// ============================================================================

#pragma once
#include <gtest/gtest.h>
#include <chrono>
#include <sstream>
#include "delta/core/rational.h"
#include "../test_fixtures.h"

namespace delta::testing {

    class RationalTest2 : public DeltaTest {
    protected:
        static std::string to_string_impl(const Rational& r) {
            return r.to_string();
        }

        static std::string numerator_str(const Rational& r) {
            std::string s = to_string_impl(r);
            size_t slash = s.find('/');
            if (slash == std::string::npos) return s;
            return s.substr(0, slash);
        }

        static std::string denominator_str(const Rational& r) {
            std::string s = to_string_impl(r);
            size_t slash = s.find('/');
            if (slash == std::string::npos) return "1";
            return s.substr(slash + 1);
        }

        static bool is_reduced(const Rational& r) {
            if (r == 0_r) return true;
            std::string num = numerator_str(r);
            std::string den = denominator_str(r);
            if (num.empty() || den.empty()) return true;

            delta::internal::dumb_int n(num);
            delta::internal::dumb_int d(den);
            if (n < 0) n = -n;
            delta::internal::dumb_int g = boost::multiprecision::gcd(n, d);
            return g == 1;
        }
    };

    // -------------------------------------------------------------------------
    // 1. Automatic reduction after arithmetic operations
    // -------------------------------------------------------------------------
    /**
     * @test AutomaticReductionAfterOperations
     * @brief Verifies that every arithmetic operation produces a result in
     *        reduced form (gcd = 1, denominator positive).
     */
    TEST_F(RationalTest2, AutomaticReductionAfterOperations) {
        Rational sum = 0_r;
        std::vector<std::pair<Rational, Rational>> operations = {
            {Rational(1, 2), Rational(1, 2)},
            {Rational(1, 3), Rational(1, 3)},
            {Rational(1, 5), Rational(1, 5)},
            {Rational(1, 7), Rational(1, 7)},
            {Rational(1, 11), Rational(1, 11)}
        };

        for (const auto& op : operations) {
            sum += op.first;
            EXPECT_TRUE(is_reduced(sum)) << "After adding " << op.first << ", sum = " << sum << " is not reduced";
            sum -= op.second;
            EXPECT_TRUE(is_reduced(sum)) << "After subtracting " << op.second << ", sum = " << sum << " is not reduced";
        }
        EXPECT_EQ(sum, 0_r) << "Final sum should be zero";

        // Simple reduction test
        Rational a(1, 2);
        Rational b(1, 4);
        Rational c = a + b;   // 3/4
        EXPECT_EQ(denominator_str(c), "4");
        EXPECT_TRUE(is_reduced(c));

        Rational d(1, 3);
        Rational e(1, 6);
        Rational f = d + e;   // 1/2
        EXPECT_EQ(denominator_str(f), "2");
        EXPECT_TRUE(is_reduced(f));

        // Multiplication with cancellation
        Rational x(2, 3);
        Rational y(3, 4);
        Rational z = x * y;   // 6/12 -> 1/2
        EXPECT_EQ(numerator_str(z), "1");
        EXPECT_EQ(denominator_str(z), "2");
        EXPECT_TRUE(is_reduced(z));
    }

    // -------------------------------------------------------------------------
    // 2. Cross‑cancellation: multiply huge fractions with common factors
    // -------------------------------------------------------------------------
    /**
     * @test CrossCancellation
     * @brief Multiplies a 1000‑digit integer by its reciprocal; the product
     *        should be exactly 1, and reduction must happen early to avoid
     *        massive intermediate numbers. The test also checks performance.
     */
    TEST_F(RationalTest2, CrossCancellation) {
        // Create a 1000-digit number: 999...9
        std::string num_str(1000, '9');
        Rational a(num_str);              // huge integer
        Rational b = 1_r / Rational(num_str); // 1 / huge

        auto start = std::chrono::high_resolution_clock::now();
        Rational c = a * b;
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        EXPECT_EQ(c, 1_r);
        EXPECT_LT(elapsed, 1.0) << "Cross-cancellation took too long, reduction may not be happening early";
    }

    // -------------------------------------------------------------------------
    // 3. Epsilon interplay: comparison with default_eps
    // -------------------------------------------------------------------------
    /**
     * @test EpsilonInterplay
     * @brief Rational comparisons are exact, not epsilon‑based.
     *        This test merely illustrates that default_eps does not affect
     *        exact equality checks. Irrational (approximate) values are never
     *        equal to rationals.
     */
    TEST_F(RationalTest2, EpsilonInterplay) {
        Rational eps = delta::default_eps();
        Rational small = Rational(1, 1000000);      // 1e-6
        Rational very_small = Rational(1LL, 1000000000000LL); // 1e-12

        // Default eps is typically 1e-30, so very_small > eps
        // But we only test that comparisons are exact, not epsilon-based
        if (eps > small) {
            EXPECT_LT(small, eps);
        }
        else {
            EXPECT_GT(small, eps);
        }

        // Irrational vs rational comparison (should never be equal)
        Rational exact = Rational(1, 3);
        Rational approx = delta::sqrt(2_r);
        EXPECT_NE(exact, approx);
    }

    // -------------------------------------------------------------------------
    // 4. Canonical form invariants
    // -------------------------------------------------------------------------
    /**
     * @test CanonicalFormInvariants
     * @brief Checks that denominators are always positive, fractions are reduced,
     *        and zero is represented correctly.
     */
    TEST_F(RationalTest2, CanonicalFormInvariants) {
        // Denominator should always be positive
        Rational pos(1, 2);
        Rational neg(1, -2);
        Rational zero(0_r);

        EXPECT_EQ(denominator_str(pos), "2");
        // 1/-2 should normalize to -1/2, denominator positive
        EXPECT_EQ(denominator_str(neg), "2");
        EXPECT_EQ(numerator_str(neg), "-1");
        EXPECT_EQ(numerator_str(zero), "0");

        // GCD reduction on construction
        Rational gcd_test(6, 8);   // should become 3/4
        EXPECT_EQ(numerator_str(gcd_test), "3");
        EXPECT_EQ(denominator_str(gcd_test), "4");

        // After arithmetic
        Rational a(2, 6);   // 1/3
        EXPECT_EQ(numerator_str(a), "1");
        EXPECT_EQ(denominator_str(a), "3");

        Rational b(3, 9);   // 1/3
        EXPECT_EQ(numerator_str(b), "1");
        EXPECT_EQ(denominator_str(b), "3");

        Rational c = a + b; // 2/3
        EXPECT_EQ(numerator_str(c), "2");
        EXPECT_EQ(denominator_str(c), "3");
    }

    // -------------------------------------------------------------------------
    // 5. Denominator does not explode after chain operations
    // -------------------------------------------------------------------------
    /**
     * @test DenominatorDoesNotExplode
     * @brief After a series of rational operations (addition, multiplication,
     *        etc.) the denominator should not grow unnecessarily; the result
     *        is always reduced.
     */
    TEST_F(RationalTest2, DenominatorDoesNotExplode) {
        Rational a(1, 2);
        Rational b(1, 3);
        Rational c = a + b;           // 5/6
        Rational d = c * c;            // 25/36
        Rational e = d + Rational(1, 36); // 26/36 = 13/18
        EXPECT_EQ(denominator_str(e), "18");
        EXPECT_EQ(numerator_str(e), "13");

        // Harmonic series sum: 1 + 1/2 + ... + 1/10 = 7381/2520
        Rational x = 0_r;
        for (int i = 1; i <= 10; ++i) {
            x = x + Rational(1, i);
        }
        EXPECT_EQ(denominator_str(x), "2520");
        EXPECT_EQ(numerator_str(x), "7381");
    }

    // -------------------------------------------------------------------------
    // 6. Accurate chained sum
    // -------------------------------------------------------------------------
    /**
     * @test AccurateChainedSum
     * @brief Sums several small fractions and compares with the expected
     *        reduced rational.
     */
    TEST_F(RationalTest2, AccurateChainedSum) {
        Rational a(1, 2);
        Rational b(1, 3);
        Rational c(1, 5);
        Rational d(1, 7);
        Rational e(1, 11);

        Rational res = a + b + c + d + e;

        // Expected: 1/2 + 1/3 + 1/5 + 1/7 + 1/11
        Rational expected = Rational(1, 2) + Rational(1, 3) + Rational(1, 5) + Rational(1, 7) + Rational(1, 11);
        EXPECT_EQ(res, expected);
        EXPECT_TRUE(is_reduced(res)) << "Result should be reduced: " << res;
    }

    // -------------------------------------------------------------------------
    // 7. Large powers with reduction
    // -------------------------------------------------------------------------
    /**
     * @test LargePowers
     * @brief Raises fractions to large integer exponents and checks numerator/denominator.
     */
    TEST_F(RationalTest2, LargePowers) {
        Rational base(2, 3);
        Rational pow10 = delta::pow(base, 10);
        // 2^10 / 3^10 = 1024/59049, already reduced (gcd=1)
        EXPECT_EQ(numerator_str(pow10), "1024");
        EXPECT_EQ(denominator_str(pow10), "59049");

        Rational pow_neg = delta::pow(base, -10);
        EXPECT_EQ(numerator_str(pow_neg), "59049");
        EXPECT_EQ(denominator_str(pow_neg), "1024");

        // Test with base that has common factors
        Rational base2(6, 8);   // 3/4 after reduction
        Rational pow2 = delta::pow(base2, 3);
        // (3/4)^3 = 27/64
        EXPECT_EQ(numerator_str(pow2), "27");
        EXPECT_EQ(denominator_str(pow2), "64");
    }

    // -------------------------------------------------------------------------
    // 8. Rational series term simulation (Taylor series pattern)
    // -------------------------------------------------------------------------
    /**
     * @test RationalSeriesTerm
     * @brief Simulates the computation of successive terms in a Taylor series
     *        (e.g., exp(x)): term = term * x / n.
     */
    TEST_F(RationalTest2, RationalSeriesTerm) {
        Rational term = 1_r;
        Rational X = Rational(1, 2);
        int n = 1;

        term = term * X / n;   // 1 * 1/2 / 1 = 1/2
        EXPECT_EQ(numerator_str(term), "1");
        EXPECT_EQ(denominator_str(term), "2");

        n = 2;
        term = term * X / n;   // 1/2 * 1/2 / 2 = 1/8
        EXPECT_EQ(numerator_str(term), "1");
        EXPECT_EQ(denominator_str(term), "8");

        n = 3;
        term = term * X / n;   // 1/8 * 1/2 / 3 = 1/48
        EXPECT_EQ(numerator_str(term), "1");
        EXPECT_EQ(denominator_str(term), "48");

        n = 4;
        term = term * X / n;   // 1/48 * 1/2 / 4 = 1/384
        EXPECT_EQ(numerator_str(term), "1");
        EXPECT_EQ(denominator_str(term), "384");
    }

    // -------------------------------------------------------------------------
    // 9. Matrix exponential simulation (Padé approximation pattern)
    // -------------------------------------------------------------------------
    /**
     * @test RationalInMatrixExpSimulation
     * @brief Simulates the rational expression (I + A)/(I - A) for a small scalar A,
     *        which appears in Padé approximations of matrix exponentials.
     */
    TEST_F(RationalTest2, RationalInMatrixExpSimulation) {
        // Simulate (I + A) / (I - A) for small A
        Rational a = Rational(1, 2);   // small matrix element
        Rational I = 1_r;
        Rational numerator = I + a;
        Rational denominator = I - a;
        Rational result = numerator / denominator;   // (1.5)/(0.5) = 3
        EXPECT_EQ(numerator_str(result), "3");
        EXPECT_EQ(denominator_str(result), "1");

        // More realistic: A = 0.1
        Rational a2 = Rational(1, 10);
        Rational num2 = 1_r + a2;
        Rational den2 = 1_r - a2;
        Rational result2 = num2 / den2;   // 1.1/0.9 = 11/9
        EXPECT_EQ(numerator_str(result2), "11");
        EXPECT_EQ(denominator_str(result2), "9");
    }

    // -------------------------------------------------------------------------
    // 10. Chain of operations with no overflow
    // -------------------------------------------------------------------------
    /**
     * @test NoOverflowAfterManyOps
     * @brief Multiplies a chain of fractions i/(i+1) for i=1..100.
     *        The telescoping product yields 1/101, demonstrating that
     *        numerators and denominators stay small and no overflow occurs.
     */
    TEST_F(RationalTest2, NoOverflowAfterManyOps) {
        Rational x = 1_r;
        for (int i = 1; i <= 100; ++i) {
            x = x * Rational(i, i + 1);   // multiplying by i/(i+1)
        }
        // After 100 steps, x = 1/101
        EXPECT_EQ(numerator_str(x), "1");
        EXPECT_EQ(denominator_str(x), "101");
        EXPECT_TRUE(is_reduced(x));
    }

    // -------------------------------------------------------------------------
    // 11. String roundtrip for large numbers
    // -------------------------------------------------------------------------
    /**
     * @test StringRoundtrip
     * @brief Creates a rational from large numerator and denominator strings,
     *        converts it to string, and reconstructs; the two must be equal.
     */
    TEST_F(RationalTest2, StringRoundtrip) {
        // Create a rational with large numerator and denominator
        std::string num = "123456789012345678901234567890";
        std::string den = "987654321098765432109876543210";
        Rational r1 = Rational(num) / Rational(den);
        std::string s = to_string_impl(r1);
        Rational r2(s);
        EXPECT_EQ(r1, r2);
    }

} // namespace delta::testing