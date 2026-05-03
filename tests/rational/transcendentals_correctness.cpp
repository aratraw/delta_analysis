// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/rational/transcendentals_correctness.cpp
// ============================================================================
// CORRECTNESS TESTS FOR TRANSCENDENTAL FUNCTIONS (EAGER AND LAZY)
// ============================================================================
//
// This file verifies the correctness of delta::Rational transcendental
// functions (sqrt, exp, log, sin, cos, pi, e, acos, asin, pow) by comparing
// against naive series implementations and checking fundamental identities.
// The tests cover:
//   - Basic eager computations (sqrt, exp, log, sin, cos, pi, e).
//   - Lazy expression construction and evaluation.
//   - Edge cases (zero, one, negative, very large/small arguments).
//   - Deeply nested compositions (eager and lazy).
//   - Varying precision (from 1e-2 down to 1e-100).
//   - Argument reduction (large angles, large exponents).
//   - Consistency between float and series paths for sin, cos, exp.
//   - Stress test: large lazy tree with many transcendental nodes.
//   - High‑precision accuracy benchmarks for π and √2 (up to 100 digits).
//   - Identities: sin(π)=0, cos(π/2)=0, sin²+cos²=1, exp(log(x))=x,
//     sqrt(x)²=x, cos(acos(x))=x, acos(x)+asin(x)=π/2.
//   - acos specifics: special values, monotonicity, derivative approximation.
//   - pow with rational exponents (e.g., 2^(1/2) ≡ √2).
//   - Lazy canonicalisation (Exp(Log(z)) → z).
//
// All tests use the global default epsilon where appropriate; for high‑precision
// tests explicit epsilon values are supplied.
// ============================================================================

#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "delta/core/rational.h"
#include "test_utils.h"
#include "lazy_rational_test_fixture.h"

namespace delta::testing {

    // -------------------------------------------------------------------------
    // Naive (reference) implementations using series / iteration.
    // Used to verify Delta's correctness at any precision.
    // -------------------------------------------------------------------------
    namespace {

        Rational naive_series_ln2(const Rational& eps) {
            Rational one(1);
            Rational three(3);
            Rational z = one / three;
            Rational z2 = z * z;
            Rational term = z, sum = term, n = one, two(2);
            while (true) {
                term = term * z2;
                n = n + two;
                sum = sum + term / n;
                if (abs(term) < eps) break;
            }
            return two * sum;
        }

        Rational naive_series_pi(const Rational& eps) {
            Rational one(1), five(5), two39(239);
            Rational sixteen(16), four(4), two(2);
            Rational a = one / five, a2 = a * a;
            Rational term = a, sum_atan5 = term, n = one;
            while (true) {
                term = term * (-a2);
                n = n + two;
                sum_atan5 = sum_atan5 + term / n;
                if (abs(term) < eps) break;
            }
            Rational b = one / two39, b2 = b * b;
            term = b;
            Rational sum_atan239 = term;
            n = one;
            while (true) {
                term = term * (-b2);
                n = n + two;
                sum_atan239 = sum_atan239 + term / n;
                if (abs(term) < eps) break;
            }
            return sixteen * sum_atan5 - four * sum_atan239;
        }

        Rational naive_series_sin(const Rational& x, const Rational& eps) {
            Rational one(1), two(2);
            Rational pi_val = naive_series_pi(eps), twopi = pi_val * two;
            Rational reduced = x;
            while (abs(reduced) > pi_val) {
                if (reduced > 0) reduced = reduced - twopi;
                else reduced = reduced + twopi;
            }
            Rational x2 = reduced * reduced;
            Rational term = reduced, sum = term, k = one;
            while (true) {
                term = term * (-x2);
                term = term / (two * k * (two * k + one));
                sum = sum + term;
                k = k + one;
                if (abs(term) < eps) break;
            }
            return sum;
        }

        Rational naive_series_cos(const Rational& x, const Rational& eps) {
            Rational one(1), two(2);
            Rational pi_val = naive_series_pi(eps), twopi = pi_val * two;
            Rational reduced = x;
            while (abs(reduced) > pi_val) {
                if (reduced > 0) reduced = reduced - twopi;
                else reduced = reduced + twopi;
            }
            Rational x2 = reduced * reduced;
            Rational term = one, sum = term, k = one;
            while (true) {
                term = term * (-x2);
                term = term / ((two * k - one) * (two * k));
                sum = sum + term;
                k = k + one;
                if (abs(term) < eps) break;
            }
            return sum;
        }

        Rational naive_series_exp(const Rational& x, const Rational& eps) {
            Rational one(1), two(2);
            int k = 0;
            Rational reduced = x;
            while (abs(reduced) > one) {
                reduced = reduced / two;
                ++k;
            }
            Rational sum = one, term = one, n = one;
            while (true) {
                term = term * reduced / n;
                sum = sum + term;
                n = n + one;
                if (abs(term) < eps) break;
            }
            Rational result = sum;
            for (int i = 0; i < k; ++i) result = result * result;
            return result;
        }

        Rational naive_series_log(const Rational& x, const Rational& eps) {
            Rational one(1), two(2), half = one / two;
            int k = 0;
            Rational m = x;
            while (m > two) { m = m / two; ++k; }
            while (m < half) { m = m * two; --k; }
            Rational ln2 = naive_series_ln2(eps);
            Rational y = (m - one) / (m + one);
            Rational y2 = y * y;
            Rational term = y, sum = term, n = one;
            while (true) {
                term = term * y2;
                n = n + two;
                sum = sum + term / n;
                if (abs(term) < eps) break;
            }
            Rational ln_m = two * sum;
            return ln_m + Rational(k) * ln2;
        }

        Rational naive_series_sqrt(const Rational& x, const Rational& eps) {
            if (x == 0_r) return 0_r;
            if (x < 0_r) throw std::domain_error("sqrt of negative number");
            Rational one(1), two(2);
            Rational guess = x / two, diff;
            size_t iter = 0;
            const size_t max_iter = 1000;
            do {
                Rational next = (guess + x / guess) / two;
                diff = abs(next - guess);
                guess = next;
                ++iter;
                if (iter > max_iter) break;
            } while (diff > eps);
            return guess;
        }

        Rational naive_series_e(const Rational& eps) {
            Rational one(1);
            Rational sum = one, term = one, n = one;
            while (true) {
                term = term / n;
                sum = sum + term;
                n = n + one;
                if (term < eps) break;
            }
            return sum;
        }

        Rational naive_series_acos(const Rational& x, const Rational& eps) {
            Rational one(1), two(2);
            Rational pi_val = naive_series_pi(eps);
            Rational half_pi = pi_val / two;
            if (x < -one || x > one)
                throw std::domain_error("acos argument out of [-1,1]");
            Rational y = (x > 0) ? half_pi * (one - x) : pi_val - half_pi * (one + x);
            const size_t max_iter = 100;
            size_t iter = 0;
            while (iter < max_iter) {
                Rational cos_y = naive_series_cos(y, eps);
                Rational sin_y = naive_series_sin(y, eps);
                if (sin_y == 0_r) break;
                Rational delta = (cos_y - x) / sin_y;
                y = y - delta;
                if (abs(delta) < eps) break;
                ++iter;
            }
            return y;
        }

        Rational naive_series_pow(const Rational& base, const Rational& exp, const Rational& eps) {
            if (base == 0_r) {
                if (exp == 0_r) throw std::domain_error("0^0 is undefined");
                if (exp < 0_r) throw std::domain_error("0^negative is undefined");
                return 0_r;
            }
            if (base == 1_r) return 1_r;
            if (exp == 0_r) return 1_r;

            // Check if exponent is an integer
            auto is_integer = [](const Rational& r) {
                return r.denominator() == 1_r;
                };
            if (is_integer(exp)) {
                auto exp_int = exp.numerator().convert_to<int>();
                if (exp_int < 0) {
                    Rational base_recip = 1_r / base;
                    return pow(base_recip, -exp_int);
                }
                Rational result = 1_r, b = base;
                int e = exp_int;
                while (e > 0) {
                    if (e & 1) result = result * b;
                    e >>= 1;
                    if (e != 0) b = b * b;
                }
                return result;
            }

            Rational log_base = naive_series_log(base, eps / 1000);
            Rational p_log = exp * log_base;
            return naive_series_exp(p_log, eps / 1000);
        }

    } // namespace

    // -----------------------------------------------------------------------------
    // Fixture for correctness tests
    // -----------------------------------------------------------------------------
    class TranscendentalCorrectnessTest : public LazyRationalTestFixture {
    protected:
        // Commonly used epsilons
        const Rational EPS_STD = "1/1000000000000"_r;           // 1e-12
        const Rational EPS_HIGH = "1/1000000000000000000000"_r; // 1e-21
        const Rational EPS_ULTRA = "1/10000000000000000000000000000000000000000"_r; // 1e-40
        const Rational EPS_EXTREME = "1/10000000000000000000000000000000000000000000000000000000000000000000000000000000"_r; // 1e-80

        // Check that Delta's result is close to the naive reference with given eps
        void expect_near_naive(const Rational& delta_val,
            const Rational& naive_val,
            const Rational& eps,
            const std::string& msg = "") {
            EXPECT_RATIONAL_NEAR(delta_val, naive_val, eps) << msg;
        }

        template<typename DeltaFunc, typename NaiveFunc>
        void test_function_against_naive(DeltaFunc delta_func,
            NaiveFunc naive_func,
            const Rational& arg,
            const Rational& eps,
            const std::string& func_name) {
            Rational delta_res = delta_func(arg, eps);
            Rational naive_res = naive_func(arg, eps);
            Rational tolerance = eps * 10; // small safety margin due to different strategies
            EXPECT_RATIONAL_NEAR(delta_res, naive_res, tolerance)
                << func_name << "(" << arg << ") with eps=" << eps;
        }
    };

    // =============================================================================
    // 1. BASIC EAGER TESTS: accuracy at standard values
    // =============================================================================

    /**
     * @test EagerSqrt
     * @brief Checks sqrt(4)=2 exactly and sqrt(2) to 1e‑12.
     */
    TEST_F(TranscendentalCorrectnessTest, EagerSqrt) {
        Rational s4 = delta::sqrt(4_r);
        EXPECT_EQ(s4, 2_r);

        Rational s2 = delta::sqrt(2_r);
        Rational expected_sqrt2 = Rational("14142135623730950488/10000000000000000000");
        EXPECT_RATIONAL_NEAR(s2, expected_sqrt2, EPS_STD);
        if (!HasFailure()) {
            std::cout << "SQRT: Eager sqrt computes exact integer roots and accurate sqrt(2) to standard precision." << std::endl;
        }
    }

    /**
     * @test EagerExp
     * @brief Checks exp(0)=1 and exp(1) approximates e.
     */
    TEST_F(TranscendentalCorrectnessTest, EagerExp) {
        Rational e0 = delta::exp(0_r);
        EXPECT_EQ(e0, 1_r);

        Rational e1 = delta::exp(1_r);
        Rational expected_e = Rational("27182818284590452354/10000000000000000000");
        EXPECT_RATIONAL_NEAR(e1, expected_e, EPS_STD);
        if (!HasFailure()) {
            std::cout << "EXP: Eager exp handles exp(0)=1 and exp(1) approximates e correctly." << std::endl;
        }
    }

    /**
     * @test EagerLog
     * @brief Checks log(1)=0 and log(2) to 1e‑12.
     */
    TEST_F(TranscendentalCorrectnessTest, EagerLog) {
        Rational l1 = delta::log(1_r);
        EXPECT_EQ(l1, 0_r);

        Rational l2 = delta::log(2_r);
        Rational expected_log2 = Rational("69314718055994530942/100000000000000000000");
        EXPECT_RATIONAL_NEAR(l2, expected_log2, EPS_STD);
        if (!HasFailure()) {
            std::cout << "LOG: Eager log returns exact log(1)=0 and accurate log(2) to standard precision." << std::endl;
        }
    }

    /**
     * @test EagerSinCos
     * @brief Checks sin(0)=0, cos(0)=1.
     */
    TEST_F(TranscendentalCorrectnessTest, EagerSinCos) {
        Rational s0 = delta::sin(0_r);
        EXPECT_EQ(s0, 0_r);

        Rational c0 = delta::cos(0_r);
        EXPECT_EQ(c0, 1_r);
        if (!HasFailure()) {
            std::cout << "SIN/COS: Eager sin and cos return exact values at x=0 (sin=0, cos=1)." << std::endl;
        }
    }

    /**
     * @test EagerPiE
     * @brief Verifies pi() and e() approximations.
     */
    TEST_F(TranscendentalCorrectnessTest, EagerPiE) {
        Rational p = delta::pi();
        Rational expected_pi = Rational("31415926535897932384626433832795028841971693993751/10000000000000000000000000000000000000000000000000");
        EXPECT_RATIONAL_NEAR(p, expected_pi, EPS_STD);

        Rational e = delta::e();
        Rational expected_e = Rational("27182818284590452353602874713526624977572470936996/10000000000000000000000000000000000000000000000000");
        EXPECT_RATIONAL_NEAR(e, expected_e, EPS_STD);
        if (!HasFailure()) {
            std::cout << "PI/E: Eager pi and e both match high-precision reference values to 1e-12 tolerance." << std::endl;
        }
    }

    // =============================================================================
    // 2. LAZY TESTS: construction and evaluation of delayed expressions
    // =============================================================================

    /**
     * @test LazySqrt
     * @brief Lazy sqrt creates a SQRT node and evaluates correctly.
     */
    TEST_F(TranscendentalCorrectnessTest, LazySqrt) {
        auto s = delta::lazy_sqrt(2_r);
        static_assert(std::is_same_v<decltype(s), LazyRational>);
        Rational expected_sqrt2 = Rational("14142135623730950488/10000000000000000000");
        EXPECT_RATIONAL_NEAR(s.eval(), expected_sqrt2, EPS_STD);
        if (!HasFailure()) {
            std::cout << "LAZY SQRT: Lazy sqrt creates correct node type and evaluates to accurate sqrt(2)." << std::endl;
        }
    }

    /**
     * @test LazyExp
     * @brief Lazy exp creates an EXP node and evaluates correctly.
     */
    TEST_F(TranscendentalCorrectnessTest, LazyExp) {
        auto e = delta::lazy_exp(1_r);
        static_assert(std::is_same_v<decltype(e), LazyRational>);
        Rational expected_e = Rational("27182818284590452354/10000000000000000000");
        EXPECT_RATIONAL_NEAR(e.eval(), expected_e, EPS_STD);
        if (!HasFailure()) {
            std::cout << "LAZY EXP: Lazy exp creates correct node type and evaluates to accurate e^1." << std::endl;
        }
    }

    /**
     * @test LazyPi
     * @brief Lazy pi creates a PI node and evaluates correctly.
     */
    TEST_F(TranscendentalCorrectnessTest, LazyPi) {
        auto p = delta::lazy_pi();
        static_assert(std::is_same_v<decltype(p), LazyRational>);
        Rational expected_pi = Rational("31415926535897932384626433832795028841971693993751/10000000000000000000000000000000000000000000000000");
        EXPECT_RATIONAL_NEAR(p.eval(), expected_pi, EPS_STD);
        if (!HasFailure()) {
            std::cout << "LAZY PI: Lazy pi creates correct node type and evaluates to accurate pi." << std::endl;
        }
    }

    // =============================================================================
    // 3. EDGE CASES: special values and extreme situations
    // =============================================================================

    /**
     * @test SqrtEdgeCases
     * @brief Handles 0, 1, negative (throws), and large arguments.
     */
    TEST_F(TranscendentalCorrectnessTest, SqrtEdgeCases) {
        EXPECT_EQ(delta::sqrt(0_r), 0_r);
        EXPECT_EQ(delta::sqrt(1_r), 1_r);
        EXPECT_THROW(delta::sqrt(-1_r), std::domain_error);
        Rational big = "10000000000000000000000000000000000000000000000000"_r;
        Rational sqrt_big = delta::sqrt(big);
        EXPECT_TRUE(sqrt_big > 0_r);
        EXPECT_RATIONAL_NEAR(sqrt_big * sqrt_big, big, EPS_STD);
        if (!HasFailure()) {
            std::cout << "SQRT EDGE CASES: Handles 0, 1, negative (throws), and large numbers correctly." << std::endl;
        }
    }

    /**
     * @test ExpEdgeCases
     * @brief Checks exp(0)=1, large positive/negative arguments.
     */
    TEST_F(TranscendentalCorrectnessTest, ExpEdgeCases) {
        EXPECT_EQ(delta::exp(0_r), 1_r);
        Rational large = 100_r;
        Rational exp_large = delta::exp(large);
        EXPECT_TRUE(exp_large > 1_r);
        Rational exp_neg = delta::exp(-large);
        EXPECT_RATIONAL_NEAR(exp_neg * exp_large, 1_r, EPS_STD);
        if (!HasFailure()) {
            std::cout << "EXP EDGE CASES: Handles exp(0)=1 and large positive/negative arguments with reciprocity." << std::endl;
        }
    }

    /**
     * @test LogEdgeCases
     * @brief Checks log(1)=0, rejects ≤0, and inverts exp.
     */
    TEST_F(TranscendentalCorrectnessTest, LogEdgeCases) {
        EXPECT_EQ(delta::log(1_r), 0_r);
        EXPECT_THROW(delta::log(0_r), std::domain_error);
        EXPECT_THROW(delta::log(-1_r), std::domain_error);
        Rational x = "3.1415926535"_r;
        Rational log_exp = delta::log(delta::exp(x));
        EXPECT_RATIONAL_NEAR(log_exp, x, EPS_STD);
        if (!HasFailure()) {
            std::cout << "LOG EDGE CASES: Handles log(1)=0, rejects non-positive arguments, and inverts exp correctly." << std::endl;
        }
    }

    /**
     * @test SinCosEdgeCases
     * @brief Checks parity, periodicity at π, and odd/even properties.
     */
    TEST_F(TranscendentalCorrectnessTest, SinCosEdgeCases) {
        EXPECT_EQ(delta::sin(0_r), 0_r);
        EXPECT_EQ(delta::cos(0_r), 1_r);
        Rational pi_val = delta::pi();
        EXPECT_RATIONAL_NEAR(delta::sin(pi_val), 0_r, EPS_STD);
        EXPECT_RATIONAL_NEAR(delta::cos(pi_val), -1_r, EPS_STD);
        Rational x = "1.5"_r;
        EXPECT_EQ(delta::sin(-x), -delta::sin(x));
        EXPECT_EQ(delta::cos(-x), delta::cos(x));
        if (!HasFailure()) {
            std::cout << "SIN/COS EDGE CASES: Satisfies sin(0)=0, cos(0)=1, sin(pi)=0, cos(pi)=-1, and odd/even properties." << std::endl;
        }
    }

    /**
     * @test PowEdgeCases
     * @brief Checks integer and zero exponents, and the additive property.
     */
    TEST_F(TranscendentalCorrectnessTest, PowEdgeCases) {
        EXPECT_EQ(delta::pow(2_r, 0_r), 1_r);
        EXPECT_EQ(delta::pow(0_r, 5_r), 0_r);
        EXPECT_THROW(delta::pow(0_r, 0_r), std::domain_error);
        EXPECT_THROW(delta::pow(0_r, -1_r), std::domain_error);
        Rational base = "2.5"_r;
        Rational a = "1.2"_r;
        Rational b = "0.7"_r;
        Rational p1 = delta::pow(base, a + b);
        Rational p2 = delta::pow(base, a) * delta::pow(base, b);
        EXPECT_RATIONAL_NEAR(p1, p2, EPS_STD);
        if (!HasFailure()) {
            std::cout << "POW EDGE CASES: Correctly handles zero, negative, and fractional exponents with additivity property." << std::endl;
        }
    }

    // =============================================================================
    // 4. DEEPLY NESTED EXPRESSIONS: composition of functions
    // =============================================================================

    /**
     * @test DeeplyNestedEager
     * @brief Eager composition sin(cos(exp(log(1+x)))) for x=0.5.
     */
    TEST_F(TranscendentalCorrectnessTest, DeeplyNestedEager) {
        auto f = [](const Rational& x) -> Rational {
            return delta::sin(delta::cos(delta::exp(delta::log(1_r + x))));
            };
        Rational x = "0.5"_r;
        Rational result = f(x);
        EXPECT_TRUE(result > -2_r && result < 2_r);
        if (!HasFailure()) {
            std::cout << "DEEPLY NESTED EAGER: Composite sin(cos(exp(log(1+x)))) evaluates without errors." << std::endl;
        }
    }

    /**
     * @test DeeplyNestedLazy
     * @brief Lazy composition Sin(Cos(Exp(Log(1+x)))) builds correct nodes.
     */
    TEST_F(TranscendentalCorrectnessTest, DeeplyNestedLazy) {
        using namespace delta;
        LazyRational x = LazyRational("0.5"_r);
        LazyRational expr = Sin(Cos(Exp(Log(1_r + x))));
        EXPECT_TRUE(has_node_with_op(expr, internal::LazyOp::SIN));
        EXPECT_TRUE(has_node_with_op(expr, internal::LazyOp::COS));
        EXPECT_TRUE(has_node_with_op(expr, internal::LazyOp::EXP));
        EXPECT_TRUE(has_node_with_op(expr, internal::LazyOp::LOG));
        Rational result = expr.eval();
        EXPECT_TRUE(result > -2_r && result < 2_r);
        if (!HasFailure()) {
            std::cout << "DEEPLY NESTED LAZY: Composite lazy expression builds correct node types and evaluates within range." << std::endl;
        }
    }

    /**
     * @test MixedEagerLazyDeep
     * @brief Combines eager constants with lazy expressions.
     */
    TEST_F(TranscendentalCorrectnessTest, MixedEagerLazyDeep) {
        Rational a = 2_r;
        LazyRational b = LazyRational(3_r);
        LazyRational c = a.as_lazy() * Sin(b) + Cos(b) * Exp(b);
        Rational eager_ver = 2_r * sin(3_r) + cos(3_r) * exp(3_r);
        Rational lazy_ver = c.eval();
        EXPECT_RATIONAL_NEAR(lazy_ver, eager_ver, EPS_STD);
        if (!HasFailure()) {
            std::cout << "MIXED EAGER/LAZY: Mixed eager-lazy expression evaluates identically to pure eager computation." << std::endl;
        }
    }

    // =============================================================================
    // 5. VARYING PRECISION: convergence with decreasing eps
    // =============================================================================

    /**
     * @test VaryingPrecisionSin
     * @brief Checks that sin(1) converges as epsilon decreases.
     */
    TEST_F(TranscendentalCorrectnessTest, VaryingPrecisionSin) {
        const Rational x = 1_r;
        std::vector<Rational> epsilons = {
            "1/100"_r,
            "1/1000000"_r,
            "1/1000000000000"_r,
            "1/1000000000000000000"_r
        };
        Rational prev = 0_r;
        Rational prev_eps = 0_r;
        for (const auto& eps : epsilons) {
            Rational s = delta::sin(x, eps);
            if (prev != 0_r) {
                Rational diff = delta::abs(s - prev);
                EXPECT_LT(diff.to_double(), 2 * prev_eps.to_double());
            }
            prev = s;
            prev_eps = eps;
        }
        if (!HasFailure()) {
            std::cout << "VARYING PRECISION SIN: Sin(1) converges stably as eps decreases from 1e-2 to 1e-18." << std::endl;
        }
    }

    // =============================================================================
    // 6. LAZY SHORT NAMES: syntactic sugar correctness
    // =============================================================================

    /**
     * @test LazyShortNamesCreateCorrectNodes
     * @brief Verifies that Sin, Cos, Pi, Exp create the expected node types.
     */
    TEST_F(TranscendentalCorrectnessTest, LazyShortNamesCreateCorrectNodes) {
        using namespace delta;
        LazyRational a = LazyRational(2_r);
        LazyRational expr = Sin(a) + Cos(Pi() / 2_r) * Exp(1_r);

        EXPECT_TRUE(has_node_with_op(expr, internal::LazyOp::SIN));
        EXPECT_TRUE(has_node_with_op(expr, internal::LazyOp::COS));
        EXPECT_TRUE(has_node_with_op(expr, internal::LazyOp::PI));
        EXPECT_TRUE(has_node_with_op(expr, internal::LazyOp::EXP));
        EXPECT_TRUE(has_node_with_op(expr, internal::LazyOp::SUM));
        EXPECT_TRUE(has_node_with_op(expr, internal::LazyOp::PRODUCT));

        Rational expected = sin(2_r) + cos(pi() / 2_r) * exp(1_r);
        EXPECT_RATIONAL_NEAR(expr.eval(), expected, EPS_STD);
        if (!HasFailure()) {
            std::cout << "LAZY SHORT NAMES: Sin, Cos, Pi, Exp syntactic sugar creates correct nodes and matches eager result." << std::endl;
        }
    }

    // =============================================================================
    // 7. ARGUMENT REDUCTION: periodicity and properties for large x
    // =============================================================================

    /**
     * @test SinCosLargeAngles
     * @brief Checks that sin(100π)=0, cos(50π)=1, cos(51π)=-1.
     */
    TEST_F(TranscendentalCorrectnessTest, SinCosLargeAngles) {
        const Rational pi_val = delta::pi();
        Rational s = delta::sin(100_r * pi_val);
        EXPECT_RATIONAL_NEAR(s, 0_r, EPS_STD);
        Rational c = delta::cos(50_r * pi_val);
        EXPECT_RATIONAL_NEAR(c, 1_r, EPS_STD);
        c = delta::cos(51_r * pi_val);
        EXPECT_RATIONAL_NEAR(c, -1_r, EPS_STD);
        if (!HasFailure()) {
            std::cout << "SIN/COS LARGE ANGLES: Correctly reduces 100*pi to sin=0 and 50*pi, 51*pi to cos=±1." << std::endl;
        }
    }

    /**
     * @test ExpLargeArgumentScaling
     * @brief Checks exp(x)^2 ≈ exp(2x) for x=100.
     */
    TEST_F(TranscendentalCorrectnessTest, ExpLargeArgumentScaling) {
        const Rational big = 100_r;
        const Rational eps = EPS_STD;
        Rational e_big = delta::exp(big, eps);
        Rational e_big_squared = delta::exp(2_r * big, eps);
        Rational product = e_big * e_big;
        Rational rel_diff = delta::abs(product - e_big_squared) / delta::abs(e_big_squared);
        EXPECT_LT(rel_diff.to_double(), 1e-10);
        if (!HasFailure()) {
            std::cout << "EXP LARGE ARGUMENT: For x=100, exp(x)^2 agrees with exp(2x) with relative error < 1e-10." << std::endl;
        }
    }

    /**
     * @test LogLargeArgumentScaling
     * @brief Checks log(x²)=2log(x) for x=100000.
     */
    TEST_F(TranscendentalCorrectnessTest, LogLargeArgumentScaling) {
        const Rational big = 100000_r;
        const Rational eps = EPS_STD;
        Rational log_big = delta::log(big, eps);
        Rational log_big2 = delta::log(big * big, eps);
        EXPECT_RATIONAL_NEAR(2_r * log_big, log_big2, eps);
        if (!HasFailure()) {
            std::cout << "LOG LARGE ARGUMENT: log(100000^2) = 2*log(100000) with additive error < 1e-12." << std::endl;
        }
    }

    // =============================================================================
    // 8. FLOAT vs SERIES PATH CONSISTENCY
    // =============================================================================

    /**
     * @test FloatVsSeriesConsistencySin
     * @brief Compares float‑path (eps=1e-21) and series‑path (eps=1e-40) for sin(2).
     */
    TEST_F(TranscendentalCorrectnessTest, FloatVsSeriesConsistencySin) {
        Rational x = 2_r;
        Rational float_sin = delta::sin(x, EPS_HIGH);   // eps=1e-21 -> float path
        Rational series_sin = delta::sin(x, EPS_ULTRA); // eps=1e-40 -> series path
        Rational diff = delta::abs(float_sin - series_sin);
        EXPECT_LT(diff.to_double(), 1e-18);
        if (!HasFailure()) {
            std::cout << "FLOAT VS SERIES SIN: Float and series paths for sin(2) agree within 1e-18." << std::endl;
        }
    }

    /**
     * @test FloatVsSeriesConsistencyExp
     * @brief Compares float‑path and series‑path for exp(1.5).
     */
    TEST_F(TranscendentalCorrectnessTest, FloatVsSeriesConsistencyExp) {
        Rational x = "1.5"_r;
        Rational float_exp = delta::exp(x, EPS_HIGH);
        Rational series_exp = delta::exp(x, EPS_ULTRA);
        Rational diff = delta::abs(float_exp - series_exp);
        EXPECT_LT(diff.to_double(), 1e-18);
        if (!HasFailure()) {
            std::cout << "FLOAT VS SERIES EXP: Float and series paths for exp(1.5) agree within 1e-18." << std::endl;
        }
    }

    // =============================================================================
    // 9. LAZY STRESS TEST: large tree with many transcendental nodes
    // =============================================================================

    /**
     * @test LazyTreeWithManyNodes
     * @brief Builds 3000 nodes (Sin, Cos, Exp) and evaluates.
     */
    TEST_F(TranscendentalCorrectnessTest, LazyTreeWithManyNodes) {
        internal::reset_pool();
        LazyRational acc = LazyRational();
        const int N = 1000;
        for (int i = 0; i < N; ++i) {
            Rational x = Rational(i) / 1000_r;
            acc + Sin(x) + Cos(x) + Exp(x);
        }
        Rational result = acc.eval();
        EXPECT_TRUE(result > 0_r);
        EXPECT_TRUE(acc.is_clean());
        if (!HasFailure()) {
            std::cout << "LAZY TREE STRESS: Successfully built and evaluated a tree with 3000 transcendental nodes." << std::endl;
        }
    }

    // =============================================================================
    // 10. HIGH‑PRECISION BENCHMARKS: pi and sqrt(2)
    // =============================================================================

    /**
     * @test PiPrecisionBenchmark
     * @brief Computes π with eps from 1e-20 to 1e-100 and checks error bound.
     */
    TEST_F(TranscendentalCorrectnessTest, PiPrecisionBenchmark) {
        // Reference π (first 100 digits) as Rational
        const Rational pi_ref(
            "31415926535897932384626433832795028841971693993751"
            "05820974944592307816406286208998628034825342117068"
            "/10000000000000000000000000000000000000000000000000"
            "00000000000000000000000000000000000000000000000000"
        );

        struct TestCase {
            std::string name;
            Rational eps;
            int expected_digits;
        };

        std::vector<TestCase> test_cases = {
            {"1e-20",  "1/100000000000000000000"_r, 20},
            {"1e-30",  "1/1000000000000000000000000000000"_r, 30},
            {"1e-40",  "1/10000000000000000000000000000000000000000"_r, 40},
            {"1e-50",  "1/100000000000000000000000000000000000000000000000000"_r, 50},
            {"1e-60",  "1/1000000000000000000000000000000000000000000000000000000000000"_r, 60},
            {"1e-70",  "1/10000000000000000000000000000000000000000000000000000000000000000000000"_r, 70},
            {"1e-80",  "1/10000000000000000000000000000000000000000000000000000000000000000000000000000000"_r, 80},
            {"1e-90",  "1/10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"_r, 90},
            {"1e-100", "1/10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"_r, 100},
        };

        for (const auto& tc : test_cases) {
            Rational pi_computed = delta::pi(tc.eps);
            Rational error = delta::abs(pi_computed - pi_ref);

            // Error should be less than requested eps with safety margin
            EXPECT_TRUE(error < tc.eps * 10)
                << "Failed for eps=" << tc.name
                << " (expected " << tc.expected_digits << " digits)"
                << "\n  Error = " << std::scientific << error.to_double()
                << "\n  Tolerance = " << (tc.eps * 10).to_double();
        }
        if (!HasFailure()) {
            std::cout << "PI PRECISION: Computes pi to 100 correct digits across eps from 1e-20 to 1e-100." << std::endl;
        }
    }

    /**
     * @test Sqrt2PrecisionBenchmark
     * @brief Computes √2 with eps from 1e-20 to 1e-100.
     */
    TEST_F(TranscendentalCorrectnessTest, Sqrt2PrecisionBenchmark) {
        // Reference √2 (first 100 digits) as Rational
        const Rational sqrt2_ref(
            "1.4142135623730950488016887242096980785696718753769480731766797379907324784621070388503875343276415727"
        );

        struct TestCase {
            std::string name;
            Rational eps;
            int expected_digits;
        };

        std::vector<TestCase> test_cases = {
            {"1e-20",  "1/100000000000000000000"_r, 20},
            {"1e-30",  "1/1000000000000000000000000000000"_r, 30},
            {"1e-40",  "1/10000000000000000000000000000000000000000"_r, 40},
            {"1e-50",  "1/100000000000000000000000000000000000000000000000000"_r, 50},
            {"1e-60",  "1/1000000000000000000000000000000000000000000000000000000000000"_r, 60},
            {"1e-70",  "1/10000000000000000000000000000000000000000000000000000000000000000000000"_r, 70},
            {"1e-80",  "1/10000000000000000000000000000000000000000000000000000000000000000000000000000000"_r, 80},
            {"1e-90",  "1/10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"_r, 90},
            {"1e-100", "1/10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"_r, 100},
        };

        for (const auto& tc : test_cases) {
            Rational sqrt2_computed = delta::sqrt(Rational(2), tc.eps);
            Rational error = delta::abs(sqrt2_computed - sqrt2_ref);

            EXPECT_TRUE(error < tc.eps * 10)
                << "Failed for eps=" << tc.name
                << " (expected " << tc.expected_digits << " digits)"
                << "\n  Error = " << std::scientific << error.to_double()
                << "\n  Tolerance = " << (tc.eps * 10).to_double();
        }
        if (!HasFailure()) {
            std::cout << "SQRT: CORRECT, UP TO 1e-100 and beyond." << std::endl;
        }
    }

    // -----------------------------------------------------------------------------
    // Identity check: sin(π) = 0 for high precisions
    // -----------------------------------------------------------------------------
    /**
     * @test PiSinConsistency
     * @brief Checks that sin(π) < 1000*eps for various eps.
     */
    TEST_F(TranscendentalCorrectnessTest, PiSinConsistency) {
        internal::reset_default_eps();
        std::vector<Rational> epsilons = {
            "1/1000000000000000000000000000000"_r,        // 1e-30
            "1/10000000000000000000000000000000000000000"_r, // 1e-40
            "1/100000000000000000000000000000000000000000000000000"_r, // 1e-50
            "1/1000000000000000000000000000000000000000000000000000000000000"_r, // 1e-60
        };

        std::cout << "\n=== PI * SIN(PI) CONSISTENCY CHECK ===\n";
        for (const auto& eps : epsilons) {
            Rational pi_val = delta::pi(eps);
            Rational sin_pi = delta::sin(pi_val, eps);

            std::cout << "eps=" << std::setw(5) << -std::log10(eps.to_double())
                << ": sin(pi) = " << sin_pi.to_double() << "\n";

            // sin(π) should be very close to 0
            EXPECT_RATIONAL_NEAR(sin_pi, 0_r, eps * 1000)
                << "sin(pi) should be close to 0 for eps=" << eps;
        }
        if (!HasFailure()) {
            std::cout << "PI-SIN CONSISTENCY: sin(pi) < 1000*eps for eps from 1e-30 to 1e-60, confirming pi and sin are mutually consistent." << std::endl;
        }
    }

    // -----------------------------------------------------------------------------
    // Identity check: cos(π/2) = 0 for high precisions
    // -----------------------------------------------------------------------------
    /**
     * @test PiCosConsistency
     * @brief Checks that cos(π/2) < 1000*eps for various eps.
     */
    TEST_F(TranscendentalCorrectnessTest, PiCosConsistency) {
        std::vector<Rational> epsilons = {
            "1/1000000000000000000000000000000"_r,        // 1e-30
            "1/10000000000000000000000000000000000000000"_r, // 1e-40
            "1/100000000000000000000000000000000000000000000000000"_r, // 1e-50
        };

        std::cout << "\n=== PI * COS(PI/2) CONSISTENCY CHECK ===\n";
        for (const auto& eps : epsilons) {
            Rational pi_val = delta::pi(eps);
            Rational half_pi = pi_val / 2_r;
            Rational cos_half_pi = delta::cos(half_pi, eps);

            std::cout << "eps=" << std::setw(5) << -std::log10(eps.to_double())
                << ": cos(pi/2) = " << cos_half_pi.to_double() << "\n";

            // cos(π/2) should be very close to 0
            EXPECT_RATIONAL_NEAR(cos_half_pi, 0_r, eps * 1000)
                << "cos(pi/2) should be close to 0 for eps=" << eps;
        }
        if (!HasFailure()) {
            std::cout << "PI-COS CONSISTENCY: cos(pi/2) < 1000*eps for eps from 1e-30 to 1e-50, confirming pi and cos are mutually consistent." << std::endl;
        }
    }

    // -----------------------------------------------------------------------------
    // Special note: Newton's method for sqrt converges quadratically,
    // so a low‑precision request may still give high accuracy.
    // Hence we test the actual squared error.
    // -----------------------------------------------------------------------------
    /**
     * @test SQRTPrecisionParameter
     * @brief Verifies that requested eps is respected for sqrt(2).
     */
    TEST_F(TranscendentalCorrectnessTest, SQRTPrecisionParameter) {
        Rational x = 2_r;
        Rational eps_low = "1/10"_r;
        Rational eps_high = "1/1000000000000000000000000000000"_r;

        Rational low = delta::sqrt(x, eps_low);
        Rational high = delta::sqrt(x, eps_high);

        // Check |sqrt(x)² - x| < eps for each case
        Rational diff_low = delta::abs(low * low - x);
        Rational diff_high = delta::abs(high * high - x);

        EXPECT_TRUE(diff_low < eps_low)
            << "diff_low=" << diff_low.to_double() << " >= eps_low=" << eps_low.to_double();
        EXPECT_TRUE(diff_high < eps_high)
            << "diff_high=" << diff_high.to_double() << " >= eps_high=" << eps_high.to_double();

        // Tighter eps gives smaller error
        EXPECT_TRUE(diff_high < diff_low);
        if (!HasFailure()) {
            std::cout << "SQRT PRECISION PARAMETER: Both low (1e-1) and high (1e-30) eps produce sqrt satisfying requested tolerances, tighter eps yields better result." << std::endl;
        }
    }

    // =============================================================================
    // 11. FUNCTION ACOS: basic checks and precision
    // =============================================================================

    /**
     * @test AcosBasic
     * @brief Checks acos at 0, ±1, and out‑of‑range.
     */
    TEST_F(TranscendentalCorrectnessTest, AcosBasic) {
        EXPECT_RATIONAL_NEAR(delta::acos(0_r), delta::pi() / 2_r, EPS_STD);
        EXPECT_EQ(delta::acos(1_r), 0_r);
        EXPECT_RATIONAL_NEAR(delta::acos(-1_r), delta::pi(), EPS_STD);
        EXPECT_THROW(delta::acos(2_r), std::domain_error);
        EXPECT_THROW(delta::acos(-2_r), std::domain_error);
        if (!HasFailure()) {
            std::cout << "ACOS BASIC: Returns correct values at 0, ±1 and rejects out-of-range arguments." << std::endl;
        }
    }

    /**
     * @test AcosPrecision
     * @brief Checks cos(acos(x)) = x.
     */
    TEST_F(TranscendentalCorrectnessTest, AcosPrecision) {
        const Rational x = "0.5"_r;
        const Rational eps = EPS_ULTRA;

        Rational acos_x = delta::acos(x, eps);
        Rational cos_acos = delta::cos(acos_x, eps);

        EXPECT_RATIONAL_NEAR(cos_acos, x, eps * 10);
        if (!HasFailure()) {
            std::cout << "ACOS PRECISION: cos(acos(0.5)) recovers 0.5 to 10*eps at 1e-40 precision." << std::endl;
        }
    }

    /**
     * @test AcosVsAsin
     * @brief Checks acos(x) + asin(x) = π/2.
     */
    TEST_F(TranscendentalCorrectnessTest, AcosVsAsin) {
        const Rational x = "0.5"_r;
        const Rational eps = EPS_ULTRA;

        Rational acos_x = delta::acos(x, eps);
        Rational asin_x = delta::asin(x, eps);
        Rational pi_half = delta::pi(eps) / 2;

        EXPECT_RATIONAL_NEAR(acos_x + asin_x, pi_half, eps * 10);
        if (!HasFailure()) {
            std::cout << "ACOS VS ASIN: acos(0.5) + asin(0.5) = pi/2 identity holds to 10*eps at 1e-40 precision." << std::endl;
        }
    }

    /**
     * @test AcosSpecialValues
     * @brief Tests acos(0)=π/2, acos(1)=0, acos(-1)=π, acos(√2/2)=π/4.
     */
    TEST_F(TranscendentalCorrectnessTest, AcosSpecialValues) {
        const Rational eps = EPS_ULTRA;
        Rational pi_val = delta::pi(eps);

        EXPECT_RATIONAL_NEAR(delta::acos(0_r, eps), pi_val / 2, eps);
        EXPECT_RATIONAL_NEAR(delta::acos(1_r, eps), 0_r, eps);
        EXPECT_RATIONAL_NEAR(delta::acos(-1_r, eps), pi_val, eps);

        Rational sqrt2 = delta::sqrt(2_r, eps);
        Rational arg = sqrt2 / 2;
        EXPECT_RATIONAL_NEAR(delta::acos(arg, eps), pi_val / 4, eps);
        if (!HasFailure()) {
            std::cout << "ACOS SPECIAL VALUES: Exact identities at 0, 1, -1, sqrt(2)/2 all hold within eps at 1e-40 precision." << std::endl;
        }
    }

    /**
     * @test AcosMonotonic
     * @brief Verifies that acos is strictly decreasing.
     */
    TEST_F(TranscendentalCorrectnessTest, AcosMonotonic) {
        const Rational eps = EPS_ULTRA;
        std::vector<Rational> args = { -"9/10"_r, -"1/2"_r, 0_r, "1/2"_r, "9/10"_r };

        for (size_t i = 1; i < args.size(); ++i) {
            Rational val1 = delta::acos(args[i - 1], eps);
            Rational val2 = delta::acos(args[i], eps);

            EXPECT_TRUE(val1 > val2)
                << "acos(" << args[i - 1] << ") = " << val1
                << " should be > acos(" << args[i] << ") = " << val2;
        }
        if (!HasFailure()) {
            std::cout << "ACOS MONOTONIC: acos is strictly decreasing on [-0.9, 0.9] with steps of 0.5." << std::endl;
        }
    }

    /**
     * @test AcosDerivative
     * @brief Checks numerical derivative matches -1/√(1-x²).
     */
    TEST_F(TranscendentalCorrectnessTest, AcosDerivative) {
        const Rational x = "0.5"_r;
        const Rational eps = EPS_ULTRA;
        const Rational dx("1/100000000000000000000000000000000000000"); // 1e-38

        Rational acos_x = delta::acos(x, eps);
        Rational acos_x_plus = delta::acos(x + dx, eps);
        Rational derivative_numeric = (acos_x_plus - acos_x) / dx;

        Rational derivative_analytic = -1_r / delta::sqrt(1_r - x * x, eps);

        EXPECT_RATIONAL_NEAR(derivative_numeric, derivative_analytic, eps * 1000);
        if (!HasFailure()) {
            std::cout << "ACOS DERIVATIVE: Numerical derivative matches -1/sqrt(1-x^2) at x=0.5 to 1000*eps tolerance." << std::endl;
        }
    }

    // =============================================================================
    // 12. EXTREME PRECISION: no hanging at 1e-80
    // =============================================================================

    /**
     * @test ExtremePrecisionDoesNotHang
     * @brief Ensures all functions complete at epsilon=1e-80.
     */
    TEST_F(TranscendentalCorrectnessTest, ExtremePrecisionDoesNotHang) {
        const Rational eps = EPS_EXTREME;
        Rational x = 2_r;
        // All calls should finish in reasonable time
        Rational s = delta::sqrt(x, eps);
        Rational e = delta::exp(1_r, eps);
        Rational p = delta::pi(eps);
        Rational l = delta::log(2_r, eps);
        Rational sin_val = delta::sin(1_r, eps);
        EXPECT_TRUE(s > 1_r);
        EXPECT_TRUE(e > 2_r);
        EXPECT_TRUE(p > 3_r);
        EXPECT_TRUE(l > 0_r);
        EXPECT_TRUE(sin_val > 0_r);
        if (!HasFailure()) {
            std::cout << "EXTREME PRECISION: All functions complete at 1e-80 without hanging and return reasonable values." << std::endl;
        }
    }

    // =============================================================================
    // 13. HIGH PRECISION: identities and naive reference
    // =============================================================================

    /**
     * @test SeriesPathHighPrecision
     * @brief Checks sin²+cos²=1, exp(log(x))=x, and matches naive implementations.
     */
    TEST_F(TranscendentalCorrectnessTest, SeriesPathHighPrecision) {
        std::vector<Rational> epsilons = { EPS_ULTRA, EPS_EXTREME };
        Rational x = "1.23456789"_r;
        for (const auto& eps : epsilons) {
            Rational s = delta::sin(x, eps);
            Rational c = delta::cos(x, eps);
            Rational e = delta::exp(x, eps);
            Rational p = delta::pi(eps);

            // sin²+cos²=1
            EXPECT_RATIONAL_NEAR(s * s + c * c, 1_r, eps * 10) << "eps=" << eps;

            // exp and log are inverses
            Rational log_e = delta::log(e, eps);
            EXPECT_RATIONAL_NEAR(log_e, x, eps * 1000) << "eps=" << eps;

            // Compare with naive implementations
            test_function_against_naive(
                [](const Rational& arg, const Rational& e) { return delta::sin(arg, e); },
                naive_series_sin, x, eps, "sin");
            test_function_against_naive(
                [](const Rational& arg, const Rational& e) { return delta::cos(arg, e); },
                naive_series_cos, x, eps, "cos");
            test_function_against_naive(
                [](const Rational& arg, const Rational& e) { return delta::exp(arg, e); },
                naive_series_exp, x, eps, "exp");
            test_function_against_naive(
                [](const Rational& arg, const Rational& e) { return delta::log(arg, e); },
                naive_series_log, x, eps, "log");
            test_function_against_naive(
                [](const Rational&, const Rational& e) { return delta::pi(e); },
                [](const Rational&, const Rational& e) { return naive_series_pi(e); },
                0_r, eps, "pi");
            test_function_against_naive(
                [](const Rational&, const Rational& e) { return delta::e(e); },
                [](const Rational&, const Rational& e) { return naive_series_e(e); },
                0_r, eps, "e");
        }
        if (!HasFailure()) {
            std::cout << "HIGH PRECISION IDENTITIES: All 6 functions satisfy identities and match naive implementations at 1e-40 and 1e-80." << std::endl;
        }
    }

    // =============================================================================
    // 14. POW WITH RATIONAL EXPONENTS: a^(p/q)
    // =============================================================================

    /**
     * @test PowRationalExponent
     * @brief Verifies pow(2,1/2)=√2, pow(16,3/4)=8, and matches naive series.
     */
    TEST_F(TranscendentalCorrectnessTest, PowRationalExponent) {
        const Rational eps = EPS_STD;
        // Square root via pow
        EXPECT_RATIONAL_NEAR(delta::pow(2_r, Rational(1, 2)), delta::sqrt(2_r), eps);
        // 16^(3/4) = 8
        EXPECT_RATIONAL_NEAR(delta::pow(16_r, Rational(3, 4)), 8_r, eps);
        // Compare with naive implementation for a fractional exponent
        Rational base = "3.5"_r;
        Rational exp = Rational(2, 3);
        Rational delta_pow = delta::pow(base, exp, eps);
        Rational naive_pow = naive_series_pow(base, exp, eps);
        EXPECT_RATIONAL_NEAR(delta_pow, naive_pow, eps * 10);
        if (!HasFailure()) {
            std::cout << "POW RATIONAL EXPONENT: pow via a*(p/q) matches both exact roots and naive series implementation." << std::endl;
        }
    }

    // =============================================================================
    // 15. FUNDAMENTAL IDENTITIES: sin²+cos², exp(log), sqrt², cos(acos)
    // =============================================================================

    /**
     * @test FundamentalIdentities
     * @brief Checks four identities for multiple x and two epsilons.
     */
    TEST_F(TranscendentalCorrectnessTest, FundamentalIdentities) {
        std::vector<Rational> values = { 0_r, "0.5"_r, 1_r, "1.5"_r, 2_r };
        std::vector<Rational> epsilons = { EPS_STD, EPS_ULTRA };

        for (const auto& x : values) {
            for (const auto& eps : epsilons) {
                Rational s = delta::sin(x, eps);
                Rational c = delta::cos(x, eps);
                EXPECT_RATIONAL_NEAR(s * s + c * c, 1_r, eps * 10)
                    << "sin^2+cos^2=1 for x=" << x << ", eps=" << eps;

                if (x > 0_r) {
                    Rational log_x = delta::log(x, eps);
                    Rational exp_log = delta::exp(log_x, eps);
                    EXPECT_RATIONAL_NEAR(exp_log, x, eps * 100)
                        << "exp(log(x)) for x=" << x << ", eps=" << eps;
                }

                if (x >= 0_r) {
                    Rational sqrt_x = delta::sqrt(x, eps);
                    EXPECT_RATIONAL_NEAR(sqrt_x * sqrt_x, x, eps * 10)
                        << "sqrt(x)^2 for x=" << x << ", eps=" << eps;
                }

                if (x >= -1_r && x <= 1_r) {
                    Rational acos_x = delta::acos(x, eps);
                    Rational cos_acos = delta::cos(acos_x, eps);
                    EXPECT_RATIONAL_NEAR(cos_acos, x, eps * 10)
                        << "cos(acos(x)) for x=" << x << ", eps=" << eps;
                }
            }
        }
        if (!HasFailure()) {
            std::cout << "FUNDAMENTAL IDENTITIES: All 4 identities hold across 5 arguments at both standard (1e-12) and ultra (1e-40) precision." << std::endl;
        }
    }

    // -----------------------------------------------------------------------------
    // Debug tests (disabled by default)
    // -----------------------------------------------------------------------------
    TEST_F(TranscendentalCorrectnessTest, LazyEpsDebug) {
        using namespace delta;
        LazyRational x = LazyRational("1.23456789"_r);

        // Compute sin directly with different eps
        Rational sin_std = sin("1.23456789"_r, EPS_STD);
        Rational sin_ultra = sin("1.23456789"_r, EPS_ULTRA);

        // Through lazy
        LazyRational lazy_sin = Sin(x);
        Rational lazy_sin_val = lazy_sin.eval();

        std::cout << "Direct sin (std):  " << sin_std.to_double() << std::endl;
        std::cout << "Direct sin (ultra): " << sin_ultra.to_double() << std::endl;
        std::cout << "Lazy sin:          " << lazy_sin_val.to_double() << std::endl;
    }

    TEST_F(TranscendentalCorrectnessTest, LazyExpLogDebug) {
        using namespace delta;

        Rational z = "2.23456789"_r;
        Rational direct = exp(log(z, EPS_STD), EPS_STD);
        std::cout << "Direct exp(log(z)): " << direct.to_double() << std::endl;
        std::cout << "Exact z:            " << z.to_double() << std::endl;

        LazyRational lz = LazyRational(z);
        LazyRational lexpr = Exp(Log(lz));
        Rational lazy_val = lexpr.eval();
        std::cout << "Lazy Exp(Log(z)):   " << lazy_val.to_double() << std::endl;

        EXPECT_RATIONAL_NEAR(direct, z, EPS_STD);

        // Lazy canonicalisation Exp(Log(z)) -> z is correct
        if (lazy_val == z) {
            std::cout << "Lazy CANONICALIZED Exp(Log(z)) -> z (expected)" << std::endl;
        }
        else {
            std::cout << "Lazy returned: " << lazy_val.to_double() << std::endl;
        }
    }

    // =============================================================================
    // 16. LAZY HIGH‑PRECISION EXPRESSIONS: canonicalisation test
    // =============================================================================

    /**
     * @test LazyWithHighPrecision
     * @brief Builds Sin(x) + Cos(2x) + (x+1) using lazy and checks exact result.
     */
    TEST_F(TranscendentalCorrectnessTest, LazyWithHighPrecision) {
        using namespace delta;

        // IMPORTANT: LazyRational is mutable. Operators +, *, Log with an lvalue
        // argument modify it in place for efficiency. Therefore each sub‑expression
        // that uses x in a mutable position must work on a separate copy.
        //
        // Sin and Cos take const&, copy the object and do not mutate it – safe to use common x.
        // However x * 2 and x + 1 (with Rational) mutate the left LazyRational,
        // so we create a temporary copy for each of them.

        // One x for non‑mutating operations (Sin)
        LazyRational x = LazyRational("1.23456789"_r);

        // Expression: Sin(x) + Cos(x * 2) + Exp(Log(x + 1))
        LazyRational expr = Sin(x)                       // x not mutated
            + Cos(x.clone() * 2_r)   // temporary copy for multiplication
            + Exp(Log(x.clone() + 1_r)); // temporary copy for addition

        // eager computes exp(log(z)) numerically
        Rational eager_res = sin("1.23456789"_r) + cos("2.46913578"_r) + exp(log("2.23456789"_r));
        Rational lazy_res_std = expr.eval();

        // With canonicalisation Exp(Log(z)) → z (exactly), tolerance is the error of exp(log(z))
        // exp(log(z)) ≈ z to EPS_STD, so:
        EXPECT_RATIONAL_NEAR(lazy_res_std, eager_res, EPS_STD * 3)
            << "Lazy (with Exp-Log cancellation) vs eager";

        // Ultra‑precision: create a new expression (old expr may have cached result)
        LazyRational x2 = LazyRational("1.23456789"_r);
        LazyRational expr2 = Sin(x2)
            + Cos(x2.clone() * 2_r)
            + Exp(Log(x2.clone() + 1_r));

        Rational lazy_res_ultra = expr2.eval(false);

        // Exact equivalent after canonicalisation: sin(x) + cos(2x) + (x+1)
        Rational exact_res = sin("1.23456789"_r) + cos("2.46913578"_r) + ("2.23456789"_r);

        EXPECT_RATIONAL_NEAR(lazy_res_ultra, exact_res, EPS_ULTRA * 10)
            << "Lazy (ultra) vs exact sin(x)+cos(2x)+(x+1)";

        if (!HasFailure()) {
            std::cout << "LAZY HIGH PRECISION: Lazy expression with Exp(Log(x+1)) correctly "
                << "canonicalizes to sin(x)+cos(2x)+(x+1) and matches exact computation "
                << "at 1e-40 precision." << std::endl;
        }
    }
} // namespace delta::testing