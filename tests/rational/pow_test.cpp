// tests/rational/pow_test.cpp
#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {

    class RationalPowTest : public RationalTest {};

    // -------------------------------------------------------------------------
    // 1. Eager pow with integer exponent (immediate)
    // -------------------------------------------------------------------------
    TEST_F(RationalPowTest, EagerPowIntegerExponent) {
        set_eager_mode(true);

        // Positive exponent
        EXPECT_EQ(pow(2_r, 3), 8_r);
        EXPECT_EQ(pow("2/3"_r, 2), "4/9"_r);
        EXPECT_EQ(pow(0_r, 5), 0_r);

        // Exponent 0
        EXPECT_EQ(pow(0_r, 0), 1_r);
        EXPECT_EQ(pow(2_r, 0), 1_r);

        // Negative exponent
        EXPECT_EQ(pow(2_r, -2), "1/4"_r);
        EXPECT_EQ(pow("2/3"_r, -1), "3/2"_r);
        EXPECT_THROW(pow(0_r, -1), std::domain_error);  // domain error

        // 1^anything = 1
        EXPECT_EQ(pow(1_r, 100), 1_r);
        EXPECT_EQ(pow(1_r, -100), 1_r);
    }

    // -------------------------------------------------------------------------
    // 2. Eager pow with rational exponent (non‑integer)
    // -------------------------------------------------------------------------
    TEST_F(RationalPowTest, EagerPowRationalExponent) {
        set_eager_mode(true);
        Rational eps = default_eps();  // 1e-30

        // sqrt(4) = 4^(1/2) = 2 exactly
        Rational p = pow(4_r, "1/2"_r, eps);
        EXPECT_EQ(p.eval(), 2_r);

        // 8^(1/3) = 2
        p = pow(8_r, "1/3"_r, eps);
        EXPECT_EQ(p.eval(), 2_r);

        // 2^(1/2) ≈ 1.4142...
        Rational p2 = pow(2_r, "1/2"_r, eps);
        Rational expected_sqrt2 = Rational("14142135623730950488/10000000000000000000");
        EXPECT_RATIONAL_NEAR(p2.eval(), expected_sqrt2, "1/1000000000000"_r);

        // (1/4)^(1/2) = 1/2
        p = pow("1/4"_r, "1/2"_r, eps);
        EXPECT_EQ(p.eval(), "1/2"_r);

        // 0^positive = 0
        p = pow(0_r, "1/2"_r, eps);
        EXPECT_EQ(p.eval(), 0_r);

        // 0^0 throws
        EXPECT_THROW(pow(0_r, 0_r, eps), std::domain_error);

        // 0^negative throws
        EXPECT_THROW(pow(0_r, -1_r, eps), std::domain_error);
    }

    TEST_F(RationalPowTest, EagerPowHighPrecision) {
        set_eager_mode(true);
        Rational eps = default_eps(); // 1e-30

        // Вычисляем точное значение 5^(1/3) через высокоточный float
        using boost::multiprecision::cpp_dec_float_100;
        cpp_dec_float_100 base = 5;
        cpp_dec_float_100 exponent = cpp_dec_float_100(1) / 3;
        cpp_dec_float_100 ref = pow(base, exponent);  // cpp_dec_float_100::pow

        // Преобразуем reference в Rational с достаточной точностью
        // Берём 100 десятичных цифр (запас гораздо больше eps=1e-30)
        std::string ref_str = ref.str(100, std::ios_base::fixed);
        Rational expected(ref_str);

        // Вычисляем результат нашей библиотеки
        Rational p = pow(5_r, "1/3"_r, eps);
        Rational val = p.eval();

        // Допуск: eps * 100 (как было в оригинале)
        Rational eps10 = eps * 10;

        EXPECT_LE(delta::abs(val - expected), eps10);
    }

    // -------------------------------------------------------------------------
    // 4. Lazy pow – tree building
    // -------------------------------------------------------------------------
    TEST_F(RationalPowTest, LazyPowTree) {
        set_eager_mode(false);

        Rational base = 2_r.lazy();
        Rational exp = "1/2"_r;
        Rational expr = pow(base, exp);   // exponent immediate, base lazy
        EXPECT_TRUE(expr.is_lazy());

        int root_idx = expr.root_index();
        const auto& node = internal::pool.nodes[root_idx];
        EXPECT_EQ(node.op, internal::LazyOp::POW);
        EXPECT_NE(node.child0, -1);
        EXPECT_NE(node.child1, -1);

        // Both lazy
        Rational exp_lazy = "1/2"_r.lazy();
        Rational expr2 = pow(base, exp_lazy);
        EXPECT_TRUE(expr2.is_lazy());
    }

    // -------------------------------------------------------------------------
    // 5. Lazy pow – simplification rules
    // -------------------------------------------------------------------------
    TEST_F(RationalPowTest, LazyPowSimplify) {
        set_eager_mode(false);

        // x^1 → x
        Rational p1 = pow(2_r.lazy(), 1_r);
        Rational simp = p1.simplify();
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, 2_r);

        // x^0 → 1
        Rational p0 = pow(2_r.lazy(), 0_r);
        simp = p0.simplify();
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, 1_r);

        // 1^y → 1
        Rational one_pow = pow(1_r.lazy(), "1/2"_r);
        simp = one_pow.simplify();
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, 1_r);

        // 0^positive → 0 (but exponent not integer)
        Rational zero_pow = pow(0_r.lazy(), "1/2"_r);
        simp = zero_pow.simplify();
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, 0_r);
    }

    // -------------------------------------------------------------------------
    // 6. Lazy pow – evaluation after building
    // -------------------------------------------------------------------------
    TEST_F(RationalPowTest, LazyPowEvaluation) {
        set_eager_mode(false);

        // 2^(1/2) lazy
        Rational p = pow(2_r.lazy(), "1/2"_r);
        Rational val = p.eval();
        Rational expected_sqrt2 = Rational("14142135623730950488/10000000000000000000");
        EXPECT_RATIONAL_NEAR(val, expected_sqrt2, "1/1000000000000"_r);

        // (1/4)^(1/2) = 1/2
        p = pow("1/4"_r.lazy(), "1/2"_r);
        val = p.eval();
        EXPECT_EQ(val, "1/2"_r);
    }

    // -------------------------------------------------------------------------
    // 7. Mixed immediate/lazy pow
    // -------------------------------------------------------------------------
    TEST_F(RationalPowTest, MixedPow) {
        set_eager_mode(false);

        // Base immediate, exponent lazy
        Rational p = pow(2_r, "1/2"_r.lazy());
        EXPECT_TRUE(p.is_lazy());
        Rational val = p.eval();
        Rational expected = delta::sqrt(2_r);
        EXPECT_EQ(val, expected);   // should be exactly same because both compute same

        // Base lazy, exponent immediate
        p = pow(2_r.lazy(), "1/2"_r);
        EXPECT_TRUE(p.is_lazy());
        val = p.eval();
        EXPECT_EQ(val, expected);

        // Both immediate in lazy mode → should still be immediate (optimization)
        p = pow(2_r, "1/2"_r);
        EXPECT_FALSE(p.is_lazy());   // both immediate -> eager
        EXPECT_EQ(p.eval(), expected);
    }

    // -------------------------------------------------------------------------
    // 8. Structural equality and hash consistency for pow nodes
    // -------------------------------------------------------------------------
    TEST_F(RationalPowTest, StructuralEqualityPow) {
        set_eager_mode(false);

        Rational a = pow(2_r.lazy(), "1/2"_r);
        Rational b = pow(2_r.lazy(), "1/2"_r);
        EXPECT_EQ(a, b);  // structural equality

        Rational c = pow(2_r.lazy(), "1/3"_r);
        EXPECT_NE(a, c);
    }

    // -------------------------------------------------------------------------
    // 9. Pow with default epsilon
    // -------------------------------------------------------------------------
    TEST_F(RationalPowTest, DefaultEps) {
        set_eager_mode(false);

        Rational base = 2_r.lazy();
        Rational exp = "1/2"_r;

        // Should use default_eps() when not specified
        Rational p1 = pow(base, exp);
        Rational p2 = pow(base, exp, default_eps());
        // They should be structurally equal (same node)
        EXPECT_EQ(p1, p2);
    }

    // -------------------------------------------------------------------------
    // 10. ScopedEagerEval with pow
    // -------------------------------------------------------------------------
    TEST_F(RationalPowTest, ScopedEagerEvalPow) {
        set_eager_mode(false);
        Rational lazy_base = 2_r.lazy();

        {
            ScopedEagerEval guard;
            Rational p = pow(lazy_base, "1/2"_r);
            EXPECT_FALSE(p.is_lazy());
            EXPECT_EQ(p.eval(), delta::sqrt(2_r));
        }

        // After block, lazy mode restored
        Rational p = pow(lazy_base, "1/2"_r);
        EXPECT_TRUE(p.is_lazy());
    }

    // -------------------------------------------------------------------------
    // 11. Domain errors for pow
    // -------------------------------------------------------------------------
    TEST_F(RationalPowTest, DomainErrors) {
        set_eager_mode(false);

        // 0^0
        EXPECT_THROW(pow(0_r, 0_r), std::domain_error);
        // 0^negative (immediate)
        EXPECT_THROW(pow(0_r, -1_r), std::domain_error);
        // 0^negative with lazy exponent — должно выбросить при eval
        EXPECT_THROW(pow(0_r.lazy(), -1_r.lazy()).eval(), std::domain_error);
        // negative base with non‑integer exponent — должно выбросить при eval
        EXPECT_THROW(pow(-2_r, "1/2"_r).eval(), std::domain_error);
    }

} // namespace delta::testing