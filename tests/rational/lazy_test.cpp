// tests/rational/lazy_test.cpp
#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "delta/rational/simplify.h"   // for internal::simplify, internal::structurally_equal
#include "test_utils.h"

namespace delta::testing {

    // -------------------------------------------------------------------------
    // 1. Lazy node creation
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, LazyNodeCreation) {
        // Ensure we are not in eager mode
        set_eager_mode(false);
        Rational sum = "1/2"_r + "1/3"_r;
        EXPECT_TRUE(sum.is_lazy());
    }

    // -------------------------------------------------------------------------
    // 2. Lazy tree structure
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, LazyTreeStructure) {
        set_eager_mode(false);
        Rational expr = ("1/2"_r + "1/3"_r) * "1/4"_r;
        ASSERT_TRUE(expr.is_lazy());
        auto node = expr.as_lazy();
        EXPECT_EQ(node->op, internal::LazyOp::MUL);
        ASSERT_EQ(node->args.size(), 2);
        // First argument should be an ADD node
        auto add_node = node->args[0]->as_lazy();
        ASSERT_TRUE(add_node != nullptr);
        EXPECT_EQ(add_node->op, internal::LazyOp::ADD);
        // Second argument should be "1/4"
        Rational arg2 = *node->args[1];
        EXPECT_EQ(arg2.evaluate(), "1/4"_r);
    }

    // -------------------------------------------------------------------------
    // 3. Simplify rules: 0+x, 1*x, x-x, x/x
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, SimplifyZeroOne) {
        set_eager_mode(false);

        // 0 + x
        Rational sum = 0_r + "1/2"_r;
        Rational simp = internal::simplify(sum);
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, "1/2"_r);

        // 1 * x
        Rational prod = 1_r * "1/2"_r;
        simp = internal::simplify(prod);
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, "1/2"_r);

        // x - x
        Rational diff = "1/2"_r - "1/2"_r;
        simp = internal::simplify(diff);
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, 0_r);

        // x / x
        Rational quot = "1/2"_r / "1/2"_r;
        simp = internal::simplify(quot);
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, 1_r);
    }

    // -------------------------------------------------------------------------
    // 4. Simplify exp(log(x)) and log(exp(x))
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, SimplifyExpLog) {
        set_eager_mode(false);
        Rational expr = delta::exp(delta::log("2"_r));
        Rational simp = internal::simplify(expr);
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, "2"_r);

        expr = delta::log(delta::exp("2"_r));
        simp = internal::simplify(expr);
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, "2"_r);
    }

    // -------------------------------------------------------------------------
    // 5. Simplify sqrt(exp(x)) -> exp(x/2)
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, SimplifySqrtExp) {
        set_eager_mode(false);
        Rational expr = delta::sqrt(delta::exp("1"_r));
        Rational simp = internal::simplify(expr);
        ASSERT_TRUE(simp.is_lazy());
        auto node = simp.as_lazy();
        EXPECT_EQ(node->op, internal::LazyOp::EXP);
        ASSERT_EQ(node->args.size(), 1);
        Rational arg = *node->args[0];
        EXPECT_EQ(arg.evaluate(), "1/2"_r);
    }

    // -------------------------------------------------------------------------
    // 6. Simplify trigonometric constants
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, SimplifyTrig) {
        set_eager_mode(false);

        // sin(0) = 0
        Rational s = delta::sin(0_r);
        Rational simp = internal::simplify(s);
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, 0_r);

        // cos(0) = 1
        Rational c = delta::cos(0_r);
        simp = internal::simplify(c);
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, 1_r);

        // acos(1) = 0
        Rational a1 = delta::acos(1_r);
        simp = internal::simplify(a1);
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, 0_r);

        // acos(0) = π/2 (should be a lazy node with PI and division)
        Rational a0 = delta::acos(0_r);
        simp = internal::simplify(a0);
        ASSERT_TRUE(simp.is_lazy());
        auto node = simp.as_lazy();
        EXPECT_EQ(node->op, internal::LazyOp::DIV);
        ASSERT_EQ(node->args.size(), 2);
        // First argument should be π
        auto pi_node = node->args[0]->as_lazy();
        ASSERT_TRUE(pi_node != nullptr);
        EXPECT_EQ(pi_node->op, internal::LazyOp::PI);
        // Second argument should be 2
        Rational denom = *node->args[1];
        EXPECT_EQ(denom.evaluate(), 2_r);
    }

    // -------------------------------------------------------------------------
    // 7. Constant folding
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, ConstantFolding) {
        set_eager_mode(false);
        Rational expr = "1/2"_r + "1/3"_r;
        // After construction, simplification is already applied.
        // But we can call simplify explicitly to ensure it's a constant.
        Rational simp = internal::simplify(expr);
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, "5/6"_r);
    }

    // -------------------------------------------------------------------------
    // 8. Structural equality
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, StructuralEquality) {
        set_eager_mode(false);
        Rational a = "1/2"_r + "1/3"_r;
        Rational b = "1/2"_r + "1/3"_r;
        EXPECT_TRUE(internal::structurally_equal(a, b));

        Rational c = "1/2"_r + "1/4"_r;
        EXPECT_FALSE(internal::structurally_equal(a, c));
    }

    // -------------------------------------------------------------------------
    // 9. Structural inequality
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, StructuralInequality) {
        set_eager_mode(false);
        Rational a = "1/2"_r + "1/3"_r;
        Rational b = "1/2"_r + "1/4"_r;
        EXPECT_FALSE(internal::structurally_equal(a, b));

        Rational d = "1/2"_r * "1/3"_r;
        EXPECT_FALSE(internal::structurally_equal(a, d));
    }

} // namespace delta::testing