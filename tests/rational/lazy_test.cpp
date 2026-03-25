#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {

    class RationalLazyTest : public RationalTest {};

    // -------------------------------------------------------------------------
    // 1. Lazy node creation
    // -------------------------------------------------------------------------
    TEST_F(RationalLazyTest, LazyNodeCreation) {
        set_eager_mode(false);
        Rational sum = "1/2"_r.lazy() + "1/3"_r;
        EXPECT_TRUE(sum.is_lazy());
    }

    // -------------------------------------------------------------------------
    // 2. Lazy tree structure
    // -------------------------------------------------------------------------
    TEST_F(RationalLazyTest, LazyTreeStructure) {
        set_eager_mode(false);
        Rational expr = ("1/2"_r.lazy() + "1/3"_r) * "1/4"_r;
        ASSERT_TRUE(expr.is_lazy());
        const auto& root = *expr.as_lazy();
        const auto& nodes = root.nodes();
        int root_idx = root.root_index();
        const auto& root_node = nodes[root_idx];
        EXPECT_EQ(root_node.op, internal::LazyOp::MUL);
        ASSERT_NE(root_node.child0, -1);
        ASSERT_NE(root_node.child1, -1);

        const auto& left_node = nodes[root_node.child0];
        EXPECT_EQ(left_node.op, internal::LazyOp::ADD);

        const auto& right_node = nodes[root_node.child1];
        EXPECT_EQ(right_node.op, internal::LazyOp::CONST);
        const auto& right_val = root.values()[right_node.value_idx];
        Rational right_rat;
        if (std::holds_alternative<internal::SmallStorage>(right_val)) {
            right_rat = Rational(std::get<internal::SmallStorage>(right_val));
        }
        else {
            right_rat = Rational(std::get<internal::BigStorage>(right_val));
        }
        EXPECT_EQ(right_rat, "1/4"_r);
    }

    // -------------------------------------------------------------------------
    // 3. Simplify rules: 0+x, 1*x, x-x, x/x
    // -------------------------------------------------------------------------
    TEST_F(RationalLazyTest, SimplifyZeroOne) {
        set_eager_mode(false);

        Rational sum = 0_r + "1/2"_r;
        Rational simp = sum.simplify();
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, "1/2"_r);

        Rational prod = 1_r * "1/2"_r;
        simp = prod.simplify();
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, "1/2"_r);

        Rational diff = "1/2"_r - "1/2"_r;
        simp = diff.simplify();
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, 0_r);

        Rational quot = "1/2"_r / "1/2"_r;
        simp = quot.simplify();
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, 1_r);
    }

    // -------------------------------------------------------------------------
    // 4. Simplify exp(log(x)) and log(exp(x))
    // -------------------------------------------------------------------------
    TEST_F(RationalLazyTest, SimplifyExpLog) {
        set_eager_mode(false);
        Rational expr = delta::exp(delta::log("2"_r));
        Rational simp = expr.simplify();
        EXPECT_EQ(simp.eval(), 2_r);

        expr = delta::log(delta::exp("2"_r));
        simp = expr.simplify();
        EXPECT_EQ(simp.eval(), 2_r);
    }

    // -------------------------------------------------------------------------
    // 5. Simplify sqrt(exp(x))
    // -------------------------------------------------------------------------
    TEST_F(RationalLazyTest, SimplifySqrtExp) {
        set_eager_mode(false);
        Rational expr = delta::sqrt(delta::exp("1"_r));
        Rational simp = expr.simplify();
        Rational expected = delta::exp(Rational(1, 2));
        EXPECT_EQ(simp.eval(), expected.eval());
    }

    // -------------------------------------------------------------------------
    // 6. Simplify trigonometric constants
    // -------------------------------------------------------------------------
    TEST_F(RationalLazyTest, SimplifyTrig) {
        set_eager_mode(false);

        Rational s = delta::sin(0_r);
        Rational simp = s.simplify();
        EXPECT_EQ(simp.eval(), 0_r);

        Rational c = delta::cos(0_r);
        simp = c.simplify();
        EXPECT_EQ(simp.eval(), 1_r);

        Rational a1 = delta::acos(1_r);
        simp = a1.simplify();
        EXPECT_EQ(simp.eval(), 0_r);

        Rational a0 = delta::acos(0_r);
        simp = a0.simplify();
        Rational expected = delta::pi() / 2_r;
        EXPECT_EQ(simp.eval(), expected.eval());
    }

    // -------------------------------------------------------------------------
    // 7. Constant folding
    // -------------------------------------------------------------------------
    TEST_F(RationalLazyTest, ConstantFolding) {
        set_eager_mode(false);
        Rational expr = "1/2"_r.lazy() + "1/3"_r;
        EXPECT_TRUE(expr.is_lazy());
        Rational val = expr.eval();
        EXPECT_EQ(val, "5/6"_r);
    }

    // -------------------------------------------------------------------------
    // 8. Structural equality
    // -------------------------------------------------------------------------
    TEST_F(RationalLazyTest, StructuralEquality) {
        set_eager_mode(false);
        Rational a = "1/2"_r.lazy() + "1/3"_r;
        Rational b = "1/2"_r.lazy() + "1/3"_r;
        EXPECT_TRUE(a == b);

        Rational c = "1/2"_r.lazy() + "1/4"_r;
        EXPECT_FALSE(a == c);
    }

    // -------------------------------------------------------------------------
    // 9. Structural inequality
    // -------------------------------------------------------------------------
    TEST_F(RationalLazyTest, StructuralInequality) {
        set_eager_mode(false);
        Rational a = "1/2"_r.lazy() + "1/3"_r;
        Rational b = "1/2"_r.lazy() + "1/4"_r;
        EXPECT_FALSE(a == b);

        Rational d = "1/2"_r.lazy() * "1/3"_r;
        EXPECT_FALSE(a == d);
    }

} // namespace delta::testing