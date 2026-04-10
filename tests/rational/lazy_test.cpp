// tests/rational/lazy_test.cpp
#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {

    class RationalLazyTest : public RationalTest {};

    TEST_F(RationalLazyTest, LazyNodeCreation) {
        set_eager_mode(false);
        Rational sum = "1/2"_r.lazy() + "1/3"_r;
        EXPECT_TRUE(sum.is_lazy());
    }

    TEST_F(RationalLazyTest, LazyTreeStructure) {
        set_eager_mode(false);
        Rational expr = ("1/2"_r.lazy() + "1/3"_r) * "1/4"_r;
        ASSERT_TRUE(expr.is_lazy());
        int root_idx = expr.root_index();
        const auto& root_node = internal::pool.nodes[root_idx];
        EXPECT_EQ(root_node.op, internal::LazyOp::MUL);
        ASSERT_NE(root_node.child0, -1);
        ASSERT_NE(root_node.child1, -1);

        const auto& left_node = internal::pool.nodes[root_node.child0];
        const auto& right_node = internal::pool.nodes[root_node.child1];

        bool left_is_add = (left_node.op == internal::LazyOp::ADD);
        bool right_is_add = (right_node.op == internal::LazyOp::ADD);
        EXPECT_TRUE((left_is_add && !right_is_add) || (!left_is_add && right_is_add));

        const internal::Node* const_node = (left_node.op == internal::LazyOp::CONST) ? &left_node : &right_node;
        ASSERT_EQ(const_node->op, internal::LazyOp::CONST);
        const auto& const_val = internal::pool.values[const_node->value_idx];
        Rational const_rat;
        if (const_val.tag == internal::ValueType::Small) {
            const_rat = Rational(const_val.storage.small);
        }
        else if (const_val.tag == internal::ValueType::Big) {
            const_rat = Rational(const_val.storage.big);
        }
        else {
            FAIL() << "Expected CONST value to be Small or Big";
        }
        EXPECT_EQ(const_rat, "1/4"_r);
    }

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

    TEST_F(RationalLazyTest, SimplifyExpLog) {
        set_eager_mode(false);
        Rational expr = delta::exp(delta::log("2"_r));
        Rational simp = expr.simplify();
        EXPECT_EQ(simp.eval(), 2_r);

        expr = delta::log(delta::exp("2"_r));
        simp = expr.simplify();
        EXPECT_EQ(simp.eval(), 2_r);
    }

    TEST_F(RationalLazyTest, SimplifySqrtExp) {
        set_eager_mode(false);
        Rational expr = delta::sqrt(delta::exp("1"_r));
        Rational simp = expr.simplify();
        Rational expected = delta::exp(Rational(1, 2));
        EXPECT_EQ(simp.eval(), expected.eval());
    }

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

    TEST_F(RationalLazyTest, ConstantFolding) {
        set_eager_mode(false);
        Rational expr = "1/2"_r.lazy() + "1/3"_r;
        EXPECT_TRUE(expr.is_lazy());
        Rational val = expr.eval();
        EXPECT_EQ(val, "5/6"_r);
    }

    TEST_F(RationalLazyTest, StructuralEquality) {
        set_eager_mode(false);
        Rational a = "1/2"_r.lazy() + "1/3"_r;
        Rational b = "1/2"_r.lazy() + "1/3"_r;
        EXPECT_TRUE(a == b);

        Rational c = "1/2"_r.lazy() + "1/4"_r;
        EXPECT_FALSE(a == c);
    }

    TEST_F(RationalLazyTest, StructuralInequality) {
        set_eager_mode(false);
        Rational a = "1/2"_r.lazy() + "1/3"_r;
        Rational b = "1/2"_r.lazy() + "1/4"_r;
        EXPECT_FALSE(a == b);

        Rational d = "1/2"_r.lazy() * "1/3"_r;
        EXPECT_FALSE(a == d);
    }

    TEST_F(RationalLazyTest, LargeAdditionChain) {
        set_eager_mode(false);
        const int N = 1000;
        Rational sum = 0_r.lazy();
        for (int i = 0; i < N; ++i) {
            sum = sum + 1_r;
        }
        EXPECT_TRUE(sum.is_lazy());
        Rational result = sum.eval();
        EXPECT_EQ(result, Rational(N));
    }

    TEST_F(RationalLazyTest, ChainStaysLazy) {
        set_eager_mode(false);
        const int N = 500;
        Rational sum = 0_r.lazy();
        for (int i = 0; i < N; ++i) {
            sum = sum + 1_r;
        }
        EXPECT_TRUE(sum.is_lazy());
        int root_idx = sum.root_index();
        const auto& root_node = internal::pool.nodes[root_idx];
        EXPECT_EQ(root_node.op, internal::LazyOp::ADD);
    }
} // namespace delta::testing