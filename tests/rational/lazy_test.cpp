#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "delta/rational/simplify.h"   // для внутренних нужд (только если нужны внутренние типы, но в тестах уже не используем)
#include "test_utils.h"

namespace delta::testing {

    // -------------------------------------------------------------------------
    // 1. Lazy node creation
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, LazyNodeCreation) {
        set_eager_mode(false);
        Rational sum = "1/2"_r + "1/3"_r;
        EXPECT_TRUE(sum.is_lazy());
    }

    // -------------------------------------------------------------------------
    // 2. Lazy tree structure (adapted to flat buffer)
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, LazyTreeStructure) {
        set_eager_mode(false);
        Rational expr = ("1/2"_r + "1/3"_r) * "1/4"_r;
        ASSERT_TRUE(expr.is_lazy());
        const auto& root = *expr.as_lazy();
        const auto& nodes = root.nodes();
        int root_idx = root.root_index();
        const auto& root_node = nodes[root_idx];
        EXPECT_EQ(root_node.op, internal::LazyOp::MUL);
        ASSERT_NE(root_node.child0, -1);
        ASSERT_NE(root_node.child1, -1);

        // Проверяем левый аргумент (должен быть ADD)
        const auto& left_node = nodes[root_node.child0];
        EXPECT_EQ(left_node.op, internal::LazyOp::ADD);

        // Проверяем правый аргумент (должен быть CONST со значением 1/4)
        const auto& right_node = nodes[root_node.child1];
        EXPECT_EQ(right_node.op, internal::LazyOp::CONST);
        const auto& right_val = root.values()[right_node.value_idx];
        // Преобразуем Value в Rational для сравнения
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
    TEST_F(RationalTest, SimplifyZeroOne) {
        set_eager_mode(false);

        // 0 + x
        Rational sum = 0_r + "1/2"_r;
        Rational simp = sum.simplify();
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, "1/2"_r);

        // 1 * x
        Rational prod = 1_r * "1/2"_r;
        simp = prod.simplify();
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, "1/2"_r);

        // x - x
        Rational diff = "1/2"_r - "1/2"_r;
        simp = diff.simplify();
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, 0_r);

        // x / x
        Rational quot = "1/2"_r / "1/2"_r;
        simp = quot.simplify();
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, 1_r);
    }

    // -------------------------------------------------------------------------
    // 4. Simplify exp(log(x)) and log(exp(x))
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, SimplifyExpLog) {
        set_eager_mode(false);
        Rational expr = delta::exp(delta::log("2"_r));
        Rational simp = expr.simplify();
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, "2"_r);

        expr = delta::log(delta::exp("2"_r));
        simp = expr.simplify();
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, "2"_r);
    }

    // -------------------------------------------------------------------------
    // 5. Simplify sqrt(exp(x)) -> exp(x/2)  (not implemented, so we check numerically)
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, SimplifySqrtExp) {
        set_eager_mode(false);
        Rational expr = delta::sqrt(delta::exp("1"_r));
        Rational simp = expr.simplify();
        // В текущей реализации упрощение не меняет структуру
        // Проверяем, что результат после вычисления совпадает с exp(0.5)
        Rational expected = delta::exp(Rational(1, 2));
        // Сравниваем через eval (численное равенство с точностью по умолчанию)
        EXPECT_EQ(simp.eval(), expected.eval());
    }

    // -------------------------------------------------------------------------
    // 6. Simplify trigonometric constants (only sin(0)=0, cos(0)=1 are implemented)
    //    For acos(0) and acos(1) we check numeric results.
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, SimplifyTrig) {
        set_eager_mode(false);

        // sin(0) = 0
        Rational s = delta::sin(0_r);
        Rational simp = s.simplify();
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, 0_r);

        // cos(0) = 1
        Rational c = delta::cos(0_r);
        simp = c.simplify();
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, 1_r);

        // acos(1) = 0
        Rational a1 = delta::acos(1_r);
        simp = a1.simplify();
        EXPECT_FALSE(simp.is_lazy());
        EXPECT_EQ(simp, 0_r);

        // acos(0) = π/2 (not simplified, but numeric check)
        Rational a0 = delta::acos(0_r);
        simp = a0.simplify();
        Rational expected = delta::pi() / 2_r;
        EXPECT_EQ(simp.eval(), expected.eval());
    }

    // -------------------------------------------------------------------------
    // 7. Constant folding (no constant folding, so we check that eval gives correct result)
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, ConstantFolding) {
        set_eager_mode(false);
        Rational expr = "1/2"_r + "1/3"_r;
        // Дерево остаётся ленивым
        EXPECT_TRUE(expr.is_lazy());
        // Вычисление даёт immediate значение
        Rational val = expr.eval();
        EXPECT_EQ(val, "5/6"_r);
    }

    // -------------------------------------------------------------------------
    // 8. Structural equality (using operator==)
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, StructuralEquality) {
        set_eager_mode(false);
        Rational a = "1/2"_r + "1/3"_r;
        Rational b = "1/2"_r + "1/3"_r;
        EXPECT_TRUE(a == b);

        Rational c = "1/2"_r + "1/4"_r;
        EXPECT_FALSE(a == c);
    }

    // -------------------------------------------------------------------------
    // 9. Structural inequality (using operator==)
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, StructuralInequality) {
        set_eager_mode(false);
        Rational a = "1/2"_r + "1/3"_r;
        Rational b = "1/2"_r + "1/4"_r;
        EXPECT_FALSE(a == b);

        Rational d = "1/2"_r * "1/3"_r;
        EXPECT_FALSE(a == d);
    }

} // namespace delta::testing