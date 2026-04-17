// tests/rational/lazy_test.cpp
// Дополнительные тесты для LazyRational (мутации, COW, сложные сценарии)
#include "lazy_rational_test_fixture.h"
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::test {

    class LazyRationalExtraTest : public LazyRationalTestFixture {};

    TEST_F(LazyRationalExtraTest, SumCowOnMultipleReferences) {
        LazyRational x = LazyRational(Rational(1, 2));
        LazyRational y = LazyRational(Rational(1, 3));
        LazyRational sum = x.clone();
        sum += y;                     // sum = 1/2 + 1/3 = 5/6
        LazyRational copy = sum.clone();
        sum += Rational(1);           // sum = 5/6 + 1 = 11/6
        EXPECT_EQ(copy.eval(), Rational(5, 6));
        EXPECT_EQ(sum.eval(), Rational(11, 6));
    }

    TEST_F(LazyRationalExtraTest, PlusEqualOnImmediate) {
        LazyRational a = LazyRational(1_r); // грязный CONST(1)
        LazyRational b = LazyRational(Rational(1, 2));
        a += b;
        EXPECT_TRUE(is_dirty(a));
        EXPECT_EQ(dirty_root_op(a), internal::LazyOp::SUM);
        EXPECT_EQ(dirty_root_children(a).size(), 2);
        EXPECT_EQ(a.eval(), Rational(3, 2));
    }

    TEST_F(LazyRationalExtraTest, SumOfTwoSums) {
        LazyRational a = LazyRational(Rational(1, 2));
        LazyRational b = LazyRational(Rational(1, 3));
        LazyRational c = LazyRational(Rational(1, 6));
        LazyRational d = LazyRational(Rational(1, 4));

        LazyRational sum1 = a.clone();
        sum1 += b;
        LazyRational sum2 = c.clone();
        sum2 += d;
        LazyRational total = sum1.clone();
        total += sum2;   // теперь total = sum1 + sum2 (плоский SUM)
        EXPECT_EQ(dirty_root_op(total), internal::LazyOp::SUM);
        auto children = dirty_root_children(total);
        // Должно быть 4 ребёнка (без вложенных SUM)
        EXPECT_EQ(children.size(), 4);
    }

    TEST_F(LazyRationalExtraTest, ChainedMultiplication) {
        LazyRational a = LazyRational(Rational(2));
        a* Rational(3)* Rational(4);   // мутируем a, без присваивания
        EXPECT_EQ(dirty_root_op(a), internal::LazyOp::PRODUCT);
        EXPECT_EQ(dirty_root_children(a).size(), 3);
        EXPECT_EQ(a.eval(), Rational(24));
    }

    TEST_F(LazyRationalExtraTest, MixedOperations) {
        LazyRational a = LazyRational(Rational(2));
        a * 3_r + 5_r;   // мутируем a: сначала умножение, затем сложение
        EXPECT_EQ(dirty_root_op(a), internal::LazyOp::SUM);
        auto children = dirty_root_children(a);
        EXPECT_EQ(children.size(), 2);
        // Первый ребёнок - PRODUCT
        EXPECT_EQ(dirty_node_op(a, children[0]), internal::LazyOp::PRODUCT);
        EXPECT_EQ(a.eval(), Rational(11));
    }

} // namespace delta::test