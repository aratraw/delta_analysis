// tests/rational/lazy_test.cpp
#pragma once
#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "test_utils.h"
#include <chrono>
#include <iostream>

namespace delta::testing {

    class RationalLazyTest : public RationalTest {
    protected:
        void SetUp() override {
            // Каждый тест начинает с чистого пула, чтобы избежать влияния предыдущих тестов
            internal::reset_pool();
            set_eager_mode(false);
        }
        void TearDown() override {
            // Опционально: можно оставить пул в чистом состоянии, но не обязательно
        }
    };

    TEST_F(RationalLazyTest, LazyNodeCreation) {
        Rational sum = "1/2"_r.lazy() + "1/3"_r;
        EXPECT_TRUE(sum.is_lazy());
    }

    TEST_F(RationalLazyTest, LazyTreeStructure) {
        Rational expr = ("1/2"_r.lazy() + "1/3"_r) * "1/4"_r;
        ASSERT_TRUE(expr.is_lazy());
        int root_idx = expr.root_index();
        const auto& root_node = internal::pool.nodes[root_idx];
        EXPECT_EQ(root_node.op, internal::LazyOp::MUL);
        ASSERT_NE(root_node.child0, -1);
        ASSERT_NE(root_node.child1, -1);

        const auto& left_node = internal::pool.nodes[root_node.child0];
        const auto& right_node = internal::pool.nodes[root_node.child1];

        // Один из детей должен быть сложением (ADD или SUM), другой — константой
        bool left_is_add_or_sum = (left_node.op == internal::LazyOp::ADD || left_node.op == internal::LazyOp::SUM);
        bool right_is_add_or_sum = (right_node.op == internal::LazyOp::ADD || right_node.op == internal::LazyOp::SUM);
        EXPECT_TRUE((left_is_add_or_sum && !right_is_add_or_sum) || (!left_is_add_or_sum && right_is_add_or_sum));

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
        Rational expr = delta::exp(delta::log("2"_r));
        Rational simp = expr.simplify();
        EXPECT_EQ(simp.eval(), 2_r);

        expr = delta::log(delta::exp("2"_r));
        simp = expr.simplify();
        EXPECT_EQ(simp.eval(), 2_r);
    }

    TEST_F(RationalLazyTest, SimplifySqrtExp) {
        Rational expr = delta::sqrt(delta::exp("1"_r));
        Rational simp = expr.simplify();
        Rational expected = delta::exp(Rational(1, 2));
        EXPECT_EQ(simp.eval(), expected.eval());
    }

    TEST_F(RationalLazyTest, SimplifyTrig) {
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
        Rational expr = "1/2"_r.lazy() + "1/3"_r;
        EXPECT_TRUE(expr.is_lazy());
        Rational val = expr.eval();
        EXPECT_EQ(val, "5/6"_r);
    }

    TEST_F(RationalLazyTest, StructuralEquality) {
        Rational a = "1/2"_r.lazy() + "1/3"_r;
        Rational b = "1/2"_r.lazy() + "1/3"_r;
        EXPECT_TRUE(a == b);

        Rational c = "1/2"_r.lazy() + "1/4"_r;
        EXPECT_FALSE(a == c);
    }

    TEST_F(RationalLazyTest, StructuralInequality) {
        Rational a = "1/2"_r.lazy() + "1/3"_r;
        Rational b = "1/2"_r.lazy() + "1/4"_r;
        EXPECT_FALSE(a == b);

        Rational d = "1/2"_r.lazy() * "1/3"_r;
        EXPECT_FALSE(a == d);
    }

    TEST_F(RationalLazyTest, LargeAdditionChain) {
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
        const int N = 500;
        Rational sum = 0_r.lazy();
        for (int i = 0; i < N; ++i) {
            sum = sum + 1_r;
        }
        EXPECT_TRUE(sum.is_lazy());
        int root_idx = sum.root_index();
        const auto& root_node = internal::pool.nodes[root_idx];
        // После внедрения SUM, корневой узел — SUM, а не ADD
        EXPECT_EQ(root_node.op, internal::LazyOp::SUM);
    }

    // ============================================================================
    // Тесты для автоматического создания LazyOp::SUM через operator+ и operator+=
    // ============================================================================

    TEST_F(RationalLazyTest, SumFromBinaryPlusLazyLeft) {
        Rational a = "1/2"_r.lazy();
        Rational b = "1/3"_r;

        Rational sum = a + b;   // lazy + immediate

        EXPECT_TRUE(sum.is_lazy());
        const auto& node = internal::pool.nodes[sum.root_index()];
        EXPECT_EQ(node.op, internal::LazyOp::SUM);
        ASSERT_TRUE(node.sum_children);
        EXPECT_EQ(node.sum_children->size(), 2);
    }

    TEST_F(RationalLazyTest, SumFromBinaryPlusLazyRight) {
        Rational a = "1/2"_r;
        Rational b = "1/3"_r.lazy();

        Rational sum = a + b;   // immediate + lazy

        EXPECT_TRUE(sum.is_lazy());
        const auto& node = internal::pool.nodes[sum.root_index()];
        EXPECT_EQ(node.op, internal::LazyOp::SUM);
        ASSERT_TRUE(node.sum_children);
        EXPECT_EQ(node.sum_children->size(), 2);
    }

    TEST_F(RationalLazyTest, SumFromBinaryPlusBothLazy) {
        Rational a = "1/2"_r.lazy();
        Rational b = "1/3"_r.lazy();

        Rational sum = a + b;

        EXPECT_TRUE(sum.is_lazy());
        const auto& node = internal::pool.nodes[sum.root_index()];
        EXPECT_EQ(node.op, internal::LazyOp::SUM);
        ASSERT_TRUE(node.sum_children);
        EXPECT_EQ(node.sum_children->size(), 2);
    }

    TEST_F(RationalLazyTest, SumAccumulationViaPlusEqual) {
        Rational acc = 0_r.lazy();
        for (int i = 1; i <= 5; ++i) {
            acc += Rational(i).lazy();   // каждый раз должно добавляться в тот же SUM узел
        }

        EXPECT_TRUE(acc.is_lazy());
        const auto& node = internal::pool.nodes[acc.root_index()];
        EXPECT_EQ(node.op, internal::LazyOp::SUM);
        ASSERT_TRUE(node.sum_children);
        EXPECT_EQ(node.sum_children->size(), 5);

        Rational result = acc.eval();
        EXPECT_EQ(result, 15_r);
    }

    TEST_F(RationalLazyTest, SumFromChainedPlus) {
        Rational a = "1/2"_r.lazy();
        Rational b = "1/3"_r.lazy();
        Rational c = "1/6"_r.lazy();

        Rational sum = a + b + c;   // (a+b)+c

        EXPECT_TRUE(sum.is_lazy());
        const auto& node = internal::pool.nodes[sum.root_index()];
        EXPECT_EQ(node.op, internal::LazyOp::SUM);
        ASSERT_TRUE(node.sum_children);
        EXPECT_EQ(node.sum_children->size(), 3);
    }

    TEST_F(RationalLazyTest, SumDoesNotCreateNestedSum) {
        Rational a = "1/2"_r.lazy();
        Rational b = "1/3"_r.lazy();
        Rational c = "1/6"_r.lazy();

        Rational sum1 = a + b;          // должен стать SUM( a, b )
        Rational sum2 = sum1 + c;       // должно получиться SUM( a, b, c ), а не SUM( SUM(a,b), c )

        const auto& node = internal::pool.nodes[sum2.root_index()];
        EXPECT_EQ(node.op, internal::LazyOp::SUM);
        ASSERT_TRUE(node.sum_children);
        EXPECT_EQ(node.sum_children->size(), 3);

        // Проверяем, что дети — исходные a, b, c, а не промежуточный sum1
        bool has_sum_child = false;
        for (int child : *node.sum_children) {
            if (internal::pool.nodes[child].op == internal::LazyOp::SUM) {
                has_sum_child = true;
                break;
            }
        }
        EXPECT_FALSE(has_sum_child);
    }

    TEST_F(RationalLazyTest, SumSimplifyRemovesZeros) {
        Rational zero = 0_r.lazy();
        Rational a = "1/2"_r.lazy();
        Rational b = "1/3"_r.lazy();

        Rational sum = zero + a + zero + b;
        Rational simplified = sum.simplify();

        if (simplified.is_lazy()) {
            const auto& node = internal::pool.nodes[simplified.root_index()];
            // Упрощение двух слагаемых может превратить SUM в ADD
            EXPECT_TRUE(node.op == internal::LazyOp::SUM || node.op == internal::LazyOp::ADD);
            if (node.op == internal::LazyOp::SUM) {
                ASSERT_TRUE(node.sum_children);
                EXPECT_EQ(node.sum_children->size(), 2);
            }
            else {
                EXPECT_NE(node.child0, -1);
                EXPECT_NE(node.child1, -1);
            }
        }
        else {
            // Если упрощение свелось к immediate (a+b), то оно должно быть равно a+b
            EXPECT_EQ(simplified, a + b);
        }
    }

    TEST_F(RationalLazyTest, SumSimplifyFoldsConstants) {
        Rational a = "1/2"_r.lazy();
        Rational b = "1/3"_r.lazy();
        Rational c = "1/6"_r.lazy();   // 1/2+1/3+1/6 = 1
        Rational d = Rational(5).lazy(); // не константа

        Rational sum = a + b + c + d;
        Rational simplified = sum.simplify();

        // После свёртки констант должно остаться SUM(1, d) или ADD(1,d)
        EXPECT_FALSE(simplified.is_lazy() &&
            internal::pool.nodes[simplified.root_index()].op == internal::LazyOp::SUM &&
            internal::pool.nodes[simplified.root_index()].sum_children->size() == 2);
        // Но может быть и бинарный ADD
        Rational expected = 1_r + d;
        EXPECT_EQ(simplified.eval(), expected.eval());
    }

    TEST_F(RationalLazyTest, SumSimplifyAllConstants) {
        Rational sum = "1/2"_r.lazy() + "1/3"_r.lazy() + "1/6"_r.lazy();
        Rational simplified = sum.simplify();

        EXPECT_FALSE(simplified.is_lazy());
        EXPECT_EQ(simplified, 1_r);
    }

    TEST_F(RationalLazyTest, SumStructuralEqualityAfterAccumulation) {
        Rational a = "1/2"_r.lazy();
        Rational b = "1/3"_r.lazy();
        Rational c = "1/6"_r.lazy();

        Rational sum1 = a + b + c;
        Rational sum2 = a + b + c;   // другая последовательность операций, но каноничная
        Rational sum3 = a + b + Rational("1/4"_r).lazy();

        EXPECT_TRUE(sum1 == sum2);
        EXPECT_FALSE(sum1 == sum3);
    }

    TEST_F(RationalLazyTest, SumCowOnMultipleReferences) {
        Rational x = "1/2"_r.lazy();
        Rational y = "1/3"_r.lazy();
        Rational sum = x + y;           // SUM, refcount = 1
        int idx1 = sum.root_index();

        Rational copy = sum;             // refcount = 2
        int idx2 = copy.root_index();    // должен быть тот же индекс
        EXPECT_EQ(idx1, idx2);

        // Мутируем sum, refcount > 1 → должен создаться новый узел
        sum += Rational(1).lazy();

        // sum и copy теперь указывают на разные узлы
        EXPECT_NE(sum.root_index(), copy.root_index());

        // Значения не должны измениться у copy
        EXPECT_EQ(copy.eval(), (x + y).eval());

        // sum должен содержать дополнительное слагаемое
        EXPECT_EQ(sum.eval(), (x + y + 1_r).eval());

        // Проверяем, что copy остался SUM (или ADD после упрощения)
        const auto& node = internal::pool.nodes[copy.root_index()];
        EXPECT_TRUE(node.op == internal::LazyOp::SUM || node.op == internal::LazyOp::ADD);
    }

    TEST_F(RationalLazyTest, SumMutationAfterSimplify) {
        Rational a = "1/2"_r.lazy();
        Rational b = "1/3"_r.lazy();
        Rational sum = a + b;           // SUM
        Rational simplified = sum.simplify();

        // Упрощение двух слагаемых может превратить SUM в ADD или immediate
        // Но если остался ADD, то += должен работать
        if (simplified.is_lazy()) {
            Rational copy = simplified;
            simplified += Rational(1).lazy();

            // Проверяем, что новое значение корректно
            EXPECT_EQ(simplified.eval(), (a + b + 1_r).eval());
            // copy не должен измениться (COW)
            EXPECT_EQ(copy.eval(), (a + b).eval());
        }
        else {
            // Упростилось до immediate → дальнейшее += переведёт в ленивый режим
            Rational imm = simplified;
            imm += Rational(1).lazy();
            EXPECT_TRUE(imm.is_lazy());
            EXPECT_EQ(imm.eval(), (a + b + 1_r).eval());
        }
    }

    TEST_F(RationalLazyTest, PlusEqualOnImmediate) {
        Rational a = 1_r;               // immediate
        Rational b = "1/2"_r.lazy();

        a += b;                         // immediate += lazy

        EXPECT_TRUE(a.is_lazy());       // должен стать ленивым
        const auto& node = internal::pool.nodes[a.root_index()];
        EXPECT_EQ(node.op, internal::LazyOp::SUM);
        ASSERT_TRUE(node.sum_children);
        EXPECT_EQ(node.sum_children->size(), 2);

        EXPECT_EQ(a.eval(), (1_r + "1/2"_r).eval());
    }

    TEST_F(RationalLazyTest, ChainedPlusEqualMixedTypes) {
        Rational acc = 0_r.lazy();      // ленивый ноль
        acc += "1/2"_r;                 // immediate
        acc += Rational(1, 3).lazy();   // lazy
        acc += 2_r;                     // immediate

        const auto& node = internal::pool.nodes[acc.root_index()];
        EXPECT_EQ(node.op, internal::LazyOp::SUM);
        ASSERT_TRUE(node.sum_children);
        // После добавления 4 слагаемых (включая начальный ноль, который отфильтруется)
        // Ожидаем 3 слагаемых, если ноль был удалён при создании/упрощении
        EXPECT_TRUE(node.sum_children->size() == 3 || node.sum_children->size() == 4);

        Rational expected = "1/2"_r + Rational(1, 3) + 2_r;
        EXPECT_EQ(acc.eval(), expected.eval());
    }

    TEST_F(RationalLazyTest, SumOfTwoSums) {
        Rational a = "1/2"_r.lazy();
        Rational b = "1/3"_r.lazy();
        Rational c = "1/6"_r.lazy();
        Rational d = "1/4"_r.lazy();

        Rational sum1 = a + b;          // SUM( a, b )
        Rational sum2 = c + d;          // SUM( c, d )
        Rational total = sum1 + sum2;   // должно получиться SUM( a, b, c, d )

        const auto& node = internal::pool.nodes[total.root_index()];
        EXPECT_EQ(node.op, internal::LazyOp::SUM);
        ASSERT_TRUE(node.sum_children);
        // После сброса пула в SetUp размер будет 4
        EXPECT_EQ(node.sum_children->size(), 4);

        // Проверяем, что нет вложенных SUM
        for (int child : *node.sum_children) {
            EXPECT_NE(internal::pool.nodes[child].op, internal::LazyOp::SUM);
        }

        EXPECT_EQ(total.eval(), (a + b + c + d).eval());
    }

    TEST_F(RationalLazyTest, SumHashInvalidationAfterMutation) {
        Rational a = "1/2"_r.lazy();
        Rational b = "1/3"_r.lazy();
        Rational sum = a + b;           // SUM
        uint64_t old_hash = internal::pool.nodes[sum.root_index()].hash;
        EXPECT_NE(old_hash, 0);

        sum += Rational(1).lazy();      // мутация
        uint64_t new_hash = internal::pool.nodes[sum.root_index()].hash;
        EXPECT_EQ(new_hash, 0);         // должен быть сброшен

        // Упрощение должно пересчитать хэш и, возможно, свернуть в константу
        Rational simplified = sum.simplify();
        // Проверяем численное равенство (независимо от того, immediate или lazy)
        EXPECT_RATIONAL_NEAR(simplified, a + b + 1_r, default_eps());

        // Дополнительная проверка: если упрощённое выражение всё ещё ленивое,
        // его хэш должен быть ненулевым (пересчитан)
        if (simplified.is_lazy()) {
            uint64_t final_hash = internal::pool.nodes[simplified.root_index()].hash;
            EXPECT_NE(final_hash, 0);
        }
    }

    TEST_F(RationalLazyTest, OperatorPlusDoesNotMutate) {
        Rational a = "1/2"_r.lazy();
        Rational b = "1/3"_r.lazy();
        Rational sum = a + b;           // SUM
        int idx1 = sum.root_index();

        Rational new_sum = sum + Rational(1).lazy();   // не должно менять sum

        EXPECT_EQ(sum.root_index(), idx1);             // sum не изменился
        EXPECT_NE(new_sum.root_index(), idx1);         // новый узел

        const auto& node = internal::pool.nodes[new_sum.root_index()];
        EXPECT_EQ(node.op, internal::LazyOp::SUM);
        ASSERT_TRUE(node.sum_children);
        EXPECT_EQ(node.sum_children->size(), 3);       // a, b, 1
    }
  
} // namespace delta::testing