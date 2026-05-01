// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

#pragma once
#include "lazy_rational_test_fixture.h"
#include "delta/core/rational.h"
#include "test_utils.h"
#include <vector>
#include <chrono>

using namespace delta;
using namespace delta::testing;


class LazyRationalContractTest : public LazyRationalTestFixture {};
// ---------------------------------------------------------------------
// 1. Конструкторы и базовое состояние
// ---------------------------------------------------------------------

TEST_F(LazyRationalContractTest, default_constructor_creates_dirty_const_zero) {
    LazyRational a;
    ASSERT_TRUE(is_dirty(a));
    EXPECT_EQ(dirty_node_count(a), 1);
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::CONST);
    EXPECT_EQ(dirty_constant(a, dirty_root_value_idx(a)), Rational(0).value());
}

TEST_F(LazyRationalContractTest, constructor_from_rational_creates_dirty_const) {
    Rational r(3, 2);
    LazyRational a(r);
    ASSERT_TRUE(is_dirty(a));
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::CONST);
    EXPECT_EQ(dirty_constant(a, dirty_root_value_idx(a)), r.value());
}

// ---------------------------------------------------------------------
// 2. Мутирующие операторы (накопление)
// ---------------------------------------------------------------------

TEST_F(LazyRationalContractTest, plus_operator_mutates_left_lvalue) {
    LazyRational a = LazyRational(Rational(1));
    LazyRational b = LazyRational(Rational(2));
    LazyRational& ref = a + b;
    EXPECT_EQ(&ref, &a);
    ASSERT_TRUE(is_dirty(a));
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::SUM);
    EXPECT_EQ(total_operands(a), 2);
    int root = dirty_root_index(a);
    EXPECT_EQ(dirty_node_leaf_count(a, root), 2);
    EXPECT_EQ(Rational(dirty_node_leaf_value(a, root, 0)), 1_r);
    EXPECT_EQ(Rational(dirty_node_leaf_value(a, root, 1)), 2_r);
}

TEST_F(LazyRationalContractTest, plus_operator_on_rvalue_mutates_temporary) {
    LazyRational a = LazyRational(Rational(1));
    LazyRational b = LazyRational(Rational(2));
    LazyRational&& result = std::move(a) + b;
    EXPECT_TRUE(is_dirty(result));
    EXPECT_EQ(dirty_root_op(result), internal::LazyOp::SUM);
    EXPECT_EQ(total_operands(result), 2);
}

TEST_F(LazyRationalContractTest, chained_plus_accumulates_in_place) {
    LazyRational a = LazyRational(Rational(1));
    a + Rational(2) + Rational(3) + Rational(4);
    ASSERT_TRUE(is_dirty(a));
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::SUM);
    EXPECT_EQ(total_operands(a), 4);
}

TEST_F(LazyRationalContractTest, compound_assign_plus_accumulates) {
    LazyRational a = LazyRational(Rational(1));
    a += Rational(2);
    a += Rational(3);
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::SUM);
    EXPECT_EQ(total_operands(a), 3);
}

// ---------------------------------------------------------------------
// 3. Вычитание через NEG
// ---------------------------------------------------------------------

TEST_F(LazyRationalContractTest, subtraction_converts_to_neg_and_sum) {
    LazyRational a = LazyRational(Rational(10));
    a - Rational(3);
    ASSERT_TRUE(is_dirty(a));
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::SUM);
    EXPECT_EQ(total_operands(a), 2);
    int root = dirty_root_index(a);
    EXPECT_EQ(dirty_node_leaf_count(a, root), 1);
    EXPECT_EQ(dirty_node_complex_count(a, root), 1);
    EXPECT_EQ(Rational(dirty_node_leaf_value(a, root, 0)), 10_r);
    int neg_node = dirty_node_complex_child(a, root, 0);
    EXPECT_EQ(dirty_node_op(a, neg_node), internal::LazyOp::NEG);
    EXPECT_EQ(dirty_node_children(a, neg_node).size(), 1);
    int const_node = dirty_node_children(a, neg_node)[0];
    EXPECT_EQ(dirty_node_op(a, const_node), internal::LazyOp::CONST);
    EXPECT_EQ(dirty_constant(a, dirty_node_value_idx(a, const_node)), Rational(3).value());
}

TEST_F(LazyRationalContractTest, double_negation_optimization_on_creation) {
    LazyRational a = LazyRational(Rational(5));
    LazyRational b = -a;
    LazyRational c = -b;
    EXPECT_EQ(c.eval(), a.eval());
}

// ---------------------------------------------------------------------
// 4. Умножение и деление
// ---------------------------------------------------------------------

TEST_F(LazyRationalContractTest, multiplication_creates_product) {
    LazyRational a = LazyRational(Rational(2));
    a* Rational(3);
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::PRODUCT);
    EXPECT_EQ(total_operands(a), 2);
}

TEST_F(LazyRationalContractTest, division_converts_to_recip_and_product) {
    LazyRational a = LazyRational(Rational(6));
    a / Rational(2);
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::PRODUCT);
    EXPECT_EQ(total_operands(a), 2);
    int root = dirty_root_index(a);
    EXPECT_EQ(dirty_node_leaf_count(a, root), 1);
    EXPECT_EQ(dirty_node_complex_count(a, root), 1);
    EXPECT_EQ(Rational(dirty_node_leaf_value(a, root, 0)), 6_r);
    int recip_node = dirty_node_complex_child(a, root, 0);
    EXPECT_EQ(dirty_node_op(a, recip_node), internal::LazyOp::RECIP);
    EXPECT_EQ(dirty_node_children(a, recip_node).size(), 1);
    int const_node = dirty_node_children(a, recip_node)[0];
    EXPECT_EQ(dirty_node_op(a, const_node), internal::LazyOp::CONST);
    EXPECT_EQ(dirty_constant(a, dirty_node_value_idx(a, const_node)), Rational(2).value());
}

// ---------------------------------------------------------------------
// 5. Канонизация (Dirty -> Clean)
// ---------------------------------------------------------------------

TEST_F(LazyRationalContractTest, canonicalize_converts_dirty_to_clean) {
    LazyRational a = LazyRational(Rational(1));
    a + Rational(2);
    ASSERT_TRUE(is_dirty(a));
    a.simplify_inplace();
    EXPECT_TRUE(is_clean(a));
    EXPECT_GE(clean_root_index(a), 0);
}

TEST_F(LazyRationalContractTest, canonicalize_removes_zero_from_sum) {
    LazyRational a = LazyRational(Rational(0));
    a + Rational(5);
    a.simplify_inplace();
    EXPECT_TRUE(is_clean(a));
    const auto& node = internal::pool.nodes[clean_root_index(a)];
    EXPECT_EQ(node.op, internal::LazyOp::CONST);
    EXPECT_EQ(internal::pool.values[node.value_idx], Rational(5).value());
}

TEST_F(LazyRationalContractTest, canonicalize_removes_one_from_product) {
    LazyRational a = LazyRational(Rational(1));
    a* Rational(7);
    a.simplify_inplace();
    EXPECT_TRUE(is_clean(a));
    const auto& node = internal::pool.nodes[clean_root_index(a)];
    EXPECT_EQ(node.op, internal::LazyOp::CONST);
    EXPECT_EQ(internal::pool.values[node.value_idx], Rational(7).value());
}

TEST_F(LazyRationalContractTest, canonicalize_flattens_nested_sums) {
    LazyRational a = LazyRational(Rational(1));
    LazyRational b = LazyRational(Rational(2));
    LazyRational c = LazyRational(Rational(3));
    (a + b) + c;
    a.simplify_inplace();
    EXPECT_TRUE(is_clean(a));
    const auto& node = internal::pool.nodes[clean_root_index(a)];
    EXPECT_EQ(node.op, internal::LazyOp::SUM);
    size_t total = node.leaf_values.size() + node.children.size();
    EXPECT_EQ(total, 3);
    std::vector<Rational> values;
    for (const auto& v : node.leaf_values) {
        values.push_back(Rational(v));
    }
    for (int32_t child : node.children) {
        const auto& child_node = internal::pool.nodes[child];
        EXPECT_EQ(child_node.op, internal::LazyOp::CONST);
        values.push_back(Rational(internal::pool.values[child_node.value_idx]));
    }
    std::sort(values.begin(), values.end());
    EXPECT_EQ(values[0], 1_r);
    EXPECT_EQ(values[1], 2_r);
    EXPECT_EQ(values[2], 3_r);
}

// ---------------------------------------------------------------------
// 6. Интернирование (кэширование чистых узлов)
// ---------------------------------------------------------------------

TEST_F(LazyRationalContractTest, identical_expressions_share_clean_nodes) {
    reset_global_pool();
    LazyRational a = LazyRational(Rational(1)) + Rational(2);
    LazyRational b = LazyRational(Rational(1)) + Rational(2);
    a.simplify_inplace();
    b.simplify_inplace();
    EXPECT_EQ(clean_root_index(a), clean_root_index(b));
    EXPECT_EQ(clean_node_refcount(a, clean_root_index(a)), 2);
}

// ---------------------------------------------------------------------
// 7. Сравнения (требуют канонизации)
// ---------------------------------------------------------------------

TEST_F(LazyRationalContractTest, comparison_canonicalizes_implicitly) {
    LazyRational a = LazyRational(Rational(1)) + Rational(2);
    LazyRational b = LazyRational(Rational(3));
    ASSERT_TRUE(is_dirty(a));
    bool eq = (a == b);
    EXPECT_TRUE(eq);
    EXPECT_TRUE(is_clean(a));
    EXPECT_TRUE(is_clean(b));
}

// ---------------------------------------------------------------------
// 8. Приблизительный интервал
// ---------------------------------------------------------------------

TEST_F(LazyRationalContractTest, approx_interval_returns_estimate) {
    LazyRational a = LazyRational(Rational(1)) + Rational(2);
    auto interval = a.approx_interval();
    EXPECT_TRUE(is_dirty(a));
    EXPECT_LE(interval.lower(), 3.0);
    EXPECT_GE(interval.upper(), 3.0);
    EXPECT_LT(interval.upper() - interval.lower(), 1e-6);
}

// ---------------------------------------------------------------------
// 9. Move-only семантика
// ---------------------------------------------------------------------

TEST_F(LazyRationalContractTest, lazy_rational_is_move_only) {
    LazyRational a = LazyRational(Rational(1));
    LazyRational b = std::move(a);
    (void)b;
}

TEST_F(LazyRationalContractTest, clone_creates_deep_copy) {
    LazyRational a = LazyRational(Rational(1)) + Rational(2);
    LazyRational b = a.clone();
    a += Rational(3);
    EXPECT_EQ(total_operands(a), 3);
    EXPECT_EQ(total_operands(b), 2);
}

// ---------------------------------------------------------------------
// 10. Широкое дерево (много слагаемых) — не вызывает переполнения стека
// ---------------------------------------------------------------------
TEST_F(LazyRationalContractTest, wide_tree_does_not_cause_stack_overflow) {
    LazyRational acc;
    const int N = 100000;
    for (int i = 0; i < N; ++i) {
        acc += Rational(i);
    }
    acc.simplify_inplace();
    Rational sum = acc.eval();
    (void)sum;
}

// ---------------------------------------------------------------------
// 10.1 Глубокое дерево (вложенные трансцендентные функции) — не вызывает
//     переполнения стека при N=5,10,20,50,100
// ---------------------------------------------------------------------
TEST_F(LazyRationalContractTest, deep_transcendental_tree_does_not_cause_stack_overflow) {
    const std::vector<int> depths = { 5, 10, 20, 50, 100 };

    for (int N : depths) {
        // Строим цепочку sin(cos(exp(log(...(x)...)))) глубины N
        // Начинаем с простой константы, чтобы каждый узел имел смысл
        LazyRational expr = LazyRational(Rational(1, 2));  // x = 0.5

        for (int i = 0; i < N; ++i) {
            // Циклически применяем sin → cos → exp → log
            switch (i % 4) {
            case 0: expr = delta::Sin(expr); break;
            case 1: expr = delta::Cos(expr); break;
            case 2: expr = delta::Exp(expr); break;
            case 3: expr = delta::Log(expr); break;
            }
        }

        // Вычисляем — это должно пройти без переполнения стека
        Rational result = expr.eval();
        (void)result;
    }
}

// ---------------------------------------------------------------------
// 10.2 Экстремально глубокое дерево (N=1000) — стресс-тест итеративного обхода
// ---------------------------------------------------------------------
TEST_F(LazyRationalContractTest, extreme_depth_tree_stress_test) {
    const int N = 1000;

    // Используем только Sin/Cos, так как Exp/Log на больших глубинах
    // могут выйти за пределы области определения
    LazyRational expr = LazyRational(Rational(1, 2));

    for (int i = 0; i < N; ++i) {
        if (i % 2 == 0)
            expr = delta::Sin(expr);
        else
            expr = delta::Cos(expr);
    }

    // Упрощаем и вычисляем — самый глубокий тест
    expr.simplify_inplace();
    Rational result = expr.eval();
    (void)result;
}

// ---------------------------------------------------------------------
// 11. Вычисление (eval)
// ---------------------------------------------------------------------

TEST_F(LazyRationalContractTest, eval_returns_correct_immediate) {
    LazyRational a = LazyRational(Rational(1, 2)) + Rational(1, 3);
    Rational r = a.eval();
    EXPECT_EQ(r, Rational(5, 6));
}

TEST_F(LazyRationalContractTest, eval_on_clean_does_not_modify) {
    LazyRational a = LazyRational(Rational(1)) + Rational(2);
    a.simplify_inplace();
    int old_index = clean_root_index(a);
    Rational r = a.eval();
    EXPECT_EQ(r, Rational(3));
    EXPECT_EQ(clean_root_index(a), old_index);
}

// ---------------------------------------------------------------------
// 12. Преобразование Rational -> LazyRational и обратно
// ---------------------------------------------------------------------

TEST_F(LazyRationalContractTest, as_lazy_creates_dirty_const) {
    Rational r(5, 2);
    LazyRational lr = r.as_lazy();
    EXPECT_TRUE(is_dirty(lr));
    EXPECT_EQ(dirty_root_op(lr), internal::LazyOp::CONST);
    EXPECT_EQ(dirty_constant(lr, dirty_root_value_idx(lr)), r.value());
}

// ---------------------------------------------------------------------
// 13. Отсутствие узлов SUB и DIV
// ---------------------------------------------------------------------

TEST_F(LazyRationalContractTest, no_sub_or_div_nodes_created) {
    LazyRational a = LazyRational(Rational(10));
    a - Rational(3);
    EXPECT_TRUE(has_node_with_op(a, internal::LazyOp::NEG));

    LazyRational b = LazyRational(Rational(6));
    b / Rational(2);
    EXPECT_TRUE(has_node_with_op(b, internal::LazyOp::RECIP));
}

// ---------------------------------------------------------------------
// 14. Каноничность чистых узлов
// ---------------------------------------------------------------------

TEST_F(LazyRationalContractTest, clean_sum_is_canonical) {
    LazyRational a = LazyRational(Rational(0)) + Rational(2) + Rational(1) + Rational(0);
    a.simplify_inplace();
    EXPECT_TRUE(is_clean(a));
    EXPECT_TRUE(is_canonical_sum(a));
    EXPECT_EQ(a.eval(), 3_r);
}

// ---------------------------------------------------------------------
// 15. Производительность
// ---------------------------------------------------------------------

TEST_F(LazyRationalContractTest, linear_time_accumulation) {
    const int N = 10000;
    LazyRational acc;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        acc += Rational(i);
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    EXPECT_LT(elapsed.count(), 100);
}

//Если будет падать этот тест - ПРИСМОТРИСЬ К import_tree и ensure_dirty.
TEST_F(LazyRationalContractTest, SumWithSqrtNoGC) {
    Rational eps = "1/1000000000000000000000000000000"_r;
    set_precision(eps);
    LazyRational half = Rational(1, 2).as_lazy();
    LazyRational sqrt2 = delta::lazy_sqrt(Rational(2).as_lazy());
    sqrt2.simplify_inplace();//changes nothing but the status: dirty to clean tree
    Rational val2 = sqrt2.eval();          // значение sqrt(2) до всяких GC
    LazyRational sum = half.clone() + sqrt2.clone();
    sum.simplify_inplace();
    Rational val3 = sum.eval();            // сумма 1/2 + sqrt(2)
    Rational expected = Rational(1, 2) + val2;
    EXPECT_EQ(val3, expected);             // должно совпадать
}