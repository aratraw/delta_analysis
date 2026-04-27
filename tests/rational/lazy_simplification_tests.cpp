// lazy_simplification_tests.cpp
// Тесты для проверки символьных упрощений:
// свёртка одинаковых скаляров, свёртка одинаковых узлов,
// дистрибутивность, удаление нейтральных элементов,
// а также воспроизведение сценария RepeatingSubgraphInterning.
// ---------------------------------------------------------------------------
#pragma once
#include "lazy_rational_test_fixture.h"
#include "delta/core/rational.h"
#include "test_utils.h"
#include <vector>
#include <chrono>
#include <stack>

using namespace delta;
using namespace delta::testing;

class LazySimplificationTests : public LazyRationalTestFixture {
//protected:
//    void SetUp() override {
//        internal::reset_pool();
//    }
};

// Вспомогательная функция – генератор повторяющегося терма
// (как в transcendentals_canonicalization_benchmark.cpp)
static LazyRational generate_repeating_term(int repeats, const Rational& x_val = "0.5"_r) {
    LazyRational term_val = Sin(x_val) * Cos(x_val);
    LazyRational acc;
    for (int i = 0; i < repeats; ++i) {
        acc + term_val;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// 1. Свёртка одинаковых скалярных констант в SUM
//    3 + 3 + 3  →  PRODUCT(3, CONST(3))   (символьно, не 9)
// ---------------------------------------------------------------------------
TEST_F(LazySimplificationTests, SumScalarFold) {
    LazyRational acc;
    acc + 3_r + 3_r + 3_r;
    acc.simplify_inplace();
    ASSERT_TRUE(is_clean(acc));

    const auto& root_node = internal::pool.nodes[clean_index(acc)];
    Rational val = acc.eval();
    EXPECT_EQ(val, 9_r);

    // Если корень – PRODUCT, проверим, что он содержит множитель 3 и 3
    if (root_node.op == internal::LazyOp::PRODUCT) {
        bool has_three_as_leaf = false;
        bool has_three_as_child = false;
        for (const auto& v : root_node.leaf_values) {
            if (v == 3_r.value()) has_three_as_leaf = true;
        }
        for (int32_t child : root_node.children) {
            const auto& child_node = internal::pool.nodes[child];
            if (child_node.op == internal::LazyOp::CONST &&
                internal::pool.values[child_node.value_idx] == 3_r.value()) {
                has_three_as_child = true;
            }
        }
        EXPECT_TRUE(has_three_as_leaf || has_three_as_child);
    }
}

// ---------------------------------------------------------------------------
// 2. Свёртка одинаковых скалярных констант в PRODUCT
//    2 * 2 * 2  →  POW(CONST(2), CONST(3))   (символьно, не 8)
// ---------------------------------------------------------------------------
TEST_F(LazySimplificationTests, ProductScalarFold) {
    LazyRational acc = LazyRational(2_r);
    acc * 2_r * 2_r;
    acc.simplify_inplace();
    ASSERT_TRUE(is_clean(acc));

    const auto& root_node = internal::pool.nodes[clean_index(acc)];
    Rational val = acc.eval();
    EXPECT_EQ(val, 8_r);

    // Ожидаем увидеть POW(2, 3)
    bool has_pow_structure = false;
    if (root_node.op == internal::LazyOp::POW) {
        const auto& base_node = internal::pool.nodes[root_node.children[0]];
        const auto& exp_node = internal::pool.nodes[root_node.children[1]];
        if (base_node.op == internal::LazyOp::CONST &&
            internal::pool.values[base_node.value_idx] == 2_r.value() &&
            exp_node.op == internal::LazyOp::CONST &&
            internal::pool.values[exp_node.value_idx] == 3_r.value()) {
            has_pow_structure = true;
        }
    }
    EXPECT_TRUE(has_pow_structure) << "Expected POW(2,3) after folding 2*2*2";
}

// ---------------------------------------------------------------------------
// 3. Свёртка одинаковых подвыражений в SUM
//    A + A  →  2 * A
// ---------------------------------------------------------------------------
TEST_F(LazySimplificationTests, SumNodeFold) {
    LazyRational x = LazyRational("0.5"_r);
    LazyRational sum = x.clone() + x.clone();
    sum.simplify_inplace();
    ASSERT_TRUE(is_clean(sum));

    const auto& root_node = internal::pool.nodes[clean_index(sum)];
    Rational val = sum.eval();
    EXPECT_EQ(val, 1_r);   // 0.5+0.5

    // Структура: должно быть произведение 2 * CONST(0.5) или что-то подобное
    // Проверим, что в дереве есть PRODUCT с листом 2
    bool found_product_with_two = false;
    std::stack<int> st;
    st.push(clean_index(sum));
    while (!st.empty()) {
        int idx = st.top(); st.pop();
        const auto& node = internal::pool.nodes[idx];
        if (node.op == internal::LazyOp::PRODUCT) {
            for (const auto& v : node.leaf_values) {
                if (v == 2_r.value()) found_product_with_two = true;
            }
        }
        for (int child : node.children) st.push(child);
    }
    EXPECT_TRUE(found_product_with_two) << "Expected a PRODUCT node with 2 after A+A folding";
}

// ---------------------------------------------------------------------------
// 4. Свёртка одинаковых подвыражений в PRODUCT
//    A * A  →  A^2
// ---------------------------------------------------------------------------
TEST_F(LazySimplificationTests, ProductNodeFold) {
    LazyRational x = LazyRational(3_r);
    LazyRational prod = x.clone() * x.clone();
    prod.simplify_inplace();
    ASSERT_TRUE(is_clean(prod));

    const auto& root_node = internal::pool.nodes[clean_index(prod)];
    Rational val = prod.eval();
    EXPECT_EQ(val, 9_r);

    // Должен быть POW с показателем 2
    bool found_pow_with_two = false;
    if (root_node.op == internal::LazyOp::POW) {
        const auto& exp_node = internal::pool.nodes[root_node.children[1]];
        if (exp_node.op == internal::LazyOp::CONST &&
            internal::pool.values[exp_node.value_idx] == 2_r.value()) {
            found_pow_with_two = true;
        }
    }
    EXPECT_TRUE(found_pow_with_two) << "Expected POW(A,2) after A*A folding";
}

// ---------------------------------------------------------------------------
// 5. Дистрибутивность: a*b + a*c  →  a*(b+c)
// ---------------------------------------------------------------------------
TEST_F(LazySimplificationTests, DistributivitySimple) {
    LazyRational a = LazyRational(2_r);
    LazyRational b = LazyRational(3_r);
    LazyRational c = LazyRational(4_r);
    LazyRational expr = (a.clone() * b.clone()) + (a.clone() * c.clone());
    expr.simplify_inplace();
    ASSERT_TRUE(is_clean(expr));

    Rational val = expr.eval();
    EXPECT_EQ(val, 14_r);   // 2*3 + 2*4 = 14

    // Проверяем, что корень — PRODUCT с множителем a (или его значением) и суммой b+c
    const auto& root_node = internal::pool.nodes[clean_index(expr)];
    if (root_node.op == internal::LazyOp::PRODUCT) {
        bool has_sum_child = false;
        for (int child : root_node.children) {
            if (internal::pool.nodes[child].op == internal::LazyOp::SUM)
                has_sum_child = true;
        }
        EXPECT_TRUE(has_sum_child) << "Expected SUM inside PRODUCT after distribution";
    }
}

// ---------------------------------------------------------------------------
// 6. Дистрибутивность с несколькими слагаемыми: a*b + a*c + a*d → a*(b+c+d)
// ---------------------------------------------------------------------------
TEST_F(LazySimplificationTests, DistributivityMultiple) {
    LazyRational a = LazyRational(2_r);
    LazyRational b = LazyRational(3_r);
    LazyRational c = LazyRational(4_r);
    LazyRational d = LazyRational(5_r);
    LazyRational expr = (a.clone() * b.clone()) + (a.clone() * c.clone()) + (a.clone() * d.clone());
    expr.simplify_inplace();
    ASSERT_TRUE(is_clean(expr));

    Rational val = expr.eval();
    EXPECT_EQ(val, 2_r * (3_r + 4_r + 5_r));

    // Аналогично, ожидаем PRODUCT с SUM
    const auto& root_node = internal::pool.nodes[clean_index(expr)];
    if (root_node.op == internal::LazyOp::PRODUCT) {
        bool has_sum = false;
        for (int child : root_node.children) {
            if (internal::pool.nodes[child].op == internal::LazyOp::SUM)
                has_sum = true;
        }
        EXPECT_TRUE(has_sum) << "Expected SUM inside PRODUCT after multiple distribution";
    }
}

// ---------------------------------------------------------------------------
// 7. Дистрибутивность с нескалярным общим множителем
//    (x*y)*z + (x*y)*2  →  (x*y)*(z+2)
// ---------------------------------------------------------------------------
TEST_F(LazySimplificationTests, DistributivityNonScalarFactor) {
    LazyRational x = LazyRational("0.5"_r);
    LazyRational y = LazyRational(1_r);
    LazyRational z = LazyRational(2_r);
    LazyRational common = x.clone() * y.clone();   // 0.5 * 1
    LazyRational expr = (common.clone() * z.clone()) + (common.clone() * 2_r);
    expr.simplify_inplace();
    ASSERT_TRUE(is_clean(expr));

    Rational val = expr.eval();
    EXPECT_EQ(val, 2_r);   // 0.5*1*2 + 0.5*1*2 = 1+1 = 2

    // Структура: PRODUCT, внутри которого множитель (x*y) и SUM (z, 2)
    const auto& root_node = internal::pool.nodes[clean_index(expr)];
    EXPECT_EQ(root_node.op, internal::LazyOp::PRODUCT);
}

// ---------------------------------------------------------------------------
// 8. Дистрибутивность не ломается, когда нет общих множителей
// ---------------------------------------------------------------------------
TEST_F(LazySimplificationTests, DistributivityNoCommon) {
    LazyRational a = LazyRational(2_r);
    LazyRational b = LazyRational(3_r);
    LazyRational c = LazyRational(4_r);
    LazyRational d = LazyRational(5_r);
    LazyRational expr = (a.clone() * b.clone()) + (c.clone() * d.clone());
    expr.simplify_inplace();
    ASSERT_TRUE(is_clean(expr));

    Rational val = expr.eval();
    EXPECT_EQ(val, 2_r * 3_r + 4_r * 5_r);

    // Корень остаётся SUM (или упростился до CONST? Нет, останется SUM)
    const auto& root_node = internal::pool.nodes[clean_index(expr)];
    EXPECT_EQ(root_node.op, internal::LazyOp::SUM);
}

// ---------------------------------------------------------------------------
// 9. Удаление нулей и единиц не мешает свёртке
//    a + 0 + a + 0 + a  →  PRODUCT(3, a)   (нули исчезли)
// ---------------------------------------------------------------------------
TEST_F(LazySimplificationTests, FoldWithZerosAndOnes) {
    LazyRational acc;
    acc + 5_r + 0_r + 5_r + 0_r + 5_r;
    acc.simplify_inplace();
    ASSERT_TRUE(is_clean(acc));

    Rational val = acc.eval();
    EXPECT_EQ(val, 15_r);

    // Должен быть PRODUCT(3, 5) или просто CONST 15 в результате дальнейшего упрощения.
    // Но как минимум нулей в дереве быть не должно.
    std::stack<int> st;
    st.push(clean_index(acc));
    bool has_zero = false;
    while (!st.empty()) {
        int idx = st.top(); st.pop();
        const auto& node = internal::pool.nodes[idx];
        if (node.op == internal::LazyOp::CONST &&
            internal::is_zero(internal::pool.values[node.value_idx])) {
            has_zero = true;
        }
        for (int child : node.children) st.push(child);
    }
    EXPECT_FALSE(has_zero) << "Zeros should be removed from expression";
}

// ---------------------------------------------------------------------------
// 10. Комбинированное упрощение: свёртка + дистрибутивность
//     a + a*b + a*c  (a = 2)  должно дать a*(1 + b + c) или подобное
// ---------------------------------------------------------------------------
TEST_F(LazySimplificationTests, CombinedFoldAndDistribute) {
    LazyRational a = LazyRational(2_r);
    // a + a*3 + a*4
    LazyRational expr = a.clone() + (a.clone() * 3_r) + (a.clone() * 4_r);
    expr.simplify_inplace();
    ASSERT_TRUE(is_clean(expr));

    Rational val = expr.eval();
    EXPECT_EQ(val, 2_r + 6_r + 8_r);   // 16

    // Ожидаем, что корень – PRODUCT, содержащий a и SUM (1,3,4) или эквивалент.
    const auto& root_node = internal::pool.nodes[clean_index(expr)];
    if (root_node.op == internal::LazyOp::PRODUCT) {
        EXPECT_TRUE(is_canonical_product(expr));
    }
    else if (root_node.op == internal::LazyOp::SUM) {
        EXPECT_TRUE(is_canonical_sum(expr));
    }
    else {
        FAIL() << "Unexpected root node type";
    }
}

// ---------------------------------------------------------------------------
// 11. Проверка, что после упрощения узлы становятся каноническими
// ---------------------------------------------------------------------------
TEST_F(LazySimplificationTests, ResultIsCanonical) {
    LazyRational a = LazyRational(2_r);
    LazyRational b = LazyRational(3_r);
    LazyRational c = LazyRational(4_r);
    LazyRational expr = (a.clone() * b.clone()) + (a.clone() * c.clone());
    expr.simplify_inplace();

    ASSERT_TRUE(is_clean(expr));
    const auto& root_node = internal::pool.nodes[clean_index(expr)];
    if (root_node.op == internal::LazyOp::SUM) {
        EXPECT_TRUE(is_canonical_sum(expr));
    }
    else if (root_node.op == internal::LazyOp::PRODUCT) {
        EXPECT_TRUE(is_canonical_product(expr));
    }
}

// ---------------------------------------------------------------------------
// 12. Проверка, что одинаковые константы после свёртки дают один узел
//     (тест на интернирование)
// ---------------------------------------------------------------------------
TEST_F(LazySimplificationTests, InterningAfterFold) {
    reset_global_pool();
    LazyRational a = LazyRational(3_r);
    a + 3_r + 3_r;
    a.simplify_inplace();
    int idx_a = clean_index(a);

    reset_global_pool();
    LazyRational b = LazyRational(3_r);
    b + 3_r + 3_r;
    b.simplify_inplace();
    int idx_b = clean_index(b);

    EXPECT_EQ(idx_a, idx_b) << "Identical expressions should share the same clean node after folding";
}

// ---------------------------------------------------------------------------
// 13-19. Воспроизведение CanonicalizationBenchmark.RepeatingSubgraphInterning
// ---------------------------------------------------------------------------

TEST_F(LazySimplificationTests, RepeatingTerm_Simplify_10) {
    LazyRational expr = generate_repeating_term(10, "0.5"_r);
    ASSERT_TRUE(is_dirty(expr));
    expr.simplify_inplace();
    ASSERT_TRUE(is_clean(expr));
    Rational val = expr.eval();
    Rational term_val = sin("0.5"_r) * cos("0.5"_r);
    Rational expected = term_val * 10;
    EXPECT_EQ(val, expected);
}

TEST_F(LazySimplificationTests, RepeatingTerm_CloneEval_10) {
    LazyRational expr = generate_repeating_term(10, "0.5"_r);
    LazyRational expr_copy = expr.clone();
    Rational val = expr_copy.eval();
    Rational term_val = sin("0.5"_r) * cos("0.5"_r);
    Rational expected = term_val * 10;
    EXPECT_EQ(val, expected);
}

TEST_F(LazySimplificationTests, RepeatingTerm_Simplify_50) {
    LazyRational expr = generate_repeating_term(50, "0.5"_r);
    expr.simplify_inplace();
    Rational val = expr.eval();
    Rational term_val = sin("0.5"_r) * cos("0.5"_r);
    Rational expected = term_val * 50;
    EXPECT_EQ(val, expected);
}

TEST_F(LazySimplificationTests, RepeatingTerm_NoSimplify_50) {
    LazyRational expr = generate_repeating_term(50, "0.5"_r);
    Rational val = expr.eval(true);
    Rational term_val = sin("0.5"_r) * cos("0.5"_r);
    Rational expected = term_val * 50;
    EXPECT_EQ(val, expected);
}

TEST_F(LazySimplificationTests, RepeatingTerm_Simplify_100) {
    LazyRational expr = generate_repeating_term(100, "0.5"_r);
    expr.simplify_inplace();
    Rational val = expr.eval();
    Rational term_val = sin("0.5"_r) * cos("0.5"_r);
    Rational expected = term_val * 100;
    EXPECT_EQ(val, expected);
}

TEST_F(LazySimplificationTests, RepeatingTerm_CloneSimplify_200) {
    LazyRational expr = generate_repeating_term(200, "0.5"_r);
    LazyRational copy = expr.clone();
    copy.simplify_inplace();
    Rational val = copy.eval();
    Rational term_val = sin("0.5"_r) * cos("0.5"_r);
    Rational expected = term_val * 200;
    EXPECT_EQ(val, expected);
}

TEST_F(LazySimplificationTests, RepeatingTerm_SimplifyOnly_500) {
    LazyRational expr = generate_repeating_term(500, "0.5"_r);
    expr.simplify_inplace();
    SUCCEED();
}