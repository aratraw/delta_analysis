#include "lazy_rational_test_fixture.h"

#include "delta/core/rational.h"
#include "test_utils.h"
#include <vector>
#include <chrono>

using namespace delta;
using namespace delta::test;

// ---------------------------------------------------------------------
// 1. Конструкторы и базовое состояние
// ---------------------------------------------------------------------

TEST_F(LazyRationalTestFixture, default_constructor_creates_dirty_const_zero) {
    LazyRational a;
    ASSERT_TRUE(is_dirty(a));
    EXPECT_EQ(dirty_node_count(a), 1);
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::CONST);
    EXPECT_EQ(dirty_constant(a, dirty_root_const_index(a)), Rational(0).value());
}

TEST_F(LazyRationalTestFixture, constructor_from_rational_creates_dirty_const) {
    Rational r(3, 2);
    LazyRational a(r);
    ASSERT_TRUE(is_dirty(a));
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::CONST);
    EXPECT_EQ(dirty_constant(a, dirty_root_const_index(a)), r.value());
}

// ---------------------------------------------------------------------
// 2. Мутирующие операторы (накопление)
// ---------------------------------------------------------------------

TEST_F(LazyRationalTestFixture, plus_operator_mutates_left_lvalue) {
    LazyRational a = LazyRational(Rational(1));
    LazyRational b = LazyRational(Rational(2));
    LazyRational& ref = a + b;
    EXPECT_EQ(&ref, &a);                     // возвращает ссылку на a
    ASSERT_TRUE(is_dirty(a));
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::SUM);
    auto children = dirty_root_children(a);
    ASSERT_EQ(children.size(), 2);
    // Дети: первый – исходный CONST(1), второй – CONST(2) (скопированный)
    EXPECT_EQ(dirty_node_op(a, children[0]), internal::LazyOp::CONST);
    EXPECT_EQ(dirty_node_op(a, children[1]), internal::LazyOp::CONST);
}

TEST_F(LazyRationalTestFixture, plus_operator_on_rvalue_mutates_temporary) {
    LazyRational a = LazyRational(Rational(1));
    LazyRational b = LazyRational(Rational(2));
    LazyRational&& result = std::move(a) + b;
    // a теперь в перемещённом состоянии, но result ссылается на временный
    EXPECT_TRUE(is_dirty(result));
    EXPECT_EQ(dirty_root_op(result), internal::LazyOp::SUM);
    // Проверка, что результат содержит два слагаемых
    EXPECT_EQ(dirty_root_children(result).size(), 2);
}

TEST_F(LazyRationalTestFixture, chained_plus_accumulates_in_place) {
    LazyRational a = LazyRational(Rational(1));
    a + Rational(2) + Rational(3) + Rational(4);
    // После цепочки a должен быть SUM с 4 детьми (1,2,3,4) в порядке добавления
    ASSERT_TRUE(is_dirty(a));
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::SUM);
    auto children = dirty_root_children(a);
    EXPECT_EQ(children.size(), 4);
}

TEST_F(LazyRationalTestFixture, compound_assign_plus_accumulates) {
    LazyRational a = LazyRational(Rational(1));
    a += Rational(2);
    a += Rational(3);
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::SUM);
    EXPECT_EQ(dirty_root_children(a).size(), 3);
}

// ---------------------------------------------------------------------
// 3. Вычитание через NEG
// ---------------------------------------------------------------------

TEST_F(LazyRationalTestFixture, subtraction_converts_to_neg_and_sum) {
    LazyRational a = LazyRational(Rational(10));
    a - Rational(3);
    // Должен стать SUM(CONST(10), NEG(CONST(3)))
    ASSERT_TRUE(is_dirty(a));
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::SUM);
    auto children = dirty_root_children(a);
    ASSERT_EQ(children.size(), 2);
    // Первый ребёнок – CONST(10)
    // Второй ребёнок – NEG
    int neg_node = children[1];
    EXPECT_EQ(dirty_node_op(a, neg_node), internal::LazyOp::NEG);
    auto neg_children = dirty_node_children(a, neg_node);
    ASSERT_EQ(neg_children.size(), 1);
    int const_node = neg_children[0];
    EXPECT_EQ(dirty_node_op(a, const_node), internal::LazyOp::CONST);
    EXPECT_EQ(dirty_constant(a, dirty_node_const_index(a, const_node)), Rational(3).value());
}

TEST_F(LazyRationalTestFixture, double_negation_optimization_on_creation) {
    LazyRational a = LazyRational(Rational(5));
    LazyRational b = -a;           // NEG(a)
    LazyRational c = -b;           // NEG(NEG(a)) -> должно упроститься до a
    EXPECT_EQ(c.eval(), a.eval());
}

// ---------------------------------------------------------------------
// 4. Умножение и деление
// ---------------------------------------------------------------------

TEST_F(LazyRationalTestFixture, multiplication_creates_product) {
    LazyRational a = LazyRational(Rational(2));
    a* Rational(3);
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::PRODUCT);
    auto children = dirty_root_children(a);
    EXPECT_EQ(children.size(), 2);
}

TEST_F(LazyRationalTestFixture, division_converts_to_recip_and_product) {
    LazyRational a = LazyRational(Rational(6));
    a / Rational(2);
    // Должен стать PRODUCT(CONST(6), RECIP(CONST(2)))
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::PRODUCT);
    auto children = dirty_root_children(a);
    ASSERT_EQ(children.size(), 2);
    int recip_node = children[1];
    EXPECT_EQ(dirty_node_op(a, recip_node), internal::LazyOp::RECIP);
    auto recip_children = dirty_node_children(a, recip_node);
    EXPECT_EQ(dirty_node_op(a, recip_children[0]), internal::LazyOp::CONST);
    EXPECT_EQ(dirty_constant(a, dirty_node_const_index(a, recip_children[0])), Rational(2).value());
}

// ---------------------------------------------------------------------
// 5. Канонизация (Dirty -> Clean)
// ---------------------------------------------------------------------

TEST_F(LazyRationalTestFixture, canonicalize_converts_dirty_to_clean) {
    LazyRational a = LazyRational(Rational(1));
    a + Rational(2);
    ASSERT_TRUE(is_dirty(a));
    a.simplify_inplace();   // канонизация
    EXPECT_TRUE(is_clean(a));
    EXPECT_GE(clean_root_index(a), 0);
}

TEST_F(LazyRationalTestFixture, canonicalize_removes_zero_from_sum) {
    LazyRational a = LazyRational(Rational(0));
    a + Rational(5);
    a.simplify_inplace();
    // После канонизации ноль должен быть удалён, останется CONST(5)
    EXPECT_TRUE(is_clean(a));
    const auto& node = internal::pool.nodes[clean_root_index(a)];
    EXPECT_EQ(node.op, internal::LazyOp::CONST);
    EXPECT_EQ(internal::pool.values[node.value_idx], Rational(5).value());
}

TEST_F(LazyRationalTestFixture, canonicalize_removes_one_from_product) {
    LazyRational a = LazyRational(Rational(1));
    a* Rational(7);
    a.simplify_inplace();
    // Должен стать CONST(7)
    EXPECT_TRUE(is_clean(a));
    const auto& node = internal::pool.nodes[clean_root_index(a)];
    EXPECT_EQ(node.op, internal::LazyOp::CONST);
    EXPECT_EQ(internal::pool.values[node.value_idx], Rational(7).value());
}

TEST_F(LazyRationalTestFixture, canonicalize_flattens_nested_sums) {
    LazyRational a = LazyRational(Rational(1));
    LazyRational b = LazyRational(Rational(2));
    LazyRational c = LazyRational(Rational(3));
    (a + b) + c;   // оператор + уже строит плоский SUM(1,2,3)
    a.simplify_inplace();
    EXPECT_TRUE(is_clean(a));
    const auto& node = internal::pool.nodes[clean_root_index(a)];
    EXPECT_EQ(node.op, internal::LazyOp::SUM);
    ASSERT_TRUE(node.children);
    EXPECT_EQ(node.children->size(), 3);

    // Проверяем, что дети – константы 1,2,3 (порядок не важен)
    std::vector<Rational> values;
    for (int32_t child : *node.children) {
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

TEST_F(LazyRationalTestFixture, identical_expressions_share_clean_nodes) {
    reset_global_pool();
    LazyRational a = LazyRational(Rational(1)) + Rational(2);
    LazyRational b = LazyRational(Rational(1)) + Rational(2);
    a.simplify_inplace();
    b.simplify_inplace();
    // Оба должны ссылаться на один и тот же чистый узел в пуле
    EXPECT_EQ(clean_root_index(a), clean_root_index(b));
    // И refcount узла должен быть 2
    EXPECT_EQ(clean_node_refcount(a, clean_root_index(a)), 2);
}

// ---------------------------------------------------------------------
// 7. Сравнения (требуют канонизации)
// ---------------------------------------------------------------------

TEST_F(LazyRationalTestFixture, comparison_canonicalizes_implicitly) {
    LazyRational a = LazyRational(Rational(1)) + Rational(2);
    LazyRational b = LazyRational(Rational(3));
    // a ещё грязный
    ASSERT_TRUE(is_dirty(a));
    bool eq = (a == b);
    EXPECT_TRUE(eq);
    // После сравнения a и b должны стать чистыми
    EXPECT_TRUE(is_clean(a));
    EXPECT_TRUE(is_clean(b));
}

// ---------------------------------------------------------------------
// 8. Приблизительный интервал (обновлённый тест)
// ---------------------------------------------------------------------

TEST_F(LazyRationalTestFixture, approx_interval_returns_estimate) {
    LazyRational a = LazyRational(Rational(1)) + Rational(2);
    auto interval = a.approx_interval();
    // Интервал вычисляется от грязного дерева, объект остаётся грязным
    EXPECT_TRUE(is_dirty(a));   // не канонизируется
    // Приблизительный интервал должен содержать точное значение 3
    EXPECT_LE(interval.lower(), 3.0);
    EXPECT_GE(interval.upper(), 3.0);
    // Дополнительно: интервал не должен быть слишком широким
    EXPECT_LT(interval.upper() - interval.lower(), 1e-6);
}

// ---------------------------------------------------------------------
// 9. Move-only семантика
// ---------------------------------------------------------------------

TEST_F(LazyRationalTestFixture, lazy_rational_is_move_only) {
    LazyRational a = LazyRational(Rational(1));
    LazyRational b = std::move(a);
    // a теперь в перемещённом состоянии
    // Проверка, что копирование запрещено (компиляция не пройдёт)
    // LazyRational c = a;  // ошибка компиляции
    (void)b;
}

TEST_F(LazyRationalTestFixture, clone_creates_deep_copy) {
    LazyRational a = LazyRational(Rational(1)) + Rational(2);
    LazyRational b = a.clone();
    // b должно быть глубокой копией, независимой от a
    a += Rational(3);
    // a теперь SUM(1,2,3), b остаётся SUM(1,2)
    EXPECT_EQ(dirty_root_children(a).size(), 3);
    EXPECT_EQ(dirty_root_children(b).size(), 2);
}

// ---------------------------------------------------------------------
// 10. Отсутствие рекурсии (косвенная проверка через глубину)
// ---------------------------------------------------------------------

TEST_F(LazyRationalTestFixture, deep_tree_does_not_cause_stack_overflow) {
    LazyRational acc;
    const int N = 100000;
    for (int i = 0; i < N; ++i) {
        acc += Rational(i);
    }
    // Просто проверяем, что не упали из-за рекурсии
    acc.simplify_inplace();   // канонизация должна быть итеративной
    Rational sum = acc.eval();
    (void)sum;
}

// ---------------------------------------------------------------------
// 11. Вычисление (eval)
// ---------------------------------------------------------------------

TEST_F(LazyRationalTestFixture, eval_returns_correct_immediate) {
    LazyRational a = LazyRational(Rational(1, 2)) + Rational(1, 3);
    Rational r = a.eval();
    EXPECT_EQ(r, Rational(5, 6));
}

TEST_F(LazyRationalTestFixture, eval_on_clean_does_not_modify) {
    LazyRational a = LazyRational(Rational(1)) + Rational(2);
    a.simplify_inplace();
    int old_index = clean_root_index(a);
    Rational r = a.eval();
    EXPECT_EQ(r, Rational(3));
    EXPECT_EQ(clean_root_index(a), old_index); // индекс не изменился
}

// ---------------------------------------------------------------------
// 12. Преобразование Rational -> LazyRational и обратно
// ---------------------------------------------------------------------

TEST_F(LazyRationalTestFixture, as_lazy_creates_dirty_const) {
    Rational r(5, 2);
    LazyRational lr = r.as_lazy();
    EXPECT_TRUE(is_dirty(lr));
    EXPECT_EQ(dirty_root_op(lr), internal::LazyOp::CONST);
    EXPECT_EQ(dirty_constant(lr, dirty_root_const_index(lr)), r.value());
}

// ---------------------------------------------------------------------
// 13. Отсутствие узлов SUB и DIV (проверка через операции)
// ---------------------------------------------------------------------

TEST_F(LazyRationalTestFixture, no_sub_or_div_nodes_created) {
    LazyRational a = LazyRational(Rational(10));
    a - Rational(3);
    // В грязном дереве не должно быть узлов SUB (их нет в LazyOp)
    // Проверяем, что вместо SUB используется NEG
    EXPECT_TRUE(has_node_with_op(a, internal::LazyOp::NEG));

    LazyRational b = LazyRational(Rational(6));
    b / Rational(2);
    // Вместо DIV используется RECIP
    EXPECT_TRUE(has_node_with_op(b, internal::LazyOp::RECIP));
}

// ---------------------------------------------------------------------
// 14. Каноничность чистых узлов (сортировка детей, удаление нейтральных)
// ---------------------------------------------------------------------

TEST_F(LazyRationalTestFixture, clean_sum_is_canonical) {
    LazyRational a = LazyRational(Rational(0)) + Rational(2) + Rational(1) + Rational(0);
    a.simplify_inplace();
    EXPECT_TRUE(is_clean(a));
    // После упрощения должно быть SUM(1,2) (нули удалены, порядок отсортирован)
    EXPECT_TRUE(is_canonical_sum(a));
    EXPECT_EQ(a.eval(), 3_r);
}

// ---------------------------------------------------------------------
// 15. Производительность (набросок)
// ---------------------------------------------------------------------

TEST_F(LazyRationalTestFixture, linear_time_accumulation) {
    const int N = 10000;
    LazyRational acc;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        acc += Rational(i);
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // Ожидаем, что время ~ O(N), не O(N^2)
    // Просто проверяем, что не слишком большое (например, < 100 мс для N=10000)
    EXPECT_LT(elapsed.count(), 100);
}