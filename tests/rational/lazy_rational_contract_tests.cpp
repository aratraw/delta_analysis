// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/rational/lazy_rational_contract_test.cpp
// ============================================================================
// CONTRACT TESTS FOR LAZYRATIONAL – MUTABLE LAZY EXPRESSION TREES
// ============================================================================
//
// This file tests the core contract of the LazyRational class:
//   - Constructors and initial state.
//   - Mutating arithmetic operators (+, -, *, /) and their accumulation behaviour.
//   - Canonicalisation (Dirty → Clean) and algebraic simplifications.
//   - Interning (hash‑consing) – identical expressions share clean nodes.
//   - Comparisons (implicitly canonicalise).
//   - Approximate interval evaluation.
//   - Move‑only semantics and deep cloning.
//   - Performance (linear time accumulation, no stack overflow on deep trees).
//   - Absence of SUB/DIV nodes (they are expressed via NEG and RECIP).
//   - Correctness of sum with sqrt (ensuring import_tree and ensure_dirty work).
//
// All tests are deterministic and use the global epsilon when needed.
// ============================================================================

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
// 1. Constructors and basic state
// ---------------------------------------------------------------------

/**
 * @test default_constructor_creates_dirty_const_zero
 * @brief Default constructor creates a dirty CONST(0) node.
 */
TEST_F(LazyRationalContractTest, default_constructor_creates_dirty_const_zero) {
    LazyRational a;
    ASSERT_TRUE(is_dirty(a));
    EXPECT_EQ(dirty_node_count(a), 1);
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::CONST);
    EXPECT_EQ(dirty_constant(a, dirty_root_value_idx(a)), Rational(0).value());
}

/**
 * @test constructor_from_rational_creates_dirty_const
 * @brief Constructor from Rational creates a dirty CONST node with that value.
 */
TEST_F(LazyRationalContractTest, constructor_from_rational_creates_dirty_const) {
    Rational r(3, 2);
    LazyRational a(r);
    ASSERT_TRUE(is_dirty(a));
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::CONST);
    EXPECT_EQ(dirty_constant(a, dirty_root_value_idx(a)), r.value());
}

// ---------------------------------------------------------------------
// 2. Mutating operators (accumulation)
// ---------------------------------------------------------------------

/**
 * @test plus_operator_mutates_left_lvalue
 * @brief a + b mutates a (left operand) and returns a reference to a.
 */
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

/**
 * @test plus_operator_on_rvalue_mutates_temporary
 * @brief std::move(a) + b treats the moved‑from temporary as mutable.
 */
TEST_F(LazyRationalContractTest, plus_operator_on_rvalue_mutates_temporary) {
    LazyRational a = LazyRational(Rational(1));
    LazyRational b = LazyRational(Rational(2));
    LazyRational&& result = std::move(a) + b;
    EXPECT_TRUE(is_dirty(result));
    EXPECT_EQ(dirty_root_op(result), internal::LazyOp::SUM);
    EXPECT_EQ(total_operands(result), 2);
}

/**
 * @test chained_plus_accumulates_in_place
 * @brief acc + 2 + 3 + 4 accumulates all terms into the same SUM node.
 */
TEST_F(LazyRationalContractTest, chained_plus_accumulates_in_place) {
    LazyRational a = LazyRational(Rational(1));
    a + Rational(2) + Rational(3) + Rational(4);
    ASSERT_TRUE(is_dirty(a));
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::SUM);
    EXPECT_EQ(total_operands(a), 4);
}

/**
 * @test compound_assign_plus_accumulates
 * @brief a += b works similarly (accumulates).
 */
TEST_F(LazyRationalContractTest, compound_assign_plus_accumulates) {
    LazyRational a = LazyRational(Rational(1));
    a += Rational(2);
    a += Rational(3);
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::SUM);
    EXPECT_EQ(total_operands(a), 3);
}

// ---------------------------------------------------------------------
// 3. Subtraction via NEG
// ---------------------------------------------------------------------

/**
 * @test subtraction_converts_to_neg_and_sum
 * @brief a - b is implemented as a + NEG(b).
 */
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

/**
 * @test double_negation_optimization_on_creation
 * @brief -(-x) simplifies to x.
 */
TEST_F(LazyRationalContractTest, double_negation_optimization_on_creation) {
    LazyRational a = LazyRational(Rational(5));
    LazyRational b = -a;
    LazyRational c = -b;
    EXPECT_EQ(c.eval(), a.eval());
}

// ---------------------------------------------------------------------
// 4. Multiplication and division
// ---------------------------------------------------------------------

/**
 * @test multiplication_creates_product
 * @brief a * b creates a PRODUCT node.
 */
TEST_F(LazyRationalContractTest, multiplication_creates_product) {
    LazyRational a = LazyRational(Rational(2));
    a* Rational(3);
    EXPECT_EQ(dirty_root_op(a), internal::LazyOp::PRODUCT);
    EXPECT_EQ(total_operands(a), 2);
}

/**
 * @test division_converts_to_recip_and_product
 * @brief a / b is implemented as a * RECIP(b).
 */
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
// 5. Canonicalisation (Dirty -> Clean)
// ---------------------------------------------------------------------

/**
 * @test canonicalize_converts_dirty_to_clean
 * @brief simplify_inplace() turns a dirty expression into a clean one.
 */
TEST_F(LazyRationalContractTest, canonicalize_converts_dirty_to_clean) {
    LazyRational a = LazyRational(Rational(1));
    a + Rational(2);
    ASSERT_TRUE(is_dirty(a));
    a.simplify_inplace();
    EXPECT_TRUE(is_clean(a));
    EXPECT_GE(clean_root_index(a), 0);
}

/**
 * @test canonicalize_removes_zero_from_sum
 * @brief 0 + x simplifies to x.
 */
TEST_F(LazyRationalContractTest, canonicalize_removes_zero_from_sum) {
    LazyRational a = LazyRational(Rational(0));
    a + Rational(5);
    a.simplify_inplace();
    EXPECT_TRUE(is_clean(a));
    const auto& node = internal::pool.nodes[clean_root_index(a)];
    EXPECT_EQ(node.op, internal::LazyOp::CONST);
    EXPECT_EQ(internal::pool.values[node.value_idx], Rational(5).value());
}

/**
 * @test canonicalize_removes_one_from_product
 * @brief 1 * x simplifies to x.
 */
TEST_F(LazyRationalContractTest, canonicalize_removes_one_from_product) {
    LazyRational a = LazyRational(Rational(1));
    a* Rational(7);
    a.simplify_inplace();
    EXPECT_TRUE(is_clean(a));
    const auto& node = internal::pool.nodes[clean_root_index(a)];
    EXPECT_EQ(node.op, internal::LazyOp::CONST);
    EXPECT_EQ(internal::pool.values[node.value_idx], Rational(7).value());
}

/**
 * @test canonicalize_flattens_nested_sums
 * @brief (a + b) + c flattens into a single SUM node.
 */
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
// 6. Interning (caching of clean nodes)
// ---------------------------------------------------------------------

/**
 * @test identical_expressions_share_clean_nodes
 * @brief Two syntactically identical expressions after canonicalisation share the same clean node.
 */
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
// 7. Comparisons (implicitly canonicalise)
// ---------------------------------------------------------------------

/**
 * @test comparison_canonicalizes_implicitly
 * @brief Operator== triggers canonicalisation on its operands.
 */
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
// 8. Approximate interval
// ---------------------------------------------------------------------

/**
 * @test approx_interval_returns_estimate
 * @brief approx_interval() returns a narrowing interval around the exact value.
 */
TEST_F(LazyRationalContractTest, approx_interval_returns_estimate) {
    LazyRational a = LazyRational(Rational(1)) + Rational(2);
    auto interval = a.approx_interval();
    EXPECT_TRUE(is_dirty(a));
    EXPECT_LE(interval.lower(), 3.0);
    EXPECT_GE(interval.upper(), 3.0);
    EXPECT_LT(interval.upper() - interval.lower(), 1e-6);
}

// ---------------------------------------------------------------------
// 9. Move‑only semantics
// ---------------------------------------------------------------------

/**
 * @test lazy_rational_is_move_only
 * @brief LazyRational is move‑only (copy constructor deleted).
 */
TEST_F(LazyRationalContractTest, lazy_rational_is_move_only) {
    LazyRational a = LazyRational(Rational(1));
    LazyRational b = std::move(a);
    (void)b;
}

/**
 * @test clone_creates_deep_copy
 * @brief clone() makes a deep copy; modifications to the original do not affect the copy.
 */
TEST_F(LazyRationalContractTest, clone_creates_deep_copy) {
    LazyRational a = LazyRational(Rational(1)) + Rational(2);
    LazyRational b = a.clone();
    a += Rational(3);
    EXPECT_EQ(total_operands(a), 3);
    EXPECT_EQ(total_operands(b), 2);
}

// ---------------------------------------------------------------------
// 10. Wide tree (many summands) – does not cause stack overflow
// ---------------------------------------------------------------------
/**
 * @test wide_tree_does_not_cause_stack_overflow
 * @brief Accumulating a large number of terms (100 000) uses iterative evaluation, not recursion.
 */
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
// 10.1 Deep transcendental tree – does not cause stack overflow
// ---------------------------------------------------------------------
/**
 * @test deep_transcendental_tree_does_not_cause_stack_overflow
 * @brief Nested sin, cos, exp, log (depth up to 100) should not cause recursion overflow.
 */
TEST_F(LazyRationalContractTest, deep_transcendental_tree_does_not_cause_stack_overflow) {
    const std::vector<int> depths = { 5, 10, 20, 50, 100 };

    for (int N : depths) {
        // Build a chain sin(cos(exp(log(...(x)...)))) of depth N.
        // Start with a simple constant (0.5) so each node is well‑defined.
        LazyRational expr = LazyRational(Rational(1, 2));

        for (int i = 0; i < N; ++i) {
            // Cycle through sin → cos → exp → log
            switch (i % 4) {
            case 0: expr = delta::Sin(expr); break;
            case 1: expr = delta::Cos(expr); break;
            case 2: expr = delta::Exp(expr); break;
            case 3: expr = delta::Log(expr); break;
            }
        }

        // Evaluate; should not overflow the stack.
        Rational result = expr.eval();
        (void)result;
    }
}

// ---------------------------------------------------------------------
// 10.2 Extreme depth (N=1000) – stress test for iterative traversal
// ---------------------------------------------------------------------
/**
 * @test extreme_depth_tree_stress_test
 * @brief Stress test with a very deep tree (1000 nested sin/cos) to ensure iterative evaluation.
 */
TEST_F(LazyRationalContractTest, extreme_depth_tree_stress_test) {
    const int N = 1000;

    // Use only Sin/Cos because Exp/Log may leave the domain at large depths.
    LazyRational expr = LazyRational(Rational(1, 2));

    for (int i = 0; i < N; ++i) {
        if (i % 2 == 0)
            expr = delta::Sin(expr);
        else
            expr = delta::Cos(expr);
    }

    expr.simplify_inplace();
    Rational result = expr.eval();
    (void)result;
}

// ---------------------------------------------------------------------
// 11. Evaluation (eval)
// ---------------------------------------------------------------------

/**
 * @test eval_returns_correct_immediate
 * @brief eval() computes the exact rational value of a lazy expression.
 */
TEST_F(LazyRationalContractTest, eval_returns_correct_immediate) {
    LazyRational a = LazyRational(Rational(1, 2)) + Rational(1, 3);
    Rational r = a.eval();
    EXPECT_EQ(r, Rational(5, 6));
}

/**
 * @test eval_on_clean_does_not_modify
 * @brief Evaluating a clean expression does not change its clean state.
 */
TEST_F(LazyRationalContractTest, eval_on_clean_does_not_modify) {
    LazyRational a = LazyRational(Rational(1)) + Rational(2);
    a.simplify_inplace();
    int old_index = clean_root_index(a);
    Rational r = a.eval();
    EXPECT_EQ(r, Rational(3));
    EXPECT_EQ(clean_root_index(a), old_index);
}

// ---------------------------------------------------------------------
// 12. Rational → LazyRational conversion and back
// ---------------------------------------------------------------------

/**
 * @test as_lazy_creates_dirty_const
 * @brief Rational::as_lazy() creates a dirty CONST node with the same value.
 */
TEST_F(LazyRationalContractTest, as_lazy_creates_dirty_const) {
    Rational r(5, 2);
    LazyRational lr = r.as_lazy();
    EXPECT_TRUE(is_dirty(lr));
    EXPECT_EQ(dirty_root_op(lr), internal::LazyOp::CONST);
    EXPECT_EQ(dirty_constant(lr, dirty_root_value_idx(lr)), r.value());
}

// ---------------------------------------------------------------------
// 13. Absence of SUB and DIV nodes
// ---------------------------------------------------------------------

/**
 * @test no_sub_or_div_nodes_created
 * @brief Subtraction and division are implemented using NEG and RECIP,
 *        not dedicated SUB/DIV nodes.
 */
TEST_F(LazyRationalContractTest, no_sub_or_div_nodes_created) {
    LazyRational a = LazyRational(Rational(10));
    a - Rational(3);
    EXPECT_TRUE(has_node_with_op(a, internal::LazyOp::NEG));

    LazyRational b = LazyRational(Rational(6));
    b / Rational(2);
    EXPECT_TRUE(has_node_with_op(b, internal::LazyOp::RECIP));
}

// ---------------------------------------------------------------------
// 14. Canonicality of clean nodes
// ---------------------------------------------------------------------

/**
 * @test clean_sum_is_canonical
 * @brief After canonicalisation, a SUM node has no zero terms, and constants are flattened.
 */
TEST_F(LazyRationalContractTest, clean_sum_is_canonical) {
    LazyRational a = LazyRational(Rational(0)) + Rational(2) + Rational(1) + Rational(0);
    a.simplify_inplace();
    EXPECT_TRUE(is_clean(a));
    EXPECT_TRUE(is_canonical_sum(a));
    EXPECT_EQ(a.eval(), 3_r);
}

// ---------------------------------------------------------------------
// 15. Performance
// ---------------------------------------------------------------------

/**
 * @test linear_time_accumulation
 * @brief Accumulating 10 000 terms should complete in less than 100 ms
 *        (linear time, not quadratic).
 */
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

// ---------------------------------------------------------------------
// 16. Sum with sqrt – test for import_tree and ensure_dirty correctness
// ---------------------------------------------------------------------
/**
 * @test SumWithSqrtNoGC
 * @brief Checks that adding a simplified sqrt expression to a constant works correctly.
 *        This test is sensitive to bugs in import_tree and ensure_dirty.
 */
TEST_F(LazyRationalContractTest, SumWithSqrtNoGC) {
    Rational eps = "1/1000000000000000000000000000000"_r;
    set_precision(eps);
    LazyRational half = Rational(1, 2).as_lazy();
    LazyRational sqrt2 = delta::lazy_sqrt(Rational(2).as_lazy());
    sqrt2.simplify_inplace(); // changes status from dirty to clean tree
    Rational val2 = sqrt2.eval();          // value of sqrt(2) before any GC
    LazyRational sum = half.clone() + sqrt2.clone();
    sum.simplify_inplace();
    Rational val3 = sum.eval();            // sum 1/2 + sqrt(2)
    Rational expected = Rational(1, 2) + val2;
    EXPECT_EQ(val3, expected);             // must match
}
