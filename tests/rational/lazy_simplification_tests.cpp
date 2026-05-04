// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/rational/lazy_simplification_tests.cpp
// ============================================================================
// SYMBOLIC SIMPLIFICATION TESTS FOR LAZYRATIONAL EXPRESSIONS
// ============================================================================
//
// This file tests the algebraic simplification rules implemented in simplify_impl.h.
// Verified transformations:
//   - Folding of duplicate scalar constants in sums (a+a+a → 3*a).
//   - Folding of duplicate scalar factors in products (a*a*a → a^3).
//   - Folding of identical sub‑expressions (A+A → 2*A, A*A → A^2).
//   - Distributive law: a*b + a*c → a*(b+c).
//   - Removal of neutral elements (0 in sums, 1 in products).
//   - Combined simplification (fold + distribute).
//   - Interning (hash‑consing) after simplification.
//   - Stress tests for repeating subgraphs (reproducing edge cases from benchmarks).
//
// All simplifications are performed symbolically without evaluating the
// constants to numbers; the resulting tree structure is verified where possible.
// ============================================================================

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

// Helper function – generator of a repeating term
// (as in transcendentals_canonicalization_benchmark.cpp)
static LazyRational generate_repeating_term(int repeats, const Rational& x_val = "0.5"_r) {
    LazyRational term_val = Sin(x_val) * Cos(x_val);
    LazyRational acc;
    for (int i = 0; i < repeats; ++i) {
        acc + term_val;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// 1. Folding of identical scalar constants in a sum
//    3 + 3 + 3  →  PRODUCT(3, CONST(3))   (symbolic, not 9)
// ---------------------------------------------------------------------------
/**
 * @test SumScalarFold
 * @brief Verifies that repeated scalar constants in a sum are folded into a product.
 */
TEST_F(LazySimplificationTests, SumScalarFold) {
    LazyRational acc;
    acc + 3_r + 3_r + 3_r;
    acc.simplify_inplace();
    ASSERT_TRUE(is_clean(acc));

    const auto& root_node = internal::pool.nodes[clean_index(acc)];
    Rational val = acc.eval();
    EXPECT_EQ(val, 9_r);

    // If the root is a PRODUCT, check that it contains factor 3 and 3
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
// 2. Folding of identical scalar constants in a product
//    2 * 2 * 2  →  POW(CONST(2), CONST(3))   (symbolic, not 8)
// ---------------------------------------------------------------------------
/**
 * @test ProductScalarFold
 * @brief Verifies that repeated scalar factors in a product are folded into a power.
 */
TEST_F(LazySimplificationTests, ProductScalarFold) {
    LazyRational acc = LazyRational(2_r);
    acc * 2_r * 2_r;
    acc.simplify_inplace();
    ASSERT_TRUE(is_clean(acc));

    const auto& root_node = internal::pool.nodes[clean_index(acc)];
    Rational val = acc.eval();
    EXPECT_EQ(val, 8_r);

    // Expect to see POW(2, 3)
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
// 3. Folding of identical sub‑expressions in a sum
//    A + A  →  2 * A
// ---------------------------------------------------------------------------
/**
 * @test SumNodeFold
 * @brief Verifies that adding the same sub‑expression twice folds into a product by 2.
 */
TEST_F(LazySimplificationTests, SumNodeFold) {
    LazyRational x = LazyRational("0.5"_r);
    LazyRational sum = x.clone() + x.clone();
    sum.simplify_inplace();
    ASSERT_TRUE(is_clean(sum));

    const auto& root_node = internal::pool.nodes[clean_index(sum)];
    Rational val = sum.eval();
    EXPECT_EQ(val, 1_r);   // 0.5+0.5

    // Structure: should be a product of 2 and CONST(0.5) or similar.
    // Check that the tree contains a PRODUCT node with a leaf 2.
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
// 4. Folding of identical sub‑expressions in a product
//    A * A  →  A^2
// ---------------------------------------------------------------------------
/**
 * @test ProductNodeFold
 * @brief Verifies that multiplying the same sub‑expression twice folds into a power of 2.
 */
TEST_F(LazySimplificationTests, ProductNodeFold) {
    LazyRational x = LazyRational(3_r);
    LazyRational prod = x.clone() * x.clone();
    prod.simplify_inplace();
    ASSERT_TRUE(is_clean(prod));

    const auto& root_node = internal::pool.nodes[clean_index(prod)];
    Rational val = prod.eval();
    EXPECT_EQ(val, 9_r);

    // Should be POW with exponent 2
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
// 5. Distributivity: a*b + a*c  →  a*(b+c)
// ---------------------------------------------------------------------------
/**
 * @test DistributivitySimple
 * @brief Checks that a*b + a*c simplifies to a*(b+c).
 */
TEST_F(LazySimplificationTests, DistributivitySimple) {
    LazyRational a = LazyRational(2_r);
    LazyRational b = LazyRational(3_r);
    LazyRational c = LazyRational(4_r);
    LazyRational expr = (a.clone() * b.clone()) + (a.clone() * c.clone());
    expr.simplify_inplace();
    ASSERT_TRUE(is_clean(expr));

    Rational val = expr.eval();
    EXPECT_EQ(val, 14_r);   // 2*3 + 2*4 = 14

    // Check that the root is a PRODUCT with factor a and a sum (b+c)
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
// 6. Distributivity with several terms: a*b + a*c + a*d → a*(b+c+d)
// ---------------------------------------------------------------------------
/**
 * @test DistributivityMultiple
 * @brief Checks that the distributive law works for three terms.
 */
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

    // Similarly, expect PRODUCT with SUM
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
// 7. Distributivity with a non‑scalar common factor
//    (x*y)*z + (x*y)*2  →  (x*y)*(z+2)
// ---------------------------------------------------------------------------
/**
 * @test DistributivityNonScalarFactor
 * @brief Checks distributivity when the common factor is itself a product.
 */
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

    // Structure: PRODUCT containing the common factor and a SUM (z, 2)
    const auto& root_node = internal::pool.nodes[clean_index(expr)];
    EXPECT_EQ(root_node.op, internal::LazyOp::PRODUCT);
}

// ---------------------------------------------------------------------------
// 8. Distributivity does not break when there is no common factor
// ---------------------------------------------------------------------------
/**
 * @test DistributivityNoCommon
 * @brief Verifies that adding two unrelated products does not trigger distribution.
 */
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

    // Root remains a SUM (or could be evaluated to constant only if numbers were evaluated, but they are not)
    const auto& root_node = internal::pool.nodes[clean_index(expr)];
    EXPECT_EQ(root_node.op, internal::LazyOp::SUM);
}

// ---------------------------------------------------------------------------
// 9. Removal of zeros and ones does not interfere with folding
//    a + 0 + a + 0 + a  →  PRODUCT(3, a)   (zeros disappear)
// ---------------------------------------------------------------------------
/**
 * @test FoldWithZerosAndOnes
 * @brief Checks that neutral elements (0 in sums, 1 in products) are removed before folding.
 */
TEST_F(LazySimplificationTests, FoldWithZerosAndOnes) {
    LazyRational acc;
    acc + 5_r + 0_r + 5_r + 0_r + 5_r;
    acc.simplify_inplace();
    ASSERT_TRUE(is_clean(acc));

    Rational val = acc.eval();
    EXPECT_EQ(val, 15_r);

    // Should be PRODUCT(3,5) or eventually CONST 15 after further simplification.
    // At the very least, there should be no zero nodes in the tree.
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
// 10. Combined simplification: folding + distributivity
//     a + a*b + a*c  (a = 2)  should become a*(1 + b + c) or equivalent
// ---------------------------------------------------------------------------
/**
 * @test CombinedFoldAndDistribute
 * @brief Tests a scenario that requires both folding of constant terms and distribution.
 */
TEST_F(LazySimplificationTests, CombinedFoldAndDistribute) {
    LazyRational a = LazyRational(2_r);
    // a + a*3 + a*4
    LazyRational expr = a.clone() + (a.clone() * 3_r) + (a.clone() * 4_r);
    expr.simplify_inplace();
    ASSERT_TRUE(is_clean(expr));

    Rational val = expr.eval();
    EXPECT_EQ(val, 2_r + 6_r + 8_r);   // 16

    // Expect the root to be a PRODUCT containing a and a SUM (1,3,4) or equivalent.
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
// 11. Check that after simplification nodes become canonical
// ---------------------------------------------------------------------------
/**
 * @test ResultIsCanonical
 * @brief Ensures that the simplified expression is in canonical form (hash‑consed, sorted).
 */
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
// 12. Check that identical constants after folding share a single clean node (interning)
// ---------------------------------------------------------------------------
/**
 * @test InterningAfterFold
 * @brief Verifies that after folding, two identical expressions (3+3+3) end up
 *        sharing the same clean node index.
 */
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
// 13-19. Reproduction of CanonicalizationBenchmark.RepeatingSubgraphInterning
// ---------------------------------------------------------------------------

/**
 * @test RepeatingTerm_Simplify_10
 * @brief Builds a sum of 10 identical transcendental terms, simplifies, and evaluates.
 */
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

/**
 * @test RepeatingTerm_CloneEval_10
 * @brief Clones the repeating term expression and evaluates without simplification.
 */
TEST_F(LazySimplificationTests, RepeatingTerm_CloneEval_10) {
    LazyRational expr = generate_repeating_term(10, "0.5"_r);
    LazyRational expr_copy = expr.clone();
    Rational val = expr_copy.eval();
    Rational term_val = sin("0.5"_r) * cos("0.5"_r);
    Rational expected = term_val * 10;
    EXPECT_EQ(val, expected);
}

/**
 * @test RepeatingTerm_Simplify_50
 * @brief Sum of 50 terms, simplified and evaluated.
 */
TEST_F(LazySimplificationTests, RepeatingTerm_Simplify_50) {
    LazyRational expr = generate_repeating_term(50, "0.5"_r);
    expr.simplify_inplace();
    Rational val = expr.eval();
    Rational term_val = sin("0.5"_r) * cos("0.5"_r);
    Rational expected = term_val * 50;
    EXPECT_EQ(val, expected);
}

/**
 * @test RepeatingTerm_NoSimplify_50
 * @brief Sum of 50 terms evaluated without simplification (direct evaluation).
 */
TEST_F(LazySimplificationTests, RepeatingTerm_NoSimplify_50) {
    LazyRational expr = generate_repeating_term(50, "0.5"_r);
    Rational val = expr.eval(true);
    Rational term_val = sin("0.5"_r) * cos("0.5"_r);
    Rational expected = term_val * 50;
    EXPECT_EQ(val, expected);
}

/**
 * @test RepeatingTerm_Simplify_100
 * @brief Sum of 100 terms, simplified and evaluated.
 */
TEST_F(LazySimplificationTests, RepeatingTerm_Simplify_100) {
    LazyRational expr = generate_repeating_term(100, "0.5"_r);
    expr.simplify_inplace();
    Rational val = expr.eval();
    Rational term_val = sin("0.5"_r) * cos("0.5"_r);
    Rational expected = term_val * 100;
    EXPECT_EQ(val, expected);
}

/**
 * @test RepeatingTerm_CloneSimplify_200
 * @brief Clone of a 200‑term sum, simplified and evaluated.
 */
TEST_F(LazySimplificationTests, RepeatingTerm_CloneSimplify_200) {
    LazyRational expr = generate_repeating_term(200, "0.5"_r);
    LazyRational copy = expr.clone();
    copy.simplify_inplace();
    Rational val = copy.eval();
    Rational term_val = sin("0.5"_r) * cos("0.5"_r);
    Rational expected = term_val * 200;
    EXPECT_EQ(val, expected);
}

/**
 * @test RepeatingTerm_SimplifyOnly_500
 * @brief Simply builds a 500‑term sum and calls simplify_inplace()
 *        to check for any performance or memory issues.
 */
TEST_F(LazySimplificationTests, RepeatingTerm_SimplifyOnly_500) {
    LazyRational expr = generate_repeating_term(500, "0.5"_r);
    expr.simplify_inplace();
    SUCCEED();
}