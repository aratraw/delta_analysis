// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0
/**
 *  test_adaptive_path.cpp
 *
 * \brief AdaptiveDeltaPath – adaptive refinement based on deviation from
 * linearity.
 *
 * Demonstrates the construction and usage of `AdaptiveDeltaPath` with
 * `MidpointOperator` and `AdaptiveOperator`. It verifies that the path
 * refines only intervals with high priority and maintains sortedness of the
 * point set. The test also explores threshold behaviour and invariance under
 * many steps.
 *
 * \ingroup examples
 */
#include <gtest/gtest.h>
#include <vector>
#include "../test_fixtures.h"

using namespace delta::testing;

/**
 * @class AdaptivePathTest
 * @brief Tests for AdaptiveDeltaPath class.
 *
 * AdaptiveDeltaPath builds a non‑uniform grid by inserting new points based on
 * a priority criterion (deviation from linearity). This suite verifies basic
 * operations, invariants, and edge cases.
 */
class AdaptivePathTest : public DeltaTest {};

// Helper constant for explicit threshold specification.
// It is small enough not to interfere but positive because AdaptiveDeltaPath
// constructor now requires threshold > 0.
const Dist DEFAULT_THRESHOLD = Rational(1, 1000000);

/**
 * @test Verify that a path can be constructed from two initial points,
 *       that it initially contains exactly those points, and that one
 *       refinement step adds a midpoint.
 */
TEST_F(AdaptivePathTest, Initialization) {
    std::vector<Addr> init = { 0_r, 1_r };
    auto func = [](const Addr& x) { return x * x; };
    MidpointOperator mid_op;

    auto path = make_adaptive_path(init, func, mid_op, DEFAULT_THRESHOLD);

    EXPECT_EQ(path.size(), 2);
    EXPECT_TRUE(is_sorted_set(path.points()));
    EXPECT_TRUE(path.advance());
    EXPECT_EQ(path.size(), 3);
}

/**
 * @test With a midpoint operator and a quadratic function, after one step
 *       the new point should be exactly the midpoint (1/2).
 */
TEST_F(AdaptivePathTest, OneStepMidpoint) {
    std::vector<Addr> init = { 0_r, 1_r };
    auto func = [](const Addr& x) { return x * x; };
    MidpointOperator mid_op;

    auto path = make_adaptive_path(init, func, mid_op, DEFAULT_THRESHOLD);

    EXPECT_TRUE(path.advance());
    auto points = path.points();
    EXPECT_EQ(points.size(), 3);
    auto it = points.begin();
    EXPECT_EQ(*it++, 0_r);
    EXPECT_EQ(*it++, 1_r / 2_r);
    EXPECT_EQ(*it, 1_r);
}

/**
 * @test After several steps using the midpoint operator, the number of points
 *       should increase by one each step, and the set must remain sorted.
 */
TEST_F(AdaptivePathTest, SeveralStepsMidpoint) {
    std::vector<Addr> init = { 0_r, 1_r };
    auto func = [](const Addr& x) { return x * x; };
    MidpointOperator mid_op;

    auto path = make_adaptive_path(init, func, mid_op, DEFAULT_THRESHOLD);

    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(path.advance());
    }
    EXPECT_EQ(path.size(), 5);
    EXPECT_TRUE(is_sorted_set(path.points()));
}

/**
 * @test If the threshold is larger than any possible deviation, no refinement
 *       should occur. For the identity function the deviation is zero, so a
 *       threshold of 1 stops all refinements.
 */
TEST_F(AdaptivePathTest, Threshold) {
    std::vector<Addr> init = { 0_r, 1_r };
    auto func = [](const Addr& x) { return x; };
    MidpointOperator mid_op;
    Dist threshold = 1_r;

    auto path = make_adaptive_path(init, func, mid_op, threshold);

    EXPECT_FALSE(path.advance());
    EXPECT_EQ(path.size(), 2);
}

/**
 * @test Using the adaptive operator (which depends on function values) should
 *       still allow at least one refinement step. The exact point is not
 *       checked, only that the size increases.
 */
TEST_F(AdaptivePathTest, AdaptiveOperator) {
    std::vector<Addr> init = { 0_r, 1_r };
    auto func = [](const Addr& x) { return x * x; };
    AdaptiveOperator adapt_op(1_r / 10_r, 1_r / 10_r);

    auto path = make_adaptive_path(init, func, adapt_op, DEFAULT_THRESHOLD);

    EXPECT_TRUE(path.advance());
    EXPECT_EQ(path.size(), 3);
}

/**
 * @test The betweenness property: all points in the set must be strictly
 *       increasing according to the comparator. This invariant is checked
 *       after every refinement step until the queue empties.
 */
TEST_F(AdaptivePathTest, BetweennessProperty) {
    std::vector<Addr> init = { 0_r, 1_r };
    auto func = [](const Addr& x) { return x * x; };
    MidpointOperator mid_op;
    auto path = make_adaptive_path(init, func, mid_op, DEFAULT_THRESHOLD);
    int steps = 0;
    while (path.advance()) {
        ++steps;
        EXPECT_TRUE(is_sorted_set(path.points()));
    }
    EXPECT_GT(steps, 0);
}

/**
 * @test Many steps with the midpoint operator; the number of points should
 *       increase linearly with the number of refinements, and the set must
 *       stay sorted.
 */
TEST_F(AdaptivePathTest, ManySteps) {
    std::vector<Addr> init = { 0_r, 1_r };
    auto func = [](const Addr& x) { return x * x; };
    MidpointOperator mid_op;

    auto path = make_adaptive_path(init, func, mid_op, DEFAULT_THRESHOLD);

    int steps = 0;
    const int max_steps = 100;
    while (steps < max_steps && path.advance()) {
        ++steps;
    }
    EXPECT_EQ(path.size(), 2 + steps);
    EXPECT_TRUE(is_sorted_set(path.points()));
}

// -------------------------------------------------------------------------
// AdaptiveDeltaPath edge cases
// -------------------------------------------------------------------------

/**
 * @test Construction with an empty list of initial points should yield a path
 *       of size 0, and advance() should do nothing.
 */
TEST_F(AdaptivePathTest, EmptyInitialPoints) {
    std::vector<Addr> init;
    auto func = [](const Addr&) { return Val(0); };
    MidpointOperator op;
    auto path = make_adaptive_path(init, func, op, DEFAULT_THRESHOLD);
    EXPECT_EQ(path.size(), 0);
    EXPECT_FALSE(path.advance()); // nothing to refine
}

/**
 * @test A single initial point leaves no interval to refine, so size stays 1
 *       and advance() returns false.
 */
TEST_F(AdaptivePathTest, SingleInitialPoint) {
    std::vector<Addr> init = { 5_r };
    auto func = [](const Addr& x) { return x; };
    MidpointOperator op;
    auto path = make_adaptive_path(init, func, op, DEFAULT_THRESHOLD);
    EXPECT_EQ(path.size(), 1);
    EXPECT_FALSE(path.advance()); // no intervals
}

/**
 * @test Two points with a threshold that does not suppress refinement:
 *       the first step should insert the midpoint.
 */
TEST_F(AdaptivePathTest, TwoPointsWithThreshold) {
    std::vector<Addr> init = { 0_r, 1_r };
    auto func = [](const Addr& x) { return x * x; };
    MidpointOperator op;
    Dist threshold = 1_r / 10_r; // 0.1
    auto path = make_adaptive_path(init, func, op, threshold);
    EXPECT_TRUE(path.advance());
    EXPECT_EQ(path.size(), 3);
    auto points = path.points();
    auto it = points.begin();
    EXPECT_EQ(*it++, 0_r);
    EXPECT_EQ(*it++, 1_r / 2_r);
    EXPECT_EQ(*it, 1_r);
}

/**
 * @test Two points with a threshold larger than any possible variation:
 *       no refinement occurs.
 */
TEST_F(AdaptivePathTest, TwoPointsAboveThreshold) {
    std::vector<Addr> init = { 0_r, 1_r };
    auto func = [](const Addr& x) { return x; };
    MidpointOperator op;
    Dist threshold = 2_r;
    auto path = make_adaptive_path(init, func, op, threshold);
    EXPECT_FALSE(path.advance());
    EXPECT_EQ(path.size(), 2);
}

/**
 * @test Invariant: after every refinement step the point set must be sorted.
 */
TEST_F(AdaptivePathTest, SortedInvariant) {
    std::vector<Addr> init = { 0_r, 1_r };
    auto func = [](const Addr& x) { return x * x; };
    MidpointOperator op;
    auto path = make_adaptive_path(init, func, op, DEFAULT_THRESHOLD);
    for (int i = 0; i < 10; ++i) {
        if (!path.advance()) break;
        EXPECT_TRUE(is_sorted_set(path.points()));
    }
}

/**
 * @test Invariant: the leftmost point always stays the original left bound,
 *       and the rightmost point always stays the original right bound.
 */
TEST_F(AdaptivePathTest, BoundsInvariant) {
    std::vector<Addr> init = { 0_r, 1_r };
    auto func = [](const Addr& x) { return x * x; };
    MidpointOperator op;
    auto path = make_adaptive_path(init, func, op, DEFAULT_THRESHOLD);
    for (int i = 0; i < 10; ++i) {
        if (!path.advance()) break;
        const auto& pts = path.points();
        EXPECT_EQ(*pts.begin(), 0_r);
        EXPECT_EQ(*pts.rbegin(), 1_r);
    }
}

/**
 * @test Basic functionality with the AdaptiveOperator (which may place points
 *       not exactly at midpoints). Only the size increase and sortedness are
 *       checked.
 */
TEST_F(AdaptivePathTest, AdaptiveOperatorBasic) {
    std::vector<Addr> init = { 0_r, 1_r };
    auto func = [](const Addr& x) { return x * x; };
    AdaptiveOperator adapt_op(1_r / 10_r, 1_r / 10_r);
    auto path = make_adaptive_path(init, func, adapt_op, DEFAULT_THRESHOLD);
    EXPECT_TRUE(path.advance());
    EXPECT_EQ(path.size(), 3);
    // The point should lie somewhere between 0 and 1, but not necessarily the midpoint.
    // Only sortedness is verified.
    EXPECT_TRUE(is_sorted_set(path.points()));
}

/**
 * @test Stress test with many refinements using the midpoint operator.
 */
TEST_F(AdaptivePathTest, ManyRefinementsMidpoint) {
    std::vector<Addr> init = { 0_r, 1_r };
    auto func = [](const Addr& x) { return x * x; };
    MidpointOperator mid_op;
    auto path = make_adaptive_path(init, func, mid_op, DEFAULT_THRESHOLD);

    const int N = 300;
    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(path.advance());
        EXPECT_TRUE(is_sorted_set(path.points()));
    }
    EXPECT_EQ(path.size(), 2 + N);
}

/**
 * @test Check that the queue eventually empties when the threshold is chosen
 *       so that deeper intervals have deviation below the threshold.
 */
TEST_F(AdaptivePathTest, QueueEmpties) {
    std::vector<Addr> init = { 0_r, 1_r };
    auto func = [](const Addr& x) { return x * x; };
    MidpointOperator op;
    Dist threshold = Rational(24, 100); // 0.24 < 0.25
    auto path = make_adaptive_path(init, func, op, threshold);
    // deviation on [0,1] = 0.25 > 0.24 → first step performed
    EXPECT_TRUE(path.advance());
    // after splitting, deviation on the sub‑intervals = 0.0625 < 0.24 → queue empty
    EXPECT_FALSE(path.advance());
    EXPECT_EQ(path.size(), 3);
}

/**
 * @test A very small threshold allows many refinement steps (well over 100).
 */
TEST_F(AdaptivePathTest, VerySmallThreshold) {
    std::vector<Addr> init = { 0_r, 1_r };
    auto func = [](const Addr& x) { return x * x; };
    MidpointOperator op;
    Dist threshold = Rational(1, 1000000);
    auto path = make_adaptive_path(init, func, op, threshold);
    int steps = 0;
    while (path.advance() && steps < 1000) {
        ++steps;
    }
    EXPECT_GT(steps, 100);
}

/**
 * @test Verify that updating the maximum oscillation does not break the
 *       sortedness invariant.
 */
TEST_F(AdaptivePathTest, MaxOscillationConsistency) {
    std::vector<Addr> init = { 0_r, 1_r };
    auto func = [](const Addr& x) { return x * x; };
    AdaptiveOperator adapt_op(1_r / 10_r, 1_r / 10_r);
    auto path = make_adaptive_path(init, func, adapt_op, Rational(1, 1000));

    for (int i = 0; i < 5; ++i) {
        path.advance();
        EXPECT_TRUE(is_sorted_set(path.points()));
    }
}