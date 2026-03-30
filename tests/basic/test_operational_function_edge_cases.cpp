// tests/basic/test_operational_function_edge_cases.cpp
#include <gtest/gtest.h>
#include "../test_fixtures.h"

namespace delta::testing {

    class OperationalFunctionGeneralTest : public DeltaTest {};

    /**
     * @test Create an operational function on a ListGrid and verify query results.
     */
    TEST_F(OperationalFunctionGeneralTest, CreateAndQuery) {
        ListGrid<Addr, Compare> grid({ 0_r, 1_r, 2_r });
        OperationalFunction<Addr, Val, decltype(grid)> func(
            grid, [](const Addr& x) { return x * x; });

        EXPECT_EQ(func(0_r), 0_r);
        EXPECT_EQ(func(1_r), 1_r);
        EXPECT_EQ(func(2_r), 4_r);
    }

    /**
     * @test Querying an address not present in the function throws std::out_of_range.
     */
    TEST_F(OperationalFunctionGeneralTest, QueryMissingAddress) {
        ListGrid<Addr, Compare> grid({ 0_r, 2_r });
        OperationalFunction<Addr, Val, decltype(grid)> func(
            grid, [](const Addr& x) { return x; });
        EXPECT_THROW(func(1_r), std::out_of_range);
    }

    /**
     * @test contains() correctly distinguishes between present and absent addresses.
     */
    TEST_F(OperationalFunctionGeneralTest, Contains) {
        ListGrid<Addr, Compare> grid({ 0_r, 2_r });
        OperationalFunction<Addr, Val, decltype(grid)> func(
            grid, [](const Addr& x) { return x; });
        EXPECT_TRUE(func.contains(0_r));
        EXPECT_TRUE(func.contains(2_r));
        EXPECT_FALSE(func.contains(1_r));
    }

    /**
     * @test Extend the function to a refined grid using midpoint interpolation.
     */
    TEST_F(OperationalFunctionGeneralTest, Extend) {
        ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
        OperationalFunction<Addr, Val, decltype(grid0)> func(
            grid0, [](const Addr& x) { return x; });

        auto grid1 = grid0.refine([](const Addr& x, const Addr& y) {
            return (x + y) / 2_r;
            });
        auto interpolator = [](const Addr&, const Addr&,
            const Val& fx, const Val& fy) {
                return (fx + fy) / 2_r;
            };
        func.extend(grid0, grid1, interpolator);

        EXPECT_TRUE(func.contains(0_r));
        EXPECT_TRUE(func.contains(1_r));
        EXPECT_TRUE(func.contains(1_r / 2_r));
        EXPECT_EQ(func(0_r), 0_r);
        EXPECT_EQ(func(1_r), 1_r);
        EXPECT_EQ(func(1_r / 2_r), 1_r / 2_r);
    }

    // -------------------------------------------------------------------------
    // OperationalFunction specialization for UniformGrid
    // -------------------------------------------------------------------------

    /**
     * @class OperationalFunctionUniformTest
     * @brief Edge‑case tests for the OperationalFunction specialization on UniformGrid.
     */
    class OperationalFunctionUniformTest : public DeltaTest {};

    /**
     * @test Create an operational function on a UniformGrid and query values.
     *       Grid covers [0,1] with step 1/4: 0, 1/4, 1/2, 3/4, 1.
     */
    TEST_F(OperationalFunctionUniformTest, CreateAndQuery) {
        UniformGrid<Addr, Compare> grid(0_r, 1_r / 4_r, 5);
        OperationalFunction<Addr, Val, decltype(grid)> func(
            grid, [](const Addr& x) { return x * x; });

        EXPECT_EQ(func(0_r), 0_r);
        EXPECT_EQ(func(1_r / 4_r), (1_r / 4_r) * (1_r / 4_r));
        EXPECT_EQ(func(1_r / 2_r), (1_r / 2_r) * (1_r / 2_r));
        EXPECT_EQ(func(3_r / 4_r), (3_r / 4_r) * (3_r / 4_r));
        EXPECT_EQ(func(1_r), 1_r);
    }

    /**
     * @test Querying an address not belonging to the uniform grid:
     *       contains() returns false, operator() throws an exception.
     */
    TEST_F(OperationalFunctionUniformTest, QueryMissingAddress) {
        UniformGrid<Addr, Compare> grid(0_r, 1_r, 2); // 0, 1
        OperationalFunction<Addr, Val, decltype(grid)> func(
            grid, [](const Addr& x) { return x; });
        // 0.5 is not in the grid
        EXPECT_FALSE(func.contains(1_r / 2_r));
        EXPECT_THROW(func(1_r / 2_r), std::exception);
    }

    /**
     * @test Extend a uniform‑grid function to a finer uniform grid.
     */
    TEST_F(OperationalFunctionUniformTest, Extend) {
        UniformGrid<Addr, Compare> grid0(0_r, 1_r, 2); // 0,1
        OperationalFunction<Addr, Val, decltype(grid0)> func(
            grid0, [](const Addr& x) { return x; });

        UniformGrid<Addr, Compare> grid1(0_r, 1_r / 2_r, 3); // 0, 0.5, 1
        auto interpolator = [](const Addr&, const Addr&,
            const Val& fx, const Val& fy) {
                return (fx + fy) / 2_r;
            };
        func.extend(grid0, grid1, interpolator);

        EXPECT_TRUE(func.contains(0_r));
        EXPECT_TRUE(func.contains(1_r / 2_r));
        EXPECT_TRUE(func.contains(1_r));
        EXPECT_EQ(func(1_r / 2_r), 1_r / 2_r);
    }

    /**
     * @test Verify that contains() correctly identifies points not on the grid.
     */
    TEST_F(OperationalFunctionUniformTest, ContainsNonGridPoint) {
        UniformGrid<Addr, Compare> grid(0_r, 1_r / 3_r, 4); // 0, 1/3, 2/3, 1
        OperationalFunction<Addr, Val, decltype(grid)> func(
            grid, [](const Addr& x) { return x; });
        EXPECT_FALSE(func.contains(1_r / 2_r));
    }

    /**
     * @test Verify that the UniformGrid specialization works with Eigen::MatrixXd values.
     *       (Similar test exists in numerical tests, but repeated here for completeness.)
     */
    TEST_F(OperationalFunctionUniformTest, EigenMatrix) {
        using Matrix = Eigen::MatrixXd;
        UniformGrid<Addr, Compare> grid(0_r, 1_r, 5);
        OperationalFunction<Addr, Matrix, decltype(grid)> func(
            grid, [](const Addr& x) {
                Matrix m(2, 2);
                m.setConstant(x.to_double());
                return m;
            });

        Matrix val = func(2_r);
        EXPECT_DOUBLE_EQ(val(0, 0), 2.0);
        EXPECT_DOUBLE_EQ(val(0, 1), 2.0);
        EXPECT_DOUBLE_EQ(val(1, 0), 2.0);
        EXPECT_DOUBLE_EQ(val(1, 1), 2.0);
    }

} // namespace delta::testing