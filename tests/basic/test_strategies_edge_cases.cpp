// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/basic/test_strategies_edge_cases.cpp
#include <gtest/gtest.h>
#include "../test_fixtures.h"

namespace delta::testing {

    /**
     * @class StaticStrategyTest
     * @brief Tests for StaticStrategy.
     *
     * StaticStrategy always returns the same operator regardless of the level.
     */
    class StaticStrategyTest : public DeltaTest {};

    /**
     * @test Verify that the same operator is returned for all levels.
     */
    TEST_F(StaticStrategyTest, SameOperatorForAllLevels) {
        MidpointOperator op;
        auto strategy = StaticStrategy<MidpointOperator>(op);

        for (std::size_t level = 0; level < 10; ++level) {
            const auto& retrieved = strategy.get_operator(level);
            auto info = make_info(0_r, 1_r, 0_r, 0_r, 1_r, level);
            Addr result = retrieved(0_r, 1_r, info);
            EXPECT_EQ(result, 1_r / 2_r);
        }
    }

    /**
     * @class DynamicStrategyTest
     * @brief Tests for DynamicStrategy.
     *
     * DynamicStrategy stores a vector of operators and returns the operator
     * corresponding to the level (falling back to the last one for out‑of‑range levels).
     */
    class DynamicStrategyTest : public DeltaTest {};

    /**
     * @test Constructing with an empty vector should throw std::invalid_argument.
     */
    TEST_F(DynamicStrategyTest, EmptyVectorThrows) {
        using OpType = MidpointOperator;
        EXPECT_THROW({
            DynamicStrategy<OpType> strategy({});
            }, std::invalid_argument);
    }

    /**
     * @test With a single operator, that operator should be returned for all levels.
     */
    TEST_F(DynamicStrategyTest, SingleOperator) {
        MidpointOperator op;
        using OpType = MidpointOperator;
        DynamicStrategy<OpType> strategy({ op });

        for (std::size_t level = 0; level < 10; ++level) {
            const auto& retrieved = strategy.get_operator(level);
            auto info = make_info(0_r, 1_r, 0_r, 0_r, 1_r, level);
            Addr result = retrieved(0_r, 1_r, info);
            EXPECT_EQ(result, 1_r / 2_r);
        }
    }

    /**
     * @test With multiple operators, the correct one should be returned for each level,
     *       and levels beyond the vector size should receive the last operator.
     */
    TEST_F(DynamicStrategyTest, MultipleOperators) {
        FixedLambdaOperator op1(1_r / 3_r);
        FixedLambdaOperator op2(2_r / 3_r);
        using OpType = FixedLambdaOperator;
        DynamicStrategy<OpType> strategy({ op1, op2 });

        // level 0 -> op1
        const auto& op0 = strategy.get_operator(0);
        auto info = make_info(0_r, 1_r, 0_r, 0_r, 1_r);
        Addr result0 = op0(0_r, 1_r, info);
        EXPECT_EQ(result0, 1_r / 3_r);

        // level 1 -> op2
        const auto& op1_level = strategy.get_operator(1);
        Addr result1 = op1_level(0_r, 1_r, info);
        EXPECT_EQ(result1, 2_r / 3_r);

        // level 2 -> fallback to last (op2)
        const auto& op2_level = strategy.get_operator(2);
        Addr result2 = op2_level(0_r, 1_r, info);
        EXPECT_EQ(result2, 2_r / 3_r);
    }

    /**
     * @class FactoryStrategyTest
     * @brief Tests for FactoryStrategy.
     *
     * FactoryStrategy creates a new operator on demand using a user‑provided factory
     * function that receives the level.
     */
    class FactoryStrategyTest : public DeltaTest {};

    /**
     * @test Verify that the factory is called with the correct level each time
     *       get_operator is invoked.
     */
    TEST_F(FactoryStrategyTest, FactoryCalledWithCorrectLevel) {
        std::vector<std::size_t> called_levels;
        using OpType = MidpointOperator;
        auto factory = [&called_levels](std::size_t level) {
            called_levels.push_back(level);
            return MidpointOperator{};
            };
        FactoryStrategy<OpType> strategy(factory);

        // Request operators for different levels
        strategy.get_operator(3);
        strategy.get_operator(5);
        strategy.get_operator(3); // again level 3

        EXPECT_EQ(called_levels.size(), 3);
        EXPECT_EQ(called_levels[0], 3);
        EXPECT_EQ(called_levels[1], 5);
        EXPECT_EQ(called_levels[2], 3);
    }

} // namespace delta::testing