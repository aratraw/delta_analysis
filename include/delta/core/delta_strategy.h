// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/core/delta_strategy.h
#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <functional>
#include "delta_operator.h"

namespace delta {

    // -----------------------------------------------------------------------------
    // DeltaStrategyConcept
    // -----------------------------------------------------------------------------

    /**
     * @concept DeltaStrategyConcept
     * @brief Requirements for a refinement strategy.
     *
     * A strategy provides a delta operator for a given refinement level.
     *
     * @tparam S           The strategy type.
     * @tparam Addr        Address type.
     * @tparam Value       Function value type.
     * @tparam Distance    Scalar type for distances.
     * @tparam Betweenness Betweenness relation type.
     * @tparam Metric      Address metric type.
     * @tparam ValueMetric Value metric type.
     *
     * The expression `s.get_operator(level)` must return an object that satisfies
     * `DeltaOperator<Addr, Value, Distance, Betweenness, Metric, ValueMetric>`.
     */
    template<typename S, typename Addr, typename Value, typename Distance,
        typename Betweenness, typename Metric, typename ValueMetric>
    concept DeltaStrategyConcept = requires(S s, std::size_t level) {
        { s.get_operator(level) } -> DeltaOperator<Addr, Value, Distance, Betweenness, Metric, ValueMetric>;
    };

    // -----------------------------------------------------------------------------
    // StaticStrategy
    // -----------------------------------------------------------------------------

    /**
     * @class StaticStrategy
     * @brief Strategy that always returns the same operator, regardless of level.
     *
     * @tparam Op The delta operator type (must satisfy DeltaOperator concept).
     */
    template<typename Op>
    class StaticStrategy {
    public:
        using operator_type = Op;

        /**
         * @brief Construct from an operator instance.
         * @param op The operator to be used at every level.
         */
        explicit StaticStrategy(Op op) : op_(std::move(op)) {}

        /**
         * @brief Return the stored operator (ignores the level argument).
         */
        const Op& get_operator(std::size_t /*level*/) const {
            return op_;
        }

    private:
        Op op_;   ///< The fixed operator.
    };

    // -----------------------------------------------------------------------------
    // DynamicStrategy
    // -----------------------------------------------------------------------------

    /**
     * @class DynamicStrategy
     * @brief Strategy that selects an operator based on the level from a pre‑defined list.
     *
     * The list may contain operators for different refinement levels.
     * If the requested level exceeds the list size, the last operator is returned.
     *
     * @tparam Op The delta operator type.
     */
    template<typename Op>
    class DynamicStrategy {
    public:
        using operator_type = Op;

        /**
         * @brief Construct from a vector of operators.
         * @param ops Vector of operators; must not be empty.
         * @throws std::invalid_argument if ops is empty.
         */
        explicit DynamicStrategy(std::vector<Op> ops) : operators_(std::move(ops)) {
            if (operators_.empty()) {
                throw std::invalid_argument("DynamicStrategy: operators vector cannot be empty");
            }
        }

        /**
         * @brief Return the operator for the given level.
         *
         * If level is within the vector size, returns operators_[level];
         * otherwise returns the last operator.
         */
        const Op& get_operator(std::size_t level) const {
            if (level < operators_.size()) {
                return operators_[level];
            }
            else {
                return operators_.back();
            }
        }

    private:
        std::vector<Op> operators_;   ///< List of operators for successive levels.
    };

    // -----------------------------------------------------------------------------
    // FactoryStrategy
    // -----------------------------------------------------------------------------

    /**
     * @class FactoryStrategy
     * @brief Strategy that creates a new operator on demand using a factory function.
     *
     * The factory is called with the level and must return an appropriate operator.
     *
     * @tparam Op The delta operator type.
     */
    template<typename Op>
    class FactoryStrategy {
    public:
        using operator_type = Op;
        /// Type of the factory function: Op(std::size_t level)
        using Factory = std::function<Op(std::size_t)>;

        /**
         * @brief Construct from a factory function.
         * @param factory Callable that takes a level and returns an operator.
         */
        explicit FactoryStrategy(Factory factory) : factory_(std::move(factory)) {}

        /**
         * @brief Invoke the factory to obtain an operator for the given level.
         */
        Op get_operator(std::size_t level) const {
            return factory_(level);
        }

    private:
        Factory factory_;   ///< The factory function.
    };

} // namespace delta