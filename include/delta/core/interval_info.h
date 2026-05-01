// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/core/interval_info.h
#pragma once

#include <cstddef>

namespace delta {

    /**
     * @struct IntervalInfo
     * @brief Context information passed to a delta operator for a specific interval.
     *
     * Contains endpoints, level, function values, maximum oscillation, and references
     * to the betweenness relation, the address metric, and the value metric.
     *
     * @tparam Addr Type of addresses.
     * @tparam Value Type of function values.
     * @tparam Distance Scalar type used for distances between values.
     * @tparam Betweenness Type of betweenness relation (from regulative idea).
     * @tparam Metric Type of metric on addresses (from regulative idea).
     * @tparam ValueMetric Type of metric on function values.
     */
    template<typename Addr, typename Value, typename Distance,
        typename Betweenness, typename Metric, typename ValueMetric>
    struct IntervalInfo {
        const Addr& left;          ///< Left endpoint of the interval.
        const Addr& right;         ///< Right endpoint of the interval.
        std::size_t level;         ///< Current refinement level.
        const Value& f_left;       ///< Function value at left endpoint.
        const Value& f_right;      ///< Function value at right endpoint.
        Distance max_oscillation;  ///< Maximum oscillation (max |f(x_i+1)-f(x_i)|) on current level.
        const Betweenness& betweenness;   ///< Betweenness relation from regulative idea.
        const Metric& metric;              ///< Address metric from regulative idea.
        const ValueMetric& value_metric;   ///< Metric on function values.

        /**
         * @brief Constructor initialising all members.
         */
        IntervalInfo(const Addr& l, const Addr& r, std::size_t lvl,
            const Value& fl, const Value& fr, const Distance& max_osc,
            const Betweenness& b, const Metric& m, const ValueMetric& vm)
            : left(l), right(r), level(lvl), f_left(fl), f_right(fr),
            max_oscillation(max_osc), betweenness(b), metric(m), value_metric(vm) {
        }
    };

} // namespace delta