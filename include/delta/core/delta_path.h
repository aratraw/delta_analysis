// include/delta/core/delta_path.h
#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <functional>
#include "list_grid.h"
#include "interval_info.h"
#include "value_metric.h"
#include "delta_strategy.h"
#include "tree_grid.h"

#ifndef DELTA_USE_CACHING
#define DELTA_USE_CACHING 1
#endif

namespace delta {

    /**
     * @class DeltaPath
     * @brief Manages a sequence of refined grids (a path) in Δ‑analysis.
     *
     * A DeltaPath holds a current grid and applies a refinement strategy at each
     * step to produce the next grid. It caches function values to avoid recomputation
     * and optionally uses OpenMP to compute the maximum oscillation in parallel.
     *
     * @tparam Addr         Address type (must satisfy GridConcept requirements).
     * @tparam Value        Function value type.
     * @tparam Distance     Scalar type for distances (e.g., Rational, double).
     * @tparam Betweenness  Betweenness relation type (from regulative idea).
     * @tparam Metric       Address metric type.
     * @tparam ValueMetric  Value metric type.
     * @tparam Strategy     Refinement strategy type (must satisfy DeltaStrategyConcept).
     * @tparam Compare      Comparison functor for ordering addresses (default std::less<Addr>).
     */
    template<typename Addr, typename Value, typename Distance,
        typename Betweenness, typename Metric, typename ValueMetric,
        typename Strategy, typename Compare = std::less<Addr>>
        requires DeltaStrategyConcept<Strategy, Addr, Value, Distance, Betweenness, Metric, ValueMetric>
    class DeltaPath {
    public:
        using GridType = ListGrid<Addr, Compare>;
        using Func = std::function<Value(const Addr&)>;
        using grid_type = GridType;      // Чтобы ProductGrid понимал, что за сетка
        using metric_type = Metric;        // Чтобы ProductPath видел тип метрики
        using value_type = Value;         // На будущее, пригодится
        using addr_type = Addr;

        /**
         * @brief Construct a path from an initial grid and a refinement strategy.
         *
         * @param initial_grid Initial grid (must contain at least the endpoints).
         * @param strategy     Strategy that provides the delta operator for each level.
         * @param betweenness  Betweenness relation (from regulative idea).
         * @param metric       Address metric.
         * @param value_metric Value metric.
         */
        DeltaPath(GridType initial_grid, Strategy strategy,
            Betweenness betweenness, Metric metric, ValueMetric value_metric)
            : current_grid_(std::move(initial_grid))
            , strategy_(std::move(strategy))
            , betweenness_(std::move(betweenness))
            , metric_(std::move(metric))
            , value_metric_(std::move(value_metric))
            , level_(0)
            , use_buffer_a_(true)
        {
        }

        /**
         * @brief Perform one refinement step.
         *
         * Computes the next grid by inserting a new point (given by the strategy's operator)
         * between each consecutive pair of the current grid. Function values are evaluated
         * and cached if DELTA_USE_CACHING is enabled. The maximum oscillation of the current
         * grid is computed (optionally in parallel with OpenMP) and passed to the operator.
         *
         * @param func The function whose values are used for refinement.
         */
        void advance(const Func& func) {
            const auto& grid = current_grid_;
            const std::size_t n = grid.size();
            if (n == 0) {
                ++level_;
                return;
            }
            const auto& op = strategy_.get_operator(level_);

#if DELTA_USE_CACHING
            std::vector<Value> values(n);
            for (std::size_t i = 0; i < n; ++i) {
                values[i] = func(grid[i]);
            }
#endif

            Distance max_osc = Distance{ 0 };

#if defined(_OPENMP) && DELTA_USE_CACHING
#pragma omp parallel
            {
                Distance local_max = Distance{ 0 };
#pragma omp for
                for (std::int64_t i = 0; i < static_cast<std::int64_t>(n - 1); ++i) {
                    Distance d = value_metric_(values[i + 1], values[i]);
                    if (d > local_max) local_max = d;
                }
#pragma omp critical
                {
                    if (local_max > max_osc) max_osc = local_max;
                }
            }
#elif DELTA_USE_CACHING
            for (std::size_t i = 0; i + 1 < n; ++i) {
                Distance d = value_metric_(values[i + 1], values[i]);
                if (d > max_osc) max_osc = d;
            }
#else
            for (std::size_t i = 0; i + 1 < n; ++i) {
                Value vleft = func(grid[i]);
                Value vright = func(grid[i + 1]);
                Distance d = value_metric_(vright, vleft);
                if (d > max_osc) max_osc = d;
            }
#endif

            // Double buffering: alternate between two vectors to avoid reallocation.
            std::vector<Addr>& next = use_buffer_a_ ? buffer_a_ : buffer_b_;
            next.clear();
            next.reserve(2 * n - 1);

            for (std::size_t i = 0; i + 1 < n; ++i) {
                const Addr& left = grid[i];
                const Addr& right = grid[i + 1];

#if DELTA_USE_CACHING
                const Value& vleft = values[i];
                const Value& vright = values[i + 1];
#else
                Value vleft = func(left);
                Value vright = func(right);
#endif

                next.push_back(left);

                IntervalInfo<Addr, Value, Distance, Betweenness, Metric, ValueMetric>
                    info{ left, right, level_, vleft, vright, max_osc,
                          betweenness_, metric_, value_metric_ };
                Addr mid = op(left, right, info);

                next.push_back(std::move(mid));
            }
            next.push_back(grid[n - 1]);

            current_grid_ = GridType(std::move(next), grid.comparator());
            use_buffer_a_ = !use_buffer_a_;
            ++level_;
        }

        /// Returns the current grid (list of addresses at this refinement level).
        const GridType& current_grid() const noexcept { return current_grid_; }

        /// Returns the current refinement level (number of advance steps performed).
        std::size_t level() const noexcept { return level_; }

        /**
         * @brief Compute the maximum gap between consecutive addresses in the current grid.
         * @return The largest distance as measured by the address metric.
         */
        Addr max_gap() const {
            Addr max_g = Addr{ 0 };
            for (std::size_t i = 0; i + 1 < current_grid_.size(); ++i) {
                Addr gap = metric_(current_grid_[i + 1], current_grid_[i]);
                if (gap > max_g) max_g = gap;
            }
            return max_g;
        }


        template<typename ExtMetric>
        auto max_gap(const ExtMetric& ext_metric) const {
            using Distance = decltype(ext_metric(current_grid_[0], current_grid_[0]));
            Distance max_g{ 0 };
            for (std::size_t i = 0; i + 1 < current_grid_.size(); ++i) {
                Distance gap = ext_metric(current_grid_[i + 1], current_grid_[i]);
                if (gap > max_g) max_g = gap;
            }
            return max_g;
        }

    private:
        GridType current_grid_;          ///< The grid at the current level.
        Strategy strategy_;               ///< Strategy providing the delta operator.
        Betweenness betweenness_;         ///< Betweenness relation.
        Metric metric_;                   ///< Address metric.
        ValueMetric value_metric_;        ///< Value metric.
        std::size_t level_;               ///< Number of refinement steps performed.

        // Double‑buffering vectors for the next grid.
        mutable std::vector<Addr> buffer_a_;
        mutable std::vector<Addr> buffer_b_;
        bool use_buffer_a_;               ///< Flag indicating which buffer is currently in use.
    };

    /**
     * @class TreeDeltaPath
     * @brief Specialised path for tree‑structured addresses (binary strings).
     *
     * This path works with a TreeGrid and uses the tree's natural refinement
     * (adding a level of children). It does not use a delta operator; instead,
     * refinement simply increases the tree depth. It is provided for compatibility
     * with algorithms that expect a path interface.
     *
     * @tparam Value        Function value type (unused in grid, only for metric).
     * @tparam ValueMetric  Value metric type (default EuclideanValueMetric).
     */
    template<typename Value, typename ValueMetric = EuclideanValueMetric>
    class TreeDeltaPath {
    public:
        using Addr = std::string;
        using GridType = TreeGrid<std::less<Addr>>;
        using Betweenness = TreeBetweenness;
        using Metric = StringUltrametric;

        using grid_type = GridType;      // Чтобы ProductGrid понимал, что за сетка
        using metric_type = Metric;        // Чтобы ProductPath видел тип метрики
        using value_type = Value;         // На будущее, пригодится
        using addr_type = Addr;
        /**
         * @brief Construct a tree path at level 0 (only the root node).
         * @param vm Value metric (default constructed).
         */
        TreeDeltaPath(ValueMetric vm = ValueMetric{})
            : grid_(0), betweenness_{}, metric_{}, value_metric_(vm) {
        }

        /// Returns the current tree grid.
        const GridType& current_grid() const noexcept { return grid_; }

        /// Returns the current tree depth (level).
        std::size_t level() const noexcept { return grid_.level(); }

        /// Refines the tree by adding one level (all children of current leaves).
        void advance() { grid_.advance(); }

        /**
         * @brief Maximum gap – not meaningful for a tree.
         * @return An empty string (default‑constructed Addr).
         */
        Addr max_gap() const { return Addr{}; }

        /**
         * @brief Maximum gap for tree grid – largest distance between consecutive nodes in lexicographic order.
         * @tparam Metric Address metric type.
         * @param metric The metric to use.
         * @return Maximum distance between consecutive nodes.
         */
        template<typename Metric>
        auto max_gap(const Metric& metric) const {
            using Distance = decltype(metric(Addr{}, Addr{}));
            Distance max_g{ 0 };
            const auto& grid = grid_;
            for (std::size_t i = 0; i + 1 < grid.size(); ++i) {
                Distance d = metric(grid[i], grid[i + 1]);
                if (d > max_g) max_g = d;
            }
            return max_g;
        }

        /// Returns the betweenness relation (TreeBetweenness).
        const Betweenness& betweenness() const noexcept { return betweenness_; }

        /// Returns the address metric (StringUltrametric).
        const Metric& metric() const noexcept { return metric_; }

        /// Returns the value metric.
        const ValueMetric& value_metric() const noexcept { return value_metric_; }

    private:
        GridType grid_;               ///< Current tree grid.
        Betweenness betweenness_;     ///< Tree betweenness relation.
        Metric metric_;               ///< String ultrametric.
        ValueMetric value_metric_;    ///< Value metric (unused in grid but kept for interface).
    };

} // namespace delta