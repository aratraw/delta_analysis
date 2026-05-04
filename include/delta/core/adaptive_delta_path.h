// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/core/adaptive_delta_path.h
#pragma once

#include <queue>
#include <vector>
#include <functional>
#include <cstddef>
#include <cassert>
#include <boost/container/flat_set.hpp>
#include <boost/container/flat_map.hpp>
#include "rational.h"
#include "value_metric.h"
#include "regulative_idea.h"
#include "delta_operator.h"
#include "interval_info.h"
#include "delta_path.h"
#include "list_grid.h"

namespace delta {

    /**
     * @class AdaptiveDeltaPath
     * @brief Adaptive refinement path using a priority queue of intervals.
     *
     * The path starts from a given set of initial points and adaptively inserts new points
     * based on a priority that reflects the deviation from linearity (i.e., how much the
     * function at the midpoint deviates from the linear interpolation of the endpoints).
     * Values of the function are cached to avoid repeated calls.
     *
     * @tparam Addr         Address type (must satisfy Address concept).
     * @tparam Value        Function value type.
     * @tparam Distance     Scalar type used for distances.
     * @tparam Betweenness  Betweenness relation type.
     * @tparam Metric       Metric on addresses.
     * @tparam ValueMetric  Metric on function values.
     * @tparam Operator     Delta operator type (must satisfy DeltaOperator concept).
     * @tparam Compare      Comparison functor for ordering addresses (default std::less<Addr>).
     */
    template<typename Addr, typename Value, typename Distance,
        typename Betweenness, typename Metric, typename ValueMetric,
        typename Operator, typename Compare = std::less<Addr>>
        requires DeltaOperator<Operator, Addr, Value, Distance, Betweenness, Metric, ValueMetric>
    class AdaptiveDeltaPath {
    public:
        using Func = std::function<Value(const Addr&)>;

        /**
         * @brief Construct an adaptive path from initial addresses and a function.
         *
         * @param initial_points  Initial addresses, must be sorted according to Compare.
         * @param func            Function to compute values.
         * @param op              Delta operator.
         * @param threshold       Priority threshold – intervals with priority ≤ threshold are not refined.
         * @param betweenness     Betweenness relation.
         * @param metric          Address metric.
         * @param value_metric    Value metric.
         *
         * @throws std::invalid_argument if threshold ≤ 0.
         */
        AdaptiveDeltaPath(const std::vector<Addr>& initial_points,
            Func func,
            Operator op,
            Distance threshold = Distance(0),
            Betweenness betweenness = Betweenness{},
            Metric metric = Metric{},
            ValueMetric value_metric = ValueMetric{})
            : func_(std::move(func))
            , op_(std::move(op))
            , threshold_(threshold)
            , betweenness_(std::move(betweenness))
            , metric_(std::move(metric))
            , value_metric_(std::move(value_metric))
            , max_oscillation_(Distance{ 0 })
            , level_(0)
        {
            if (threshold <= Distance(0)) {
                throw std::invalid_argument("AdaptiveDeltaPath: threshold must be positive. Precision within zero margin of error will not be attained within several lifetimes of the universe. Get real.");
            }
            for (const auto& addr : initial_points) {
                points_.insert(addr);
                values_[addr] = func_(addr);
            }
            rebuild_queue();
        }

        /**
         * @brief Factory: create an adaptive path after performing several uniform refinement levels.
         *
         * @param initial_points  Initial addresses (e.g., {left, right}).
         * @param func            Function to compute values.
         * @param op              Delta operator.
         * @param uniform_levels  Number of uniform refinement levels to apply first.
         * @param threshold       Priority threshold for subsequent adaptive refinement.
         * @param betweenness     Betweenness relation.
         * @param metric          Address metric.
         * @param value_metric    Value metric.
         * @return AdaptiveDeltaPath initialized with the uniformly refined grid.
         *
         * @throws std::invalid_argument if threshold ≤ 0.
         */
        static AdaptiveDeltaPath from_uniform(const std::vector<Addr>& initial_points,
            Func func,
            Operator op,
            std::size_t uniform_levels,
            Distance threshold = Distance(0),
            Betweenness betweenness = Betweenness{},
            Metric metric = Metric{},
            ValueMetric value_metric = ValueMetric{}) {
            if (threshold <= Distance(0)) {
                throw std::invalid_argument("AdaptiveDeltaPath::from_uniform: threshold must be positive. Precision within zero margin of error will not be attained within several lifetimes of the universe. Get real.");
            }
            // Build a uniform path using DeltaPath
            ListGrid<Addr, Compare> grid0(initial_points.begin(), initial_points.end(), Compare{});
            auto strategy = StaticStrategy<Operator>(op);
            DeltaPath<Addr, Value, Distance, Betweenness, Metric, ValueMetric,
                decltype(strategy), Compare>
                uniform_path(grid0, strategy, betweenness, metric, value_metric);

            for (std::size_t i = 0; i < uniform_levels; ++i) {
                uniform_path.advance(func);
            }

            // Collect addresses from the final uniform grid
            const auto& final_grid = uniform_path.current_grid();
            std::vector<Addr> points;
            points.reserve(final_grid.size());
            for (const auto& addr : final_grid) {
                points.push_back(addr);
            }

            // Construct adaptive path with these points (values will be recomputed via func)
            return AdaptiveDeltaPath(points, func, op, threshold,
                betweenness, metric, value_metric);
        }

        /**
         * @brief Perform one adaptive refinement step.
         *
         * The highest‑priority interval is taken from the queue. Its midpoint (already computed)
         * is inserted into the point set, and the two sub‑intervals are created and pushed
         * into the queue (with their own midpoints computed using the new priority formula).
         *
         * @return true if an interval was refined, false if queue is empty.
         */
        bool advance() {
            if (queue_.empty()) return false;

            Interval intv = queue_.top();
            queue_.pop();
            ++level_;

            // Insert the midpoint (already cached) into the point set
            points_.insert(intv.mid);
            // The value is already in values_ from when the interval was created

            // Determine if the popped interval had the current maximum oscillation
            bool was_max = (intv.priority == max_oscillation_);

            // Compute variations for the two potential new intervals
            Distance var_left = value_metric_(intv.f_left, intv.f_mid);
            Distance var_right = value_metric_(intv.f_mid, intv.f_right);

            // Update global maximum oscillation incrementally
            if (was_max) {
                update_max_oscillation();
            }
            else {
                if (var_left > max_oscillation_) max_oscillation_ = var_left;
                if (var_right > max_oscillation_) max_oscillation_ = var_right;
            }

            // Create sub‑intervals
            std::size_t child_level = level_;
            auto create_sub = [&](Addr left, Addr right, Value f_left, Value f_right) {
                if (!points_.key_comp()(left, right)) return;  // skip zero-length interval

                IntervalInfo<Addr, Value, Distance, Betweenness, Metric, ValueMetric>
                    info{ left, right, child_level, f_left, f_right, max_oscillation_,
                         betweenness_, metric_, value_metric_ };

                Addr mid = op_(left, right, info);
                if (!betweenness_(left, mid, right)) {
#ifndef NDEBUG
                    std::cerr << "WARNING: Operator returned point not between, using midpoint\n";
#endif
                    mid = (left + right) / 2;
                    if (!betweenness_(left, mid, right)) {
                        // Even midpoint is not between – interval cannot be refined further
#ifndef NDEBUG
                        std::cerr << "WARNING: Interval cannot be refined further, skipping\n";
#endif
                        return;  // skip this interval entirely
                    }
                }
                Value f_mid = func_(mid);
                values_[mid] = f_mid;

                // Compute deviation from linearity (main priority criterion)
                Value linear = (f_left + f_right) / 2;
                Distance deviation = value_metric_(linear, f_mid);

                // Optionally compute total variation (not used for priority, but kept for completeness)
                Distance var_l = value_metric_(f_left, f_mid);
                Distance var_r = value_metric_(f_mid, f_right);
                Distance total_var = std::max(var_l, var_r);

                // Priority is solely based on deviation
                Distance priority = deviation;

                if (priority > threshold_) {
                    queue_.push(Interval{ left, right, mid, f_left, f_right, f_mid, priority, child_level });
                }
                };

            create_sub(intv.left, intv.mid, intv.f_left, intv.f_mid);
            create_sub(intv.mid, intv.right, intv.f_mid, intv.f_right);

            return true;
        }

        /// Returns the current set of addresses (sorted).
        const boost::container::flat_set<Addr, Compare>& points() const { return points_; }

        /// Returns the number of points currently in the path.
        std::size_t size() const { return points_.size(); }

        /// Returns the value of the function at a given address (assumes it exists).
        Value value_at(const Addr& x) const { return values_.at(x); }

        /// Returns the current level (number of refinement steps performed).
        std::size_t level() const { return level_; }

        /// Returns the current maximum oscillation between consecutive points.
        Distance max_oscillation() const { return max_oscillation_; }

    private:
        /**
         * @brief Rebuild the priority queue from the current point set.
         *
         * Iterates over all consecutive pairs in points_, computes the midpoint using the operator,
         * evaluates the function at the midpoint, computes the priority (deviation from linearity),
         * and pushes intervals with priority > threshold into the queue.
         * Also updates max_oscillation_ at the end.
         */
        void rebuild_queue() {
            // Clear existing queue
            queue_ = {};

            if (points_.size() < 2) return;

            auto it = points_.begin();
            auto next = std::next(it);
            while (next != points_.end()) {
                Addr left = *it;
                Addr right = *next;
                Value f_left = values_[left];
                Value f_right = values_[right];

                IntervalInfo<Addr, Value, Distance, Betweenness, Metric, ValueMetric>
                    info{ left, right, 0, f_left, f_right, max_oscillation_,
                         betweenness_, metric_, value_metric_ };

                Addr mid = op_(left, right, info);
                bool mid_ok = betweenness_(left, mid, right);
                if (!mid_ok) {
#ifndef NDEBUG
                    std::cerr << "WARNING: Operator returned point not between, using midpoint\n";
#endif
                    mid = (left + right) / 2;
                    mid_ok = betweenness_(left, mid, right);
                    if (!mid_ok) {
                        // Even midpoint is not between – interval cannot be refined
#ifndef NDEBUG
                        std::cerr << "WARNING: Interval cannot be refined, skipping\n";
#endif
                        ++it;
                        ++next;
                        continue; // skip this interval entirely
                    }
                }
                Value f_mid = func_(mid);
                values_[mid] = f_mid;

                // Compute deviation from linearity
                Value linear = (f_left + f_right) / 2;
                Distance deviation = value_metric_(linear, f_mid);

                // Optionally compute total variation (not used for priority)
                Distance var_l = value_metric_(f_left, f_mid);
                Distance var_r = value_metric_(f_mid, f_right);
                Distance total_var = std::max(var_l, var_r);

                // Priority is solely based on deviation
                Distance priority = deviation;

                if (priority > threshold_) {
                    queue_.push(Interval{ left, right, mid, f_left, f_right, f_mid, priority, 0 });
                }

                ++it;
                ++next;
            }

            update_max_oscillation();
        }

        /**
         * @brief Recalculate the maximum oscillation over all consecutive points.
         *
         * Iterates through points_ and updates max_oscillation_ as the maximum
         * value_metric_ difference between values at consecutive addresses.
         */
        void update_max_oscillation() {
            max_oscillation_ = Distance{ 0 };
            auto it = points_.begin();
            if (it == points_.end()) return;
            auto next = std::next(it);
            while (next != points_.end()) {
                Distance d = value_metric_(values_[*next], values_[*it]);
                if (d > max_oscillation_) max_oscillation_ = d;
                ++it;
                ++next;
            }
        }

        /// @brief Represents an interval with its endpoints, midpoint, function values and priority.
        struct Interval {
            Addr left;      ///< Left endpoint.
            Addr right;     ///< Right endpoint.
            Addr mid;       ///< Pre‑computed midpoint.
            Value f_left;   ///< Value at left endpoint.
            Value f_right;  ///< Value at right endpoint.
            Value f_mid;    ///< Value at midpoint.
            Distance priority; ///< Priority (deviation from linearity) – higher means more urgent.
            std::size_t level; ///< Level at which this interval was created.

            /// Comparison for priority queue (largest priority first).
            bool operator<(const Interval& other) const {
                return priority < other.priority;
            }
        };

        boost::container::flat_set<Addr, Compare> points_;   ///< All addresses in the path (sorted).
        boost::container::flat_map<Addr, Value, Compare> values_; ///< Cached function values.
        std::priority_queue<Interval> queue_;                ///< Priority queue of intervals to refine.
        Func func_;                                          ///< Function being approximated.
        Operator op_;                                        ///< Delta operator.
        Distance threshold_;                                 ///< Priority threshold.
        Betweenness betweenness_;                            ///< Betweenness relation.
        [[maybe_unused]] Metric metric_;                     ///< Address metric (may be unused).
        ValueMetric value_metric_;                           ///< Value metric.
        Distance max_oscillation_;                           ///< Current maximum oscillation.
        std::size_t level_;                                  ///< Number of refinement steps performed.
    };

} // namespace delta