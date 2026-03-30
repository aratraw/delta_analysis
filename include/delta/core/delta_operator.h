// include/delta/core/delta_operator.h
#pragma once

#include <concepts>
#include "interval_info.h"
#include "rational.h"
#include "regulative_idea.h"

namespace delta {

    // -----------------------------------------------------------------------------
    // DeltaOperator concept
    // -----------------------------------------------------------------------------

    /**
     * @concept DeltaOperator
     * @brief Requirements for a delta operator that refines an interval.
     *
     * A delta operator takes the left and right endpoints of an interval,
     * together with contextual information (IntervalInfo), and returns a new
     * address that lies strictly between them (according to the betweenness relation).
     *
     * @tparam Op           The candidate operator type.
     * @tparam Addr         Address type.
     * @tparam Value        Function value type.
     * @tparam Distance     Scalar type for distances.
     * @tparam Betweenness  Betweenness relation type.
     * @tparam Metric       Address metric type.
     * @tparam ValueMetric  Value metric type.
     */
    template<typename Op, typename Addr, typename Value, typename Distance,
        typename Betweenness, typename Metric, typename ValueMetric>
    concept DeltaOperator = requires(Op op,
        const Addr & l, const Addr & r,
        const IntervalInfo<Addr, Value, Distance,
        Betweenness, Metric, ValueMetric>&info) {
            { op(l, r, info) } -> std::convertible_to<Addr>;
    };

    // -----------------------------------------------------------------------------
    // Common predefined operators
    // -----------------------------------------------------------------------------

    /**
     * @struct MidpointOperator
     * @brief Delta operator that always returns the arithmetic midpoint.
     *
     * This operator ignores all contextual information and simply returns
     * (left + right) / 2. It requires that Addr satisfies LinearAddress.
     */
    struct MidpointOperator {
        template<typename Addr, typename Value, typename Distance,
            typename Betweenness, typename Metric, typename ValueMetric>
            requires LinearAddress<Addr>
        Addr operator()(const Addr& left, const Addr& right,
            const IntervalInfo<Addr, Value, Distance,
            Betweenness, Metric, ValueMetric>&) const {
            return (left + right) / Addr{ 2 };
        }
    };

    /**
     * @class FixedLambdaOperator
     * @brief Delta operator that places a new point at a fixed fraction λ of the interval.
     *
     * The new point is computed as left + λ·(right - left). If the computed point
     * lies outside the open interval (due to numerical issues), it falls back to the midpoint
     * and (in debug mode) prints a warning.
     *
     * @note Requires Addr to be LinearAddress with scalar type Rational.
     */
    class FixedLambdaOperator {
    public:
        /**
         * @param lambda Fixed fraction; must be in (0,1) for meaningful results.
         */
        explicit FixedLambdaOperator(const Rational& lambda) : lambda_(lambda) {}

        template<typename Addr, typename Value, typename Distance,
            typename Betweenness, typename Metric, typename ValueMetric>
            requires LinearAddress<Addr, Rational>
        Addr operator()(const Addr& left, const Addr& right,
            const IntervalInfo<Addr, Value, Distance,
            Betweenness, Metric, ValueMetric>&) const {
            Addr mid = left + lambda_ * (right - left);
            // Guard against numerical issues: if mid is outside, fallback to midpoint
            if (mid <= left || mid >= right) {
#ifndef NDEBUG
                std::cerr << "WARNING: FixedLambdaOperator produced out-of-bounds point, using midpoint\n";
#endif
                return (left + right) / Addr{ 2 };
            }
            return mid;
        }

    private:
        Rational lambda_;   ///< Fixed fraction.
    };

    /**
     * @class DynamicLambdaOperator
     * @brief Delta operator where the fraction λ depends on the refinement level.
     *
     * The fraction is obtained from a user‑provided function λ = f(level).
     * If the computed point lies outside the interval, it falls back to the midpoint.
     *
     * @note Requires Addr to be LinearAddress
     */
    class DynamicLambdaOperator {
    public:
        using LambdaFunc = std::function<Rational(std::size_t)>;

        explicit DynamicLambdaOperator(LambdaFunc lambda_gen)
            : lambda_gen_(std::move(lambda_gen)) {
        }

        template<typename Addr, typename Value, typename Distance,
            typename Betweenness, typename Metric, typename ValueMetric>
            requires LinearAddress<Addr, Rational>
        Addr operator()(const Addr& left, const Addr& right,
            const IntervalInfo<Addr, Value, Distance,
            Betweenness, Metric, ValueMetric>& info) const {
            Rational lambda = lambda_gen_(info.level);
            Addr mid = left + lambda * (right - left);
            if (mid <= left || mid >= right) {
#ifndef NDEBUG
                std::cerr << "WARNING: DynamicLambdaOperator produced out-of-bounds point, using midpoint\n";
#endif
                return (left + right) / Addr{ 2 };
            }
            return mid;
        }

    private:
        LambdaFunc lambda_gen_;
    };
    /**
     * @class AdaptiveOperator
     * @brief Delta operator that adaptively places points based on function variation.
     *
     * The point is chosen as left + α·(right - left), where α is computed from the
     * ratio of the variation on the current interval to the global maximum oscillation.
     * This clusters points in regions where the function changes rapidly.
     *
     * - If max_oscillation == 0, returns the midpoint.
     * - If df (|f(right)-f(left)|) ≤ threshold, returns the midpoint.
     * - Otherwise α = df / max_oscillation, clamped to [epsilon, 1‑epsilon].
     *
     * @tparam Addr must satisfy LinearAddress<Addr, Distance> (Distance is the scalar type).
     */
    class AdaptiveOperator {
    public:
        /**
         * @param threshold  If |f(right)-f(left)| ≤ threshold, use midpoint.
         * @param epsilon    Lower and upper clamp for α (ensures α ∈ [epsilon, 1‑epsilon]).
         */
        AdaptiveOperator(const Rational& threshold, const Rational& epsilon)
            : threshold_(threshold), epsilon_(epsilon) {
        }

        template<typename Addr, typename Value, typename Distance,
            typename Betweenness, typename Metric, typename ValueMetric>
            requires LinearAddress<Addr, Distance>
        Addr operator()(const Addr& left, const Addr& right,
            const IntervalInfo<Addr, Value, Distance,
            Betweenness, Metric, ValueMetric>& info) const {
            if (info.max_oscillation == Distance(0)) {
                return (left + right) / Addr{ 2 };
            }
            Distance df = info.value_metric(info.f_right, info.f_left);
            if (df <= Distance(threshold_)) {
                return (left + right) / Addr{ 2 };
            }
            Distance alpha = df / info.max_oscillation;
            if (alpha < Distance(epsilon_)) alpha = Distance(epsilon_);
            if (alpha > Distance(1) - Distance(epsilon_)) alpha = Distance(1) - Distance(epsilon_);
            Addr mid = left + alpha * (right - left);
            if (mid <= left || mid >= right) {
#ifndef NDEBUG
                std::cerr << "WARNING: AdaptiveOperator produced out-of-bounds point, using midpoint\n";
#endif
                return (left + right) / Addr{ 2 };
            }
            return mid;
        }

    private:
        Rational threshold_;   ///< Variation threshold.
        Rational epsilon_;     ///< Clamping factor.
    };

    // -----------------------------------------------------------------------------
    // Operators for non‑linear regulative ideas (stubs / specialisations)
    // -----------------------------------------------------------------------------

    /**
     * @struct MatrixMidpointOperator
     * @brief Midpoint operator for matrix addresses (Eigen::MatrixXd).
     *
     * Returns (left + right) * 0.5. Requires Addr = Eigen::MatrixXd.
     */
    struct MatrixMidpointOperator {
        template<typename Addr, typename Value, typename Distance,
            typename Betweenness, typename Metric, typename ValueMetric>
            requires std::is_same_v<Addr, Eigen::MatrixXd>
        Addr operator()(const Addr& left, const Addr& right,
            const IntervalInfo<Addr, Value, Distance,
            Betweenness, Metric, ValueMetric>&) const {
            return (left + right) * 0.5;
        }
    };

    /**
     * @struct TreeMidpointOperator
     * @brief Placeholder for a midpoint operator on tree addresses.
     *
     * @note Not implemented – triggers a static_assert if instantiated.
     */
    struct TreeMidpointOperator {
        template<typename Addr, typename Value, typename Distance,
            typename Betweenness, typename Metric, typename ValueMetric>
        Addr operator()(const Addr& left, const Addr& right,
            const IntervalInfo<Addr, Value, Distance,
            Betweenness, Metric, ValueMetric>&) const {
            static_assert(sizeof(Addr) == 0, "TreeMidpointOperator is not implemented");
            return Addr{};
        }
    };

    /**
     * @struct PAdicMidpointOperator
     * @brief Midpoint operator for p‑adic rational addresses.
     *
     * For Rational addresses, simply returns the arithmetic midpoint (left + right)/2.
     * The p‑adic metric does not affect the choice of the point.
     *
     * @tparam p The prime (unused, kept for consistency).
     */
    template<int p>
    struct PAdicMidpointOperator {
        template<typename Addr, typename Value, typename Distance,
            typename Betweenness, typename Metric, typename ValueMetric>
            requires std::is_same_v<Addr, Rational>
        Addr operator()(const Addr& left, const Addr& right,
            const IntervalInfo<Addr, Value, Distance,
            Betweenness, Metric, ValueMetric>&) const {
            return (left + right) / 2;
        }
    };

} // namespace delta