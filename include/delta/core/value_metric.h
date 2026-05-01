// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/core/value_metric.h
#pragma once

#include <concepts>
#include <Eigen/Dense>
#include "rational.h"

namespace delta {

    /**
     * @concept ValueMetric
     * @brief Concept for a metric on function values.
     *
     * A ValueMetric must be callable as `vm(a, b)` and return a type convertible to `Distance`.
     * It is used to measure the distance (or difference) between two function values,
     * for example in continuity and differentiability checks.
     *
     * @tparam VM The candidate metric type.
     * @tparam Value The value type of the function.
     * @tparam Distance The scalar type representing distances.
     */
    template<typename VM, typename Value, typename Distance>
    concept ValueMetric = requires(VM vm, const Value & a, const Value & b) {
        { vm(a, b) } -> std::convertible_to<Distance>;
    };

    /**
     * @struct EuclideanValueMetric
     * @brief A metric that computes the Euclidean (absolute) distance between two values.
     *
     * This metric works for:
     * - Arithmetic types (int, double, etc.) via abs.
     * - Rational or custom Rational via delta::abs()
     * - Eigen::MatrixXd via the Frobenius norm.
     *
     * The returned type is the same as the result of the absolute operation for the given type.
     */
    struct EuclideanValueMetric {
        template<typename T>
        auto operator()(const T& a, const T& b) const -> decltype(abs(a - b)) {
            return abs(a - b);
        }

        auto operator()(const Rational& a, const Rational& b) const {
            using delta::abs;
            return abs(a - b);// abs returns our custom Rational 
        }

        double operator()(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) const {
            return (a - b).norm();
        }

        template<typename T>
        auto operator()(const std::complex<T>& a, const std::complex<T>& b) const {
            using std::abs;
            return abs(a - b);
        }

        // Для double – оставляем double (для полноты)
        double operator()(double a, double b) const {
            return std::abs(a - b);
        }
    };

} // namespace delta