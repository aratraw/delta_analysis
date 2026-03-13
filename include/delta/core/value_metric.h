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
     * - Rational (boost::multiprecision number) via boost::multiprecision::abs.
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
            using boost::multiprecision::abs;
            return abs(a - b);
        }

        double operator()(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) const {
            return (a - b).norm();
        }

        template<typename T>
        auto operator()(const std::complex<T>& a, const std::complex<T>& b) const {
            using std::abs;
            return abs(a - b);
        }
    };


    // Verify that EuclideanValueMetric satisfies the ValueMetric concept for double.
    static_assert(ValueMetric<EuclideanValueMetric, double, double>);

} // namespace delta