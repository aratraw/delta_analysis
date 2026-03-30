// include/delta/calculus/modulus.h
#pragma once

#include <cmath>
#include <concepts>
#include <type_traits>
#include <cstddef>
#include <limits>
#include "delta/core/rational.h"

namespace delta::calculus {

    // -----------------------------------------------------------------------------
    // Modulus concept
    // -----------------------------------------------------------------------------

    /**
     * @concept Modulus
     * @brief A modulus of continuity or convergence.
     *
     * A modulus is a function that maps a positive step size (e.g., the maximum gap of a grid)
     * to a bound on the variation or error. It is used in continuity and differentiability checks.
     *
     * @tparam M The modulus type.
     * @tparam T The argument and return type (typically a scalar like double or Rational).
     *
     * The expression `m(delta)` must be valid and return a type convertible to T.
     */
    template<typename M, typename T>
    concept Modulus = requires(M m, T delta) {
        { m(delta) } -> std::convertible_to<T>;
    };

    // -----------------------------------------------------------------------------
    // Helper: maximum gap of a grid
    // -----------------------------------------------------------------------------

    /**
     * @brief Compute the maximum distance between consecutive addresses in a grid.
     *
     * For a grid with points x0, x1, ..., x_{n-1}, returns max_{i} (x_{i+1} - x_i).
     *
     * @tparam Grid A type satisfying GridConcept; its value_type must be subtractable.
     * @param grid The grid.
     * @return The largest gap; if grid.size() < 2, returns a default‑constructed value (zero).
     */
    template<typename Grid>
        requires SimpleGrid<Grid>&& SubtractableAddress<typename Grid::value_type>
    typename Grid::value_type max_gap(const Grid& grid) {
        using T = typename Grid::value_type;
        if (grid.size() < 2) return T{ 0 };
        T max_g = T{ 0 };
        for (std::size_t i = 0; i + 1 < grid.size(); ++i) {
            T gap = grid[i + 1] - grid[i];
            if (gap > max_g) max_g = gap;
        }
        return max_g;
    }

    // -----------------------------------------------------------------------------
    // Predefined modulus implementations
    // -----------------------------------------------------------------------------

    /**
     * @class PowerModulus
     * @brief Power‑law modulus: ω(δ) = C * δ^α.
     *
     * @tparam T Scalar type (default double). Must support std::pow and arithmetic.
     */
    template<typename T = double>
    class PowerModulus {
    public:
        /**
         * @brief Construct a power modulus.
         * @param C     Coefficient (multiplier).
         * @param alpha Exponent.
         */
        PowerModulus(T C, T alpha) : C_(C), alpha_(alpha) {}

        /**
         * @brief Evaluate the modulus at a given δ.
         * @param delta Step size (must be non‑negative in meaningful contexts).
         * @return C * δ^α.
         */
        T operator()(T delta) const {
            using std::pow;
            return C_ * pow(delta, alpha_);
        }

    private:
        T C_, alpha_;
    };

    /**
     * @brief Specialisation of PowerModulus for Rational.
     *
     * Since std::pow is not directly available for Rational, we convert to double,
     * compute the power, and convert back. This yields an approximate value,
     * which is sufficient for testing purposes.
     */
    template<>
    class PowerModulus<Rational> {
    public:
        /**
         * @brief Construct a power modulus with Rational parameters.
         * @param C     Coefficient (Rational).
         * @param alpha Exponent (Rational).
         */
        PowerModulus(Rational C, Rational alpha) : C_(C), alpha_(alpha) {}

        /**
         * @brief Evaluate approximately using double arithmetic.
         * @param delta Step size (Rational).
         * @return Rational approximation of C * δ^α.
         */
        Rational operator()(Rational delta) const {
            //SHOULD MAYBE REWRITE IT TO ACTUALLY USE RATIONAL WHEN TESTS SETTLE??
            double d = delta.to_double();
            double a = alpha_.to_double();
            double c = C_.to_double();
            double result = c * std::pow(d, a);
            return Rational(std::to_string(result));
        }

    private:
        Rational C_, alpha_;
    };

    /**
     * @class LogarithmicModulus
     * @brief Logarithmic modulus: ω(δ) = C / |ln δ|^p, for δ > 0.
     *
     * Typically used for slowly converging sequences. For δ ≤ 0, returns infinity.
     *
     * @tparam T Scalar type (default double).
     */
    template<typename T = double>
    class LogarithmicModulus {
    public:
        /**
         * @brief Construct a logarithmic modulus.
         * @param C Coefficient (multiplier).
         * @param p Exponent in the denominator.
         */
        LogarithmicModulus(T C, T p) : C_(C), p_(p) {}

        /**
         * @brief Evaluate the modulus at a given δ.
         * @param delta Step size (must be > 0; if ≤ 0, returns infinity).
         * @return C / |ln δ|^p.
         */
        T operator()(T delta) const {
            if (delta <= 0) return std::numeric_limits<T>::infinity();
            using std::log;
            using std::pow;
            return C_ / pow(abs(log(delta)), p_);
        }

    private:
        T C_, p_;
    };

    // A Rational specialisation of LogarithmicModulus can be added if needed.
    // TOTALLY IS NEEDED. AFTER TESTS SETTLE, LEAVE FULLY DOUBLE SPEC AS LEGACY REDUNDANT CODE FOR GOOD MEASURE
    // BUT IMPLEMENT RATIONAL SPEC AND USE IT WHERE NEEDED INSTEAD OF PETTY DOUBLE.

} // namespace delta::calculus