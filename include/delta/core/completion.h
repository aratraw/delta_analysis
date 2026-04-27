// include/delta/core/completion.h
#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <type_traits>
#include "rational.h"

namespace delta {

    // -------------------------------------------------------------------------
    // Convergence modulus concept and predefined moduli for sequences
    // -------------------------------------------------------------------------

    /**
     * @concept ConvergenceModulus
     * @brief A modulus of convergence for fundamental sequences.
     *
     * A convergence modulus is a function m(n) that returns an upper bound
     * on the error at level n, i.e., |x_n - x| ≤ m(n) for the limit x.
     * For a fundamental sequence, we require that m(n) → 0 as n → ∞.
     *
     * @tparam M The modulus type.
     * The expression M::value_type must be a scalar type (typically Rational).
     * The expression m(n) for std::size_t n must return a value convertible to that scalar.
     */
    template<typename M>
    concept ConvergenceModulus = requires(M m, std::size_t n) {
        typename M::value_type;
        { m(n) } -> std::convertible_to<typename M::value_type>;
    };

    /**
     * @class ExponentialModulus
     * @brief Exponential decay modulus: error ≤ C * r^n, with 0 < r < 1.
     */
    class ExponentialModulus {
    public:
        using value_type = Rational;

        ExponentialModulus(Rational C, Rational r) : C_(std::move(C)), r_(std::move(r)) {
            if (r_ <= 0 || r_ >= 1) {
                throw std::invalid_argument("ExponentialModulus: rate r must be in (0,1)");
            }
        }

        Rational operator()(std::size_t n) const {
            Rational result = C_;
            for (std::size_t i = 0; i < n; ++i) result *= r_;
            return result;
        }

        const Rational& C() const { return C_; }
        const Rational& r() const { return r_; }

    private:
        Rational C_, r_;
    };

    /**
     * @class PowerDecayModulus
     * @brief Power‑law decay modulus: error ≤ C * n^{-α}, with α > 0.
     */
    class PowerDecayModulus {
    public:
        using value_type = Rational;

        PowerDecayModulus(Rational C, Rational alpha) : C_(std::move(C)), alpha_(std::move(alpha)) {
            if (alpha_ <= 0) {
                throw std::invalid_argument("PowerDecayModulus: exponent alpha must be positive");
            }
        }

        /**
         * @brief Evaluate the modulus at level n.
         * @param n Level index (must be ≥ 1 for meaningful results).
         * @return C * n^{-alpha} as Rational.
         */
        Rational operator()(std::size_t n) const {
            if (n == 0) return C_; // fallback, but n should be >= start_level > 0
            // Compute n^{-alpha} = 1 / n^{alpha} exactly using rational arithmetic
            Rational n_rational(static_cast<long long>(n));
            Rational n_pow_alpha = delta::pow(n_rational, alpha_);
            return C_ / n_pow_alpha;
        }

        const Rational& C() const { return C_; }
        const Rational& alpha() const { return alpha_; }

    private:
        Rational C_, alpha_;
    };

    // -------------------------------------------------------------------------
    // FundamentalSequence – now templated on modulus
    // -------------------------------------------------------------------------

    /**
     * @class FundamentalSequence
     * @brief A fundamental (Cauchy) sequence with a given convergence modulus.
     *
     * The sequence {x_n} is defined for n ≥ start_level and satisfies
     * |x_m - x_n| ≤ modulus(min(m,n)) for all m,n.
     *
     * @tparam Modulus A type satisfying ConvergenceModulus (default ExponentialModulus).
     */
    template<ConvergenceModulus Modulus = ExponentialModulus>
    class FundamentalSequence {
    public:
        using value_type = typename Modulus::value_type;
        using modulus_type = Modulus;

        /**
         * @brief Construct from generator and modulus.
         *
         * @param generator   Function x_n = f(n) for n ≥ start_level.
         * @param modulus     Convergence modulus object.
         * @param start_level First defined level.
         */
        FundamentalSequence(std::function<value_type(std::size_t)> generator,
            Modulus modulus, std::size_t start_level = 0)
            : gen_(std::move(generator)), modulus_(std::move(modulus)), start_(start_level) {
        }

        /**
         * @brief Construct for exponential decay (backward compatibility).
         *
         * @param generator   Generator.
         * @param C           Constant factor.
         * @param r           Rate (0<r<1).
         * @param start_level Start level.
         */
        FundamentalSequence(std::function<value_type(std::size_t)> generator,
            Rational C, Rational r, std::size_t start_level = 0)
            : gen_(std::move(generator)), modulus_(ExponentialModulus(std::move(C), std::move(r))), start_(start_level) {
        }

        /**
         * @brief Access the element at level n.
         *
         * @param n Level (must be ≥ start_level).
         * @return x_n.
         * @throws std::out_of_range if n < start_level.
         */
        value_type operator()(std::size_t n) const {
            if (n < start_) {
                throw std::out_of_range("Level " + std::to_string(n) + " below start level " + std::to_string(start_));
            }
            return gen_(n);
        }

        /// Returns the convergence modulus.
        const Modulus& modulus() const { return modulus_; }

        /// Returns the first defined level.
        std::size_t start_level() const { return start_; }

        // For backward compatibility with old code that expects bound() and rate()
        // These are only available if Modulus is ExponentialModulus.
        template<typename M = Modulus>
        auto bound() const -> std::enable_if_t<std::is_same_v<M, ExponentialModulus>, Rational> {
            return modulus_.C();
        }

        template<typename M = Modulus>
        auto rate() const -> std::enable_if_t<std::is_same_v<M, ExponentialModulus>, Rational> {
            return modulus_.r();
        }

    private:
        std::function<value_type(std::size_t)> gen_;
        Modulus modulus_;
        std::size_t start_;
    };

    // -------------------------------------------------------------------------
    // Equivalence testing for fundamental sequences (now with modulus awareness)
    // -------------------------------------------------------------------------

    /**
     * @brief Check whether two fundamental sequences are equivalent.
     *
     * Two sequences are equivalent if there exists a constant K > 0 such that
     * |x_n - y_n| ≤ K * (modulus1(n) + modulus2(n)) for all n.
     * This function estimates K and verifies the condition for a range of levels.
     *
     * @param seq1 First sequence.
     * @param seq2 Second sequence.
     * @param K    Output estimated constant.
     * @return true if sequences appear equivalent.
     */
    template<ConvergenceModulus M1, ConvergenceModulus M2>
    static inline bool are_equivalent(const FundamentalSequence<M1>& seq1,
        const FundamentalSequence<M2>& seq2,
        Rational& K) {
        std::size_t start = std::max(seq1.start_level(), seq2.start_level());
        const std::size_t N = 20; // number of levels to estimate K

        Rational maxK = 0;
        for (std::size_t i = 0; i < N; ++i) {
            std::size_t n = start + i;
            Rational diff = seq1(n) - seq2(n);
            if (diff < 0) diff = -diff;
            Rational total_err = seq1.modulus()(n) + seq2.modulus()(n);
            if (total_err == 0) {
                // Exact sequences: diff must be zero for equivalence
                if (diff != 0) return false;
                continue;
            }
            Rational Ki = diff / total_err;
            if (Ki > maxK) maxK = Ki;
        }
        K = maxK;

        // Verify for next N levels
        for (std::size_t i = N; i < 2 * N; ++i) {
            std::size_t n = start + i;
            Rational diff = seq1(n) - seq2(n);
            if (diff < 0) diff = -diff;
            Rational total_err = seq1.modulus()(n) + seq2.modulus()(n);
            // Allow a tiny tolerance to account for rounding.
            if (diff > K * total_err + Rational(1, 1000000)) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Simplified equivalence test (discards K).
     */
    template<ConvergenceModulus M1, ConvergenceModulus M2>
    static inline bool are_equivalent(const FundamentalSequence<M1>& seq1,
        const FundamentalSequence<M2>& seq2) {
        Rational K;
        return are_equivalent(seq1, seq2, K);
    }

    // -------------------------------------------------------------------------
    // RealNumber – kept as originally (only works with exponential sequences)
    // to avoid massive refactoring. It uses FundamentalSequence<ExponentialModulus>.
    // -------------------------------------------------------------------------

    /**
     * @class RealNumber
     * @brief A real number represented as an equivalence class of fundamental sequences.
     *
     * This class demonstrates the completion of rationals to reals.
     * It provides equality via sequence equivalence and approximate comparison
     * with a given tolerance.
     *
     * @note This version only works with exponential sequences (ExponentialModulus)
     *       for simplicity. For general moduli, a type‑erased wrapper would be needed.
     */
    class RealNumber {
    public:
        using value_type = Rational;

        /**
         * @brief Construct a real number from a rational (constant sequence).
         *
         * @param q The rational value.
         */
        explicit RealNumber(value_type q)
            : seq_(std::make_shared<FundamentalSequence<ExponentialModulus>>(
                [q](std::size_t) { return q; }, Rational(0), Rational(1, 2), 0)) {
        }

        /**
         * @brief Construct a real number from an arbitrary exponential fundamental sequence.
         *
         * @param seq Shared pointer to the sequence (must be non‑null).
         */
        explicit RealNumber(std::shared_ptr<FundamentalSequence<ExponentialModulus>> seq)
            : seq_(std::move(seq)) {
        }

        /**
         * @brief Obtain an approximation at a given level.
         *
         * @param n Level (must be ≥ sequence's start_level).
         * @return The element x_n of the underlying sequence.
         */
        value_type approximate(std::size_t n) const {
            return (*seq_)(n);
        }

        /**
         * @brief Equality of real numbers (via equivalence of their sequences).
         */
        bool operator==(const RealNumber& other) const {
            return are_equivalent(*seq_, *other.seq_);
        }

        /**
         * @brief Approximate equality within a given tolerance.
         *
         * Finds a level n such that the theoretical error bound of both sequences
         * is ≤ eps, then checks that the actual difference at that level does not
         * exceed that bound. Returns false if such a level cannot be found within
         * a reasonable number of iterations.
         *
         * @param other The other real number.
         * @param eps   Allowed absolute error.
         * @return true if the numbers are approximately equal.
         */
        bool approx_equal(const RealNumber& other, const Rational& eps) const {
            const Rational& C1 = seq_->bound();
            const Rational& r1 = seq_->rate();
            const Rational& C2 = other.seq_->bound();
            const Rational& r2 = other.seq_->rate();
            std::size_t n = std::max(seq_->start_level(), other.seq_->start_level());
            const int max_iter = 100;
            for (int iter = 0; iter < max_iter; ++iter, ++n) {
                Rational err1 = C1;
                for (std::size_t i = 0; i < n - seq_->start_level(); ++i) err1 = err1 * r1;
                Rational err2 = C2;
                for (std::size_t i = 0; i < n - other.seq_->start_level(); ++i) err2 = err2 * r2;
                Rational total_err = err1 + err2;
                if (total_err <= eps) {
                    Rational diff = approximate(n) - other.approximate(n);
                    if (diff < 0) diff = -diff;
                    // Если погрешность нулевая (точные числа), сравниваем с eps
                    if (total_err == 0) {
                        return diff <= eps;
                    }
                    return diff <= total_err;
                }
            }
            return false;
        }

    private:
        std::shared_ptr<FundamentalSequence<ExponentialModulus>> seq_; ///< Underlying exponential sequence.
    };

    // -------------------------------------------------------------------------
    // For backward compatibility, we keep a typedef for the old name
    // -------------------------------------------------------------------------
    using FundamentalSequenceExponential = FundamentalSequence<ExponentialModulus>;

} // namespace delta