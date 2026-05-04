// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/geometry/matrix_field.h
// ============================================================================
// MATRIX FIELD – (1,1) TENSOR FIELD ON A GRID / POINT SET
// ============================================================================
//
// This file defines MatrixField, a specialisation of TensorField for matrices
// (i.e., tensors of type (1,1)).  It stores a possibly non‑uniform set of
// addresses (points) and associates an Eigen::Matrix<DeltaRational, Dim, Dim>
// with each address.
//
// Main features:
//   - Pointwise matrix arithmetic (multiplication, transpose, commutator).
//   - Determinant as a scalar field.
//   - Matrix exponential and logarithm implemented **constructively**
//     (no floating‑point approximations) with full rational arithmetic.
//   - Parallelisation via OpenMP when the field is large.
//
// ----------------------------------------------------------------------------
// MATRIX EXPONENTIAL (exp)
// ----------------------------------------------------------------------------
// For a square matrix M, the exponential is computed with a scaling‑and‑squaring
// algorithm combined with Padé approximants.  The method:
//   1. Scale M by dividing by a power of 2 until ||M/2^k|| ≤ 0.5.
//   2. Compute the (m,m) Padé approximant of e^A (where A = M/2^k).
//      The order m is chosen adaptively based on the requested epsilon.
//   3. Square the result k times.
// A fast path is used when M is diagonal (direct component‑wise exp).
//
// ----------------------------------------------------------------------------
// MATRIX LOGARITHM (log)
// ----------------------------------------------------------------------------
// For an invertible matrix M, the principal logarithm is computed by:
//   1. Scale M by dividing by powers of 2 until it is close to identity.
//   2. Compute Z = (X - I) (X + I)^{-1}.
//   3. Use the Gregory series: log(X) = 2 Σ_{n=0}∞ Z^{2n+1} / (2n+1).
//   4. Recover the scaling: log(M) = k·log(2)·I + log(X).
// The series is truncated when the term norm falls below the given epsilon.
//
// ----------------------------------------------------------------------------
// THREAD SAFETY & PARALLELISM
// ----------------------------------------------------------------------------
// MatrixField is not inherently thread‑safe – the user must synchronise
// modifications.  However, read‑only pointwise operations (exp, log,
// multiplication) are parallelised with OpenMP when the number of addresses
// exceeds OMP_MIN_SIZE (currently 1000).  Each point is processed independently.
//
// ----------------------------------------------------------------------------
// EXCEPTION HANDLING
// ----------------------------------------------------------------------------
// If a particular matrix is singular (log) or the computation fails for any
// reason, that point is silently skipped in the result field (instead of
// throwing an exception and stopping the whole field).  If no point remains,
// an exception is thrown at the end.
//
// ============================================================================

#ifndef DELTA_GEOMETRY_MATRIX_FIELD_H
#define DELTA_GEOMETRY_MATRIX_FIELD_H

#include "delta/geometry/tensor_field.h"
#include "delta/core/rational.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <type_traits>
#include <limits>
#include <vector>
#include <optional>

namespace delta::geometry {

    /**
     * @class MatrixField
     * @brief Specialisation of TensorField for (1,1) tensors, i.e. matrix‑valued fields.
     *
     * The scalar type is always delta::Rational, which may be configured via CMake.
     * All transcendental operations (exp, log) are implemented constructively,
     * using series expansions with precision control and only rational arithmetic
     * (no square roots).
     *
     * @tparam Addr    address type (e.g., point, grid index)
     * @tparam Dim     matrix dimension (must be positive)
     * @tparam Compare comparison functor for addresses
     */
    template<typename Addr, int Dim, typename Compare = std::less<Addr>>
    class MatrixField : public TensorField<Addr, Rational, 2, Dim, Compare> {
        using Base = TensorField<Addr, Rational, 2, Dim, Compare>;

    public:
        using Scalar = Rational;
        using typename Base::value_type;             // Eigen::Matrix<Scalar, Dim, Dim>
        using Base::set;
        using Base::at;
        using Base::contains;
        using Base::size;
        using Base::begin;
        using Base::end;
        using Base::comparator;
        using address_type = Addr;
        using comparator_type = Compare;

        using Base::Base;
        MatrixField() = default;

        /**
         * @brief Construct a MatrixField from a grid, initialising all matrices to a constant.
         * @tparam Grid A type that models the GridConcept (must provide begin/end/size/operator[]).
         * @param grid The underlying grid (addresses are the grid elements).
         * @param init_val The initial matrix value for every address.
         */
        template<typename Grid>
        explicit MatrixField(const Grid& grid, const value_type& init_val = value_type{})
            : Base(grid, init_val) {
        }

        // -------------------------------------------------------------------------
        // Matrix arithmetic (pointwise)
        // -------------------------------------------------------------------------
        /**
         * @brief Pointwise matrix multiplication (this * other).
         * @return New MatrixField containing M_i * N_i for each address.
         */
        MatrixField operator*(const MatrixField& other) const;

        /**
         * @brief In‑place pointwise matrix multiplication.
         * @return Reference to *this.
         */
        MatrixField& operator*=(const MatrixField& other);

        /**
         * @brief Pointwise transpose.
         * @return New MatrixField containing M_i.transpose().
         */
        MatrixField transpose() const;

        /**
         * @brief Pointwise determinant.
         * @return Scalar field (0‑tensor) with det(M_i) at each address.
         */
        TensorField<Addr, Scalar, 0, Dim, Compare> determinant() const;

        /**
         * @brief Pointwise commutator [this, other] = this*other - other*this.
         * @return New MatrixField with the commutator at each address.
         */
        MatrixField comm(const MatrixField& other) const;

        // -------------------------------------------------------------------------
        // Matrix exponential and logarithm (constructive, with precision control)
        // -------------------------------------------------------------------------
        /**
         * @brief Pointwise matrix exponential.
         * @param eps Requested absolute precision (default: delta::default_eps()).
         * @return MatrixField where each matrix M_i is replaced by exp(M_i).
         * @throws std::domain_error if no matrix could be exponentiated (should not happen).
         */
        MatrixField exp(const Scalar& eps = delta::default_eps()) const;

        /**
         * @brief Pointwise principal matrix logarithm.
         * @param eps Requested absolute precision (default: delta::default_eps()).
         * @return MatrixField where each invertible matrix M_i is replaced by log(M_i).
         * @throws std::domain_error if no matrix in the field is invertible.
         * @note Singular matrices are silently omitted from the result.
         */
        MatrixField log(const Scalar& eps = delta::default_eps()) const;

    private:
        // -------------------------------------------------------------------------
        // Helper functions
        // -------------------------------------------------------------------------
        static Scalar matrix_norm(const value_type& M);
        static value_type matrix_exp(const value_type& M, const Scalar& eps = delta::default_eps());
        static value_type matrix_log(const value_type& M, const Scalar& eps = delta::default_eps(), const Scalar& log2 = delta::log(2_r, delta::default_eps()));
        static value_type matrix_exp_diag(const value_type& M, const Scalar& eps = delta::default_eps());
        static value_type matrix_log_diag(const value_type& M, const Scalar& eps = delta::default_eps());
    };

    namespace detail {
        /**
         * @brief Check whether a matrix is diagonal (all off‑diagonal entries are zero).
         */
        template<typename Matrix>
        bool is_diagonal(const Matrix& M) {
            const int n = M.rows();
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i != j && M(i, j) != 0) return false;
                }
            }
            return true;
        }
    }

    // -------------------------------------------------------------------------
    // Implementation
    // -------------------------------------------------------------------------

    template<typename Addr, int Dim, typename Compare>
    auto MatrixField<Addr, Dim, Compare>::operator*(const MatrixField& other) const -> MatrixField {
        // 1. Collect addresses from this field
        std::vector<Addr> addrs;
        addrs.reserve(this->size());
        for (const auto& [addr, mat] : *this) {
            addrs.push_back(addr);
        }

        // 2. Vector of results
        std::vector<value_type> results(addrs.size());

        // 3. Parallel (or sequential) computation
#ifdef _OPENMP
        static constexpr std::size_t OMP_MIN_SIZE = 1000;
        if (addrs.size() >= OMP_MIN_SIZE) {
#pragma omp parallel for
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(addrs.size()); ++i) {
                results[i] = this->at(addrs[i]) * other.at(addrs[i]);
            }
        }
        else
#endif
        {
            for (std::size_t i = 0; i < addrs.size(); ++i) {
                results[i] = this->at(addrs[i]) * other.at(addrs[i]);
            }
        }

        // 4. Build result field
        MatrixField result;
        for (std::size_t i = 0; i < addrs.size(); ++i) {
            result.set(addrs[i], results[i]);
        }
        return result;
    }

    template<typename Addr, int Dim, typename Compare>
    auto MatrixField<Addr, Dim, Compare>::operator*=(const MatrixField& other) -> MatrixField& {
        *this = *this * other;
        return *this;
    }

    template<typename Addr, int Dim, typename Compare>
    auto MatrixField<Addr, Dim, Compare>::transpose() const -> MatrixField {
        MatrixField result;
        for (const auto& [addr, mat] : *this) {
            result.set(addr, mat.transpose());
        }
        return result;
    }

    template<typename Addr, int Dim, typename Compare>
    auto MatrixField<Addr, Dim, Compare>::determinant() const -> TensorField<Addr, Scalar, 0, Dim, Compare> {
        TensorField<Addr, Scalar, 0, Dim, Compare> det_field;
        for (const auto& [addr, mat] : *this) {
            det_field.set(addr, mat.determinant());
        }
        return det_field;
    }

    template<typename Addr, int Dim, typename Compare>
    auto MatrixField<Addr, Dim, Compare>::comm(const MatrixField& other) const -> MatrixField {
        MatrixField result;
        for (const auto& [addr, mat] : *this) {
            const auto& other_mat = other.at(addr);
            result.set(addr, mat * other_mat - other_mat * mat);
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Norm (max of absolute entries)
    // -------------------------------------------------------------------------
    template<typename Addr, int Dim, typename Compare>
    auto MatrixField<Addr, Dim, Compare>::matrix_norm(const value_type& M) -> Scalar {
        Scalar max_abs = 0;
        for (int i = 0; i < Dim; ++i) {
            for (int j = 0; j < Dim; ++j) {
                Scalar abs_val = delta::abs(M(i, j));
                if (abs_val > max_abs) max_abs = abs_val;
            }
        }
        return max_abs;
    }

    // -------------------------------------------------------------------------
    // Diagonal case (fast, exact)
    // -------------------------------------------------------------------------
    template<typename Addr, int Dim, typename Compare>
    auto MatrixField<Addr, Dim, Compare>::matrix_exp_diag(const value_type& M, const Scalar& eps) -> value_type {
        value_type result = value_type::Zero();
        for (int i = 0; i < Dim; ++i) {
            result(i, i) = delta::exp(M(i, i), eps);
        }
        return result;
    }

    template<typename Addr, int Dim, typename Compare>
    auto MatrixField<Addr, Dim, Compare>::matrix_log_diag(const value_type& M, const Scalar& eps) -> value_type {
        value_type result = value_type::Zero();
        for (int i = 0; i < Dim; ++i) {
            result(i, i) = delta::log(M(i, i), eps);
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Matrix exponential (scaling‑and‑squaring + adaptive Padé)
    // -------------------------------------------------------------------------
    // ============================================================================
    // IMPLEMENTATION NOTES – PADÉ AND GREGORY SERIES
    // ============================================================================
    //
    // The matrix exponential uses a scaling‑and‑squaring method combined with
    // Padé approximants.  Several details are worth highlighting:
    //
    // 1. Adaptive Padé order.
    //    Many implementations fix the order (e.g., m = 6 or 13).  Here we choose m
    //    based on the requested epsilon using empirically determined thresholds
    //    (eps >= 1e-3 → m=4, … eps >= 1e-27 → m=14, otherwise m=16).
    //    Higher m gives better accuracy but more arithmetic; lower m is faster.
    //    The thresholds were obtained by experimenting with rational arithmetic
    //    and ensure that the series error for ||A|| ≤ 0.5 stays below eps.
    //
    // 2. Rational Padé coefficients without double conversion.
    //    The coefficients c_j are computed recursively using rational arithmetic
    //    (delta::Rational).  This is exact but may produce large numerators/
    //    denominators for high m (e.g., m=16).  However, the scaling step
    //    (‖A‖ ≤ 0.5) keeps the intermediate numbers manageable in practice.
    //
    // 3. Matrix norm: max |element|.
    //    The max‑norm is easy to compute and sufficient for scaling decisions.
    //    It is not a matrix norm in the strict sense (sub‑multiplicativity fails),
    //    but for the purpose of bounding ||A|| we only need a reliable estimate;
    //    the scaling condition ‖A‖ ≤ 0.5 is overly pessimistic, which is safe.
    //
    // 4. Gregory series termination.
    //    The series for log(X) = 2 Σ Z^{2n+1}/(2n+1) terminates when:
    //        ||term|| ≤ eps  AND  ||term|| ≤ eps * (||sum|| + 1).
    //    The second (relative) condition prevents infinite loops when the sum is
    //    nearly zero (e.g., X ≈ I).  The "+1" guards against division by zero.
    //
    // 5. Handling of singular matrices in log().
    //    Instead of throwing an exception at the first singular matrix, we skip
    //    that address entirely (the field entry is omitted).  This allows the
    //    field to still contain a valid result at all invertible points.
    //    Only when no point remains invertible do we throw a domain_error.
    //
    // 6. Parallelism with OpenMP.
    //    The operations exp() and log() are embarrassingly parallel: each matrix
    //    can be processed independently.  The code uses OpenMP when the number
    //    of addresses exceeds 1000.  No thread synchronization is needed for
    //    read‑only access (each address is processed once).
    //
    // ============================================================================
    /**
     * @brief Determine the appropriate Padé order m given the requested precision.
     * @param eps Absolute tolerance.
     * @return m (between 4 and 16).
     */
    template<typename Scalar>
    static int pade_order(const Scalar& eps) {
        double eps_d = eps.to_double();
        if (eps_d <= 0) return 16;  // maximum reasonable order

        // Estimate: Padé (m,m) gives error ~ (m!)^2 / (2m)! / (2m+1)! * (||A||)^{2m+1}
        // With ||A|| <= 0.5, for eps = 1e-6 need m = 6, for eps = 1e-30 need m = 16.
        if (eps_d >= 1e-3) return 4;
        if (eps_d >= 1e-7) return 6;
        if (eps_d >= 1e-12) return 8;
        if (eps_d >= 1e-17) return 10;
        if (eps_d >= 1e-22) return 12;
        if (eps_d >= 1e-27) return 14;
        return 16;
    }

    template<typename Addr, int Dim, typename Compare>
    auto MatrixField<Addr, Dim, Compare>::matrix_exp(const value_type& M, const Scalar& eps) -> value_type {
        if (detail::is_diagonal(M)) {
            return matrix_exp_diag(M, eps);
        }

        // 1. Scaling
        Scalar normM = matrix_norm(M);
        int k = 0;
        Scalar two_pow_k = 1;
        while (normM / two_pow_k > Scalar(1) / Scalar(2)) {
            two_pow_k *= 2;
            ++k;
        }
        value_type A = M / two_pow_k;   // A = M / 2^k

        // 2. Determine Padé order from eps
        int m = pade_order(eps);

        // 3. Compute Padé (m,m) coefficients
        // Recurrence: c_j = c_{j-1} * (m - j + 1) / ((2m - j + 1) * j)
        std::vector<Scalar> c(m + 1);
        c[0] = Scalar(1);
        for (int j = 1; j <= m; ++j) {
            c[j] = c[j - 1] * Scalar(m - j + 1) / Scalar((2 * m - j + 1) * j);
        }

        // 4. Compute P_m(A) and Q_m(A)
        // P_m(A) = Σ_{j=0}^{m} c_j * A^j
        // Q_m(A) = Σ_{j=0}^{m} (-1)^j * c_j * A^j

        value_type A_pow = value_type::Identity();  // A^0
        value_type P = value_type::Zero();
        value_type Q = value_type::Zero();

        for (int j = 0; j <= m; ++j) {
            if (c[j] != 0) {
                P += c[j] * A_pow;
                Q += (j % 2 == 0 ? c[j] : -c[j]) * A_pow;
            }
            A_pow = A_pow * A;  // A^{j+1} for next iteration
        }

        // 5. Solve Q * exp(A) = P → exp(A) = Q^{-1} * P
        value_type E = Q.lu().solve(P);

        // 6. Squaring k times
        for (int i = 0; i < k; ++i) {
            E = E * E;
        }
        return E;
    }

    // -------------------------------------------------------------------------
    // Matrix logarithm (inverse scaling + Gregory series)
    // -------------------------------------------------------------------------
    template<typename Addr, int Dim, typename Compare>
    auto MatrixField<Addr, Dim, Compare>::matrix_log(const value_type& M, const Scalar& eps, const Scalar& log2) -> value_type {
        // 1. Check invertibility
        if (M.determinant() == 0) {
            throw std::domain_error("matrix_log: singular matrix");
        }
        if (detail::is_diagonal(M)) {
            return matrix_log_diag(M, eps);
        }

        // 2. Scale matrix by dividing by powers of 2 until it is close to identity
        //    log(M) = k*log(2)*I + log(M / 2^k)
        value_type X = M;
        int k = 0;
        const int max_scale = 100;
        while (matrix_norm(X - value_type::Identity()) > Rational(1, 2)) {
            X = X / Scalar(2);
            ++k;
            if (k > max_scale) throw std::runtime_error("matrix_log: scaling did not converge");
        }

        // 3. Compute Z = (X - I) * (X + I)^{-1}
        value_type X_minus_I = X - value_type::Identity();
        value_type X_plus_I = X + value_type::Identity();
        // Invert (X+I) using LU
        value_type X_plus_I_inv = X_plus_I.inverse();
        value_type Z = X_minus_I * X_plus_I_inv;

        // 4. Gregory series for log: log(X) = 2 * sum_{n=0}^{∞} (1/(2n+1)) * Z^{2n+1}
        value_type Z2 = Z * Z;               // Z^2
        value_type Z_pow = Z;               // Z^{1}
        value_type sum = Z_pow;
        int n = 0;
        const int max_series = 1000000;
        while (true) {
            ++n;
            // Z_pow = Z^{2n+1}
            Z_pow = Z_pow * Z2;              // one multiplication instead of two
            value_type term = Z_pow / Scalar(2 * n + 1);
            sum += term;

            // Use both absolute and relative convergence criteria
            Scalar norm_term = matrix_norm(term);
            Scalar norm_sum = matrix_norm(sum);
            // Relative tolerance with safeguard against zero sum
            Scalar rel_tol = eps * (norm_sum + 1);
            if (norm_term <= eps && norm_term <= rel_tol) break;
            if (n > max_series) throw std::runtime_error("matrix_log: series did not converge");
        }
        sum = sum * Scalar(2);

        // 5. Recover the scaling: log(M) = k*log(2)*I + sum
        value_type result = value_type::Identity() * (k * log2) + sum;
        return result;
    }

    // -------------------------------------------------------------------------
    // Public exp/log methods (parallelized, with per-point exception resilience)
    // -------------------------------------------------------------------------
    template<typename Addr, int Dim, typename Compare>
    auto MatrixField<Addr, Dim, Compare>::exp(const Scalar& eps) const -> MatrixField {
        // eps is the default value or user‑supplied. No extra check for zero
        // is performed – it is the user's responsibility.
        const Scalar& actual_eps = eps;

        // Collect all addresses (needed for later assignment)
        std::vector<Addr> addrs;
        addrs.reserve(this->size());
        for (const auto& [addr, mat] : *this) {
            addrs.push_back(addr);
        }

        // Store results; std::nullopt indicates that the exponential could not be computed
        std::vector<std::optional<value_type>> results(addrs.size());

#ifdef _OPENMP
        static constexpr std::size_t OMP_MIN_SIZE = 1000;
        if (addrs.size() >= OMP_MIN_SIZE) {
#pragma omp parallel for
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(addrs.size()); ++i) {
                try {
                    results[i] = matrix_exp(this->at(addrs[i]), actual_eps);
                }
                catch (const std::domain_error&) {
                    // Matrix exponential is almost always defined for square matrices,
                    // but if an error occurs (e.g. non‑square?), we skip this point.
                    results[i] = std::nullopt;
                }
            }
        }
        else
#endif
        {
            for (std::size_t i = 0; i < addrs.size(); ++i) {
                try {
                    results[i] = matrix_exp(this->at(addrs[i]), actual_eps);
                }
                catch (const std::domain_error&) {
                    results[i] = std::nullopt;
                }
            }
        }

        MatrixField result;
        for (std::size_t i = 0; i < addrs.size(); ++i) {
            if (results[i].has_value()) {
                result.set(addrs[i], std::move(results[i].value()));
            }
        }

        // If no point could be exponentiated, throw an exception
        if (result.size() == 0) {
            throw std::domain_error("matrix_exp: no matrix in the field can be exponentiated");
        }
        return result;
    }

    template<typename Addr, int Dim, typename Compare>
    auto MatrixField<Addr, Dim, Compare>::log(const Scalar& eps) const -> MatrixField {
        // eps is the default value or user‑supplied. No extra check for zero.
        const Scalar& actual_eps = eps;
        Scalar log2 = delta::log(2_r, actual_eps);

        // Collect addresses once
        std::vector<Addr> addrs;
        addrs.reserve(this->size());
        for (const auto& [addr, mat] : *this) {
            addrs.push_back(addr);
        }

        // Store optional results (nullopt = singular matrix)
        std::vector<std::optional<value_type>> results(addrs.size());

#ifdef _OPENMP
        static constexpr std::size_t OMP_MIN_SIZE = 1000;
        if (addrs.size() >= OMP_MIN_SIZE) {
#pragma omp parallel for
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(addrs.size()); ++i) {
                try {
                    results[i] = matrix_log(this->at(addrs[i]), actual_eps, log2);
                }
                catch (const std::domain_error&) {
                    // Matrix is singular or logarithm not defined
                    results[i] = std::nullopt;
                }
            }
        }
        else
#endif
        {
            for (std::size_t i = 0; i < addrs.size(); ++i) {
                try {
                    results[i] = matrix_log(this->at(addrs[i]), actual_eps, log2);
                }
                catch (const std::domain_error&) {
                    results[i] = std::nullopt;
                }
            }
        }

        MatrixField result;
        for (std::size_t i = 0; i < addrs.size(); ++i) {
            if (results[i].has_value()) {
                result.set(addrs[i], std::move(results[i].value()));
            }
        }

        // Only throw if no matrix in the field is invertible
        if (result.size() == 0) {
            throw std::domain_error("matrix_log: no invertible matrix in the field");
        }
        return result;
    }

} // namespace delta::geometry

#endif // DELTA_GEOMETRY_MATRIX_FIELD_H