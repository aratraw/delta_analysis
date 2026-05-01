// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/geometry/matrix_field.h
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
     * @tparam Addr    address type
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

        template<typename Grid>
        explicit MatrixField(const Grid& grid, const value_type& init_val = value_type{})
            : Base(grid, init_val) {
        }

        // -------------------------------------------------------------------------
        // Matrix arithmetic (pointwise)
        // -------------------------------------------------------------------------
        MatrixField operator*(const MatrixField& other) const;
        MatrixField& operator*=(const MatrixField& other);   // in‑place
        MatrixField transpose() const;
        TensorField<Addr, Scalar, 0, Dim, Compare> determinant() const;
        MatrixField comm(const MatrixField& other) const;

        // -------------------------------------------------------------------------
        // Matrix exponential and logarithm (constructive, with precision control)
        // -------------------------------------------------------------------------
        MatrixField exp(const Scalar& eps = delta::default_eps()) const;
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
    // Helper to determine Padé order based on eps
    template<typename Scalar>
    static int pade_order(const Scalar& eps) {
        double eps_d = eps.to_double();
        if (eps_d <= 0) return 16;  // maximum reasonable order

        // Estimate: Padé (m,m) gives error ~ (m!)^2 / (2m)! / (2m+1)! * (||A||)^{2m+1}
        // With ||A|| <= 0.5, for eps = 1e-6 need m = 6
        // For eps = 1e-30 need m = 16

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
        // eps уже является значением по умолчанию или переданным пользователем.
        // Никакой дополнительной проверки на 0 не требуется — это ответственность
        // пользователя.
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
        // eps уже является значением по умолчанию или переданным пользователем.
        // Никакой дополнительной проверки на 0 не требуется — это ответственность
        // пользователя.
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