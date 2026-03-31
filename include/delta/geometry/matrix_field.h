// include/delta/geometry/matrix_field.h
#ifndef DELTA_GEOMETRY_MATRIX_FIELD_H
#define DELTA_GEOMETRY_MATRIX_FIELD_H

#include "delta/geometry/tensor_field.h"
#include "delta/core/rational.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <type_traits>
#include <limits>

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
        MatrixField exp(const Scalar& eps = Scalar(0)) const;
        MatrixField log(const Scalar& eps = Scalar(0)) const;

    private:
        // -------------------------------------------------------------------------
        // Helper functions
        // -------------------------------------------------------------------------
        static Scalar matrix_norm(const value_type& M);
        static double matrix_norm_double(const value_type& M);
        static value_type matrix_exp(const value_type& M, const Scalar& eps);
        static value_type matrix_log(const value_type& M, const Scalar& eps, const Scalar& log2);
        static value_type matrix_exp_diag(const value_type& M, const Scalar& eps);
        static value_type matrix_log_diag(const value_type& M, const Scalar& eps);
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
        MatrixField result;
        for (const auto& [addr, mat] : *this) {
            result.set(addr, mat * other.at(addr));
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

    template<typename Addr, int Dim, typename Compare>
    double MatrixField<Addr, Dim, Compare>::matrix_norm_double(const value_type& M) {
        double max_abs = 0.0;
        for (int i = 0; i < Dim; ++i) {
            for (int j = 0; j < Dim; ++j) {
                double val = M(i, j).convert_to<double>();
                double abs_val = val < 0 ? -val : val;
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
    // Matrix exponential (scaling‑and‑squaring + Padé(6,6))
    // -------------------------------------------------------------------------
    template<typename Addr, int Dim, typename Compare>
    auto MatrixField<Addr, Dim, Compare>::matrix_exp(const value_type& M, const Scalar& eps) -> value_type {
        if (detail::is_diagonal(M)) {
            return matrix_exp_diag(M, eps);
        }
        // 1. Scaling to reduce norm
        Scalar normM = matrix_norm(M);
        int k = 0;
        Scalar two_pow_k = 1;
        while (normM / two_pow_k > Rational(1, 2)) {
            two_pow_k *= 2;
            ++k;
        }
        value_type A = M / two_pow_k;   // A = M / 2^k

        // 2. Padé(6,6) approximation of exp(A)
        const Scalar b[] = {
            Scalar(1) / Scalar(2),      // b1 = 1/2
            Scalar(1) / Scalar(12),     // b2 = 1/12
            Scalar(1) / Scalar(120),    // b3 = 1/120
            Scalar(1) / Scalar(1680),   // b4 = 1/1680
            Scalar(1) / Scalar(30240),  // b5 = 1/30240
            Scalar(1) / Scalar(665280)  // b6 = 1/665280
        };
        // Precompute powers
        value_type A2 = A * A;
        value_type A3 = A2 * A;
        value_type A4 = A3 * A;
        value_type A5 = A4 * A;
        value_type A6 = A5 * A;

        // Numerator P = I + b1*A + b2*A^2 + ... + b6*A^6
        value_type P = value_type::Identity();
        P += b[0] * A + b[1] * A2 + b[2] * A3 + b[3] * A4 + b[4] * A5 + b[5] * A6;

        // Denominator Q = I - b1*A + b2*A^2 - b3*A^3 + ... + b6*A^6
        value_type Q = value_type::Identity();
        Q -= b[0] * A;
        Q += b[1] * A2;
        Q -= b[2] * A3;
        Q += b[3] * A4;
        Q -= b[4] * A5;
        Q += b[5] * A6;

        // Solve Q * exp(A) = P  => exp(A) = Q^{-1} * P
        value_type E = Q.lu().solve(P);

        // 3. Squaring k times
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
            // Fast convergence check using double
            double norm_term = matrix_norm_double(term);
            if (norm_term <= eps.convert_to<double>()) break;
            if (n > max_series) throw std::runtime_error("matrix_log: series did not converge");
        }
        sum = sum * Scalar(2);

        // 5. Recover the scaling: log(M) = k*log(2)*I + sum
        value_type result = value_type::Identity() * (k * log2) + sum;
        return result;
    }

    // -------------------------------------------------------------------------
    // Public exp/log methods
    // -------------------------------------------------------------------------
    template<typename Addr, int Dim, typename Compare>
    auto MatrixField<Addr, Dim, Compare>::exp(const Scalar& eps) const -> MatrixField {
        Scalar actual_eps = (eps == 0) ? delta::default_eps() : eps;
        MatrixField result;
        for (const auto& [addr, mat] : *this) {
            result.set(addr, matrix_exp(mat, actual_eps));
        }
        return result;
    }

    template<typename Addr, int Dim, typename Compare>
    auto MatrixField<Addr, Dim, Compare>::log(const Scalar& eps) const -> MatrixField {
        Scalar actual_eps = (eps == 0) ? delta::default_eps() : eps;
        // Compute log2 once for the whole field
        Scalar log2 = delta::log(2_r, actual_eps);
        MatrixField result;
        for (const auto& [addr, mat] : *this) {
            result.set(addr, matrix_log(mat, actual_eps, log2));
        }
        return result;
    }

} // namespace delta::geometry

#endif // DELTA_GEOMETRY_MATRIX_FIELD_H