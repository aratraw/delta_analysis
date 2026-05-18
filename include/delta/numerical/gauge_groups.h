// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/numerical/gauge_groups.h
// ============================================================================
// CALIBRATIONAL GROUPS – U(1), SU(2), SU(3)
// ============================================================================
//
// Provides template classes for the gauge groups U(1) (as SO(2)), SU(2) and
// SU(3).  For Scalar = Rational the SU(2) and SU(3) groups use GaussQi as the
// complex number type, and the exponential / logarithm are implemented via the
// general matrix functions delta::exp / delta::log (which employ trace
// normalisation for the complex case).
//
// ============================================================================

#pragma once

#include <Eigen/Dense>
#include <complex>
#include "delta/core/rational.h"
#include "delta/rational/gauss_qi.h"             // GaussQi
#include "delta/rational/transcendentals.h"      // delta::cos, sin, acos, exp, log, sinpi, cospi
#include "delta/core/eigen_integration.h"        // matrix exp/log for GaussQi

namespace delta::numerical {

    // =========================================================================
    // U(1) = SO(2)
    // =========================================================================
    template<typename Scalar>
    struct U1 {
        using ScalarType = Scalar;
        using matrix_type = Eigen::Matrix<Scalar, 2, 2>;
        using algebra_type = Eigen::Matrix<Scalar, 2, 2>;   // skew‑symmetric
        static constexpr int N = 1;

        static matrix_type identity() {
            return matrix_type::Identity();
        }

        // used to compute Wilson Action, needs to be Rational - metric scalar type.
        static Rational real_trace(const matrix_type& U) {
            return U.trace();
        }

        // Exponential of a skew‑symmetric matrix [[0, -θ],[θ, 0]]
        static matrix_type exp(const algebra_type& A,
            const Scalar& eps = delta::default_eps()) {
            Scalar theta = A(1, 0);   // the (1,0) entry is θ
            Scalar c = delta::cos(theta, eps);
            Scalar s = delta::sin(theta, eps);
            matrix_type R;
            R << c, -s,
                s, c;
            return R;
        }

        // Principal logarithm of a rotation matrix
        static algebra_type log(const matrix_type& U,
            const Scalar& eps = delta::default_eps()) {
            Scalar c = U(0, 0);
            Scalar s = U(1, 0);
            Scalar theta = delta::acos(c, eps);
            if (s < 0) theta = -theta;
            algebra_type A;
            A << 0, -theta,
                theta, 0;
            return A;
        }

        // Extract angle θ ∈ [0, 2π) from a rotation matrix.
        static Scalar to_angle(const matrix_type& U, const Scalar& eps = delta::default_eps()) {
            Scalar c = U(0, 0);
            Scalar s = U(1, 0);
            return delta::atan2(s, c, eps);
        }

        // Construct rotation matrix from angle = θ·π (θ in units of π)
        static matrix_type from_angle_pi(const Scalar& theta_pi,
            const Scalar& eps = delta::default_eps()) {
            Scalar c = delta::cospi(theta_pi, eps);
            Scalar s = delta::sinpi(theta_pi, eps);
            matrix_type R;
            R << c, -s,
                s, c;
            return R;
        }

        // Extract angle in units of π: φ = angle/π ∈ [0,2)
        static Scalar to_angle_pi(const matrix_type& U,
            const Scalar& eps = delta::default_eps()) {
            Scalar theta = to_angle(U, eps);
            return theta / delta::pi(eps);
        }
    };

    // =========================================================================
    // SU(2)
    // =========================================================================
    template<typename Scalar>
    struct SU2;

    template<>
    struct SU2<Rational> {
        using ScalarType = GaussQi;
        using matrix_type = Eigen::Matrix<GaussQi, 2, 2>;
        using algebra_type = Eigen::Matrix<GaussQi, 2, 2>;   // i * Hermitian traceless
        static constexpr int N = 2;

        static matrix_type identity() {
            return matrix_type::Identity();
        }

        // used to compute Wilson Action, needs to be Rational - metric scalar type.
        static Rational real_trace(const matrix_type& U) {
            return U.trace().real();
        }

        static matrix_type exp(const algebra_type& A,
            const Rational& eps = delta::default_eps()) {
            return delta::exp(A, eps);
        }

        static algebra_type log(const matrix_type& U,
            const Rational& eps = delta::default_eps()) {
            return delta::log(U, eps);
        }
    };

    // For double (if ever needed)
    template<>
    struct SU2<double> {
        using ScalarType = std::complex<double>;
        using matrix_type = Eigen::Matrix<std::complex<double>, 2, 2>;
        using algebra_type = Eigen::Matrix<std::complex<double>, 2, 2>;
        static constexpr int N = 2;
        static matrix_type identity() { return matrix_type::Identity(); }
        // ... trivial wrappers if required
    };

    // =========================================================================
    // SU(3)
    // =========================================================================
    template<typename Scalar>
    struct SU3;

    template<>
    struct SU3<Rational> {
        using ScalarType = GaussQi;
        using matrix_type = Eigen::Matrix<GaussQi, 3, 3>;
        using algebra_type = Eigen::Matrix<GaussQi, 3, 3>;   // i * Hermitian traceless
        static constexpr int N = 3;

        static matrix_type identity() {
            return matrix_type::Identity();
        }

        // used to compute Wilson Action, needs to be Rational - metric scalar type.
        static Rational real_trace(const matrix_type& U) {
            return U.trace().real();
        }

        static matrix_type exp(const algebra_type& A,
            const Rational& eps = delta::default_eps()) {
            return delta::exp(A, eps);
        }

        static algebra_type log(const matrix_type& U,
            const Rational& eps = delta::default_eps()) {
            return delta::log(U, eps);
        }
    };

} // namespace delta::numerical