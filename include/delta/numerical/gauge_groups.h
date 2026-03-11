#pragma once

#include <complex>
#include <Eigen/Dense>

namespace delta::numerical {

    // U(1) group
    class U1 {
    public:
        using value_type = std::complex<double>;
        static constexpr int dimension = 1;

        U1() : phase_(1.0) {}
        explicit U1(double theta) : phase_(std::exp(std::complex<double>(0, theta))) {}
        explicit U1(const std::complex<double>& z) : phase_(z / std::abs(z)) {} // normalize

        static U1 identity() { return U1(0.0); }

        U1 inverse() const { return U1(std::conj(phase_)); }

        U1 operator*(const U1& other) const { return U1(phase_ * other.phase_); }

        double trace() const { return std::real(phase_); }

        // For Wilson action: return real part of trace
        double real_tr() const { return std::real(phase_); }

    private:
        std::complex<double> phase_;
    };

    // SU(2) group
    class SU2 {
    public:
        using value_type = double;
        static constexpr int dimension = 2;

        SU2() : mat_(Eigen::Matrix2cd::Identity()) {}
        explicit SU2(const Eigen::Matrix2cd& mat) : mat_(mat) { normalize(); }

        static SU2 identity() { return SU2(); }

        SU2 inverse() const { return SU2(mat_.adjoint()); }

        SU2 operator*(const SU2& other) const { return SU2(mat_ * other.mat_); }

        double trace() const { return std::real(mat_.trace()); }

        double real_tr() const { return std::real(mat_.trace()); }

        const Eigen::Matrix2cd& matrix() const { return mat_; }

    private:
        Eigen::Matrix2cd mat_;

        void normalize() {
            // Ensure determinant = 1
            std::complex<double> det = mat_.determinant();
            mat_ /= std::sqrt(det);
        }
    };

    // SU(3) group – можно добавить позже при необходимости

} // namespace delta::numerical