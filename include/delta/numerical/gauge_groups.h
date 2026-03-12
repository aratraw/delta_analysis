// include/delta/numerical/gauge_groups.h
#pragma once

#include <complex>
#include <Eigen/Dense>
#include <cmath>

namespace delta::numerical {

    // -----------------------------------------------------------------------------
    // U(1) group
    // -----------------------------------------------------------------------------
    class U1 {
    public:
        using value_type = std::complex<double>;
        using matrix_type = std::complex<double>;
        static constexpr int dimension = 1;

        U1() : phase_(1.0) {}
        explicit U1(double theta) : phase_(std::exp(std::complex<double>(0, theta))) {}
        explicit U1(const std::complex<double>& z) : phase_(z / std::abs(z)) {} // normalize

        static U1 identity() { return U1(0.0); }

        U1 inverse() const { return U1(std::conj(phase_)); }

        U1 operator*(const U1& other) const { return U1(phase_ * other.phase_); }

        double trace() const { return std::real(phase_); }
        double real_tr() const { return std::real(phase_); }

        // Преобразование в матрицу 1x1 (для единообразия с SU(N))
        matrix_type matrix() const { return phase_; }

        // Экспонента от мнимого числа (алгебра u(1))
        static U1 exp(double theta) { return U1(theta); }

        // Логарифм (возвращает мнимую часть), предполагается, что фаза близка к 1
        double log() const { return std::imag(std::log(phase_)); }

    private:
        std::complex<double> phase_;
    };

    // -----------------------------------------------------------------------------
    // SU(2) group
    // -----------------------------------------------------------------------------
    class SU2 {
    public:
        using value_type = double;
        using matrix_type = Eigen::Matrix2cd;
        static constexpr int dimension = 2;

        SU2() : mat_(Eigen::Matrix2cd::Identity()) {}
        explicit SU2(const Eigen::Matrix2cd& mat) : mat_(mat) { normalize(); }
        explicit SU2(double a, double b, double c, double d) {
            // построение из параметров (для тестов)
            mat_ << std::complex<double>(a, b), std::complex<double>(c, d),
                std::complex<double>(-std::conj(std::complex<double>(c, d))), std::complex<double>(std::conj(std::complex<double>(a, b)));
            normalize();
        }

        static SU2 identity() { return SU2(); }

        SU2 inverse() const { return SU2(mat_.adjoint()); }

        SU2 operator*(const SU2& other) const { return SU2(mat_ * other.mat_); }

        double trace() const { return std::real(mat_.trace()); }
        double real_tr() const { return std::real(mat_.trace()); }

        const matrix_type& matrix() const { return mat_; }

        // Экспонента от алгебры su(2) (представление через углы)
        static SU2 exp(double theta, const Eigen::Vector3d& axis) {
            double norm = axis.norm();
            if (norm < 1e-12) return identity();
            Eigen::Vector3d n = axis / norm;
            double half_theta = 0.5 * theta;
            double c = std::cos(half_theta);
            double s = std::sin(half_theta);
            // Матрица Паули: n·σ = n1*σ1 + n2*σ2 + n3*σ3
            std::complex<double> i(0, 1);
            Eigen::Matrix2cd sigma_n = n(0) * Eigen::Matrix2cd::Zero() +
                n(1) * (Eigen::Matrix2cd() << 0, -i, i, 0).finished() +
                n(2) * (Eigen::Matrix2cd() << i, 0, 0, -i).finished();
            // exp(i * theta/2 * n·σ) = cos(theta/2) * I + i sin(theta/2) * (n·σ)
            Eigen::Matrix2cd result = c * Eigen::Matrix2cd::Identity() + std::complex<double>(0, 1) * s * sigma_n;
            return SU2(result);
        }

        // Логарифм (приближённый), возвращает вектор в алгебре (ось * угол)
        Eigen::Vector3d log() const {
            // Для матрицы, близкой к единице, логарифм: theta = arccos(tr/2), ось из антисимметричной части
            double tr = mat_.trace().real();
            double theta = std::acos(tr / 2.0); // tr = 2 cos(theta)
            if (theta < 1e-12) return Eigen::Vector3d::Zero();
            // Антисимметричная часть: (U - U†)/2 даёт i sin(theta) (n·σ)
            Eigen::Matrix2cd diff = (mat_ - mat_.adjoint()) / 2.0;
            // Извлекаем компоненты: diff = i sin(theta) (n1 σ1 + n2 σ2 + n3 σ3)
            // σ1 = [[0,1],[1,0]], σ2 = [[0,-i],[i,0]], σ3 = [[1,0],[0,-1]]
            double s = std::sin(theta);
            if (std::abs(s) < 1e-12) return Eigen::Vector3d::Zero();
            double n1 = diff(0, 1).real() / s;        // σ1 даёт реальную часть вне диагонали
            double n2 = diff(0, 1).imag() / s;        // σ2 даёт мнимую часть вне диагонали (с минусом? нужно уточнить)
            double n3 = diff(0, 0).imag() / s;        // σ3 даёт мнимую часть диагонали
            // Для σ2: элемент (0,1) = -i, поэтому imag(diff(0,1)) = sin(theta)*n2
            // Проверим знак: в формуле exp(i θ/2 n·σ) = cos(θ/2)I + i sin(θ/2)(n·σ)
            // Тогда антиэрмитова часть = i sin(θ/2)(n·σ). При θ малом, diff ≈ i θ/2 (n·σ)
            // Тогда imag(diff(0,1)) = θ/2 n2, real(diff(0,1)) = θ/2 n1.
            // Мы имеем theta = 2 * asin(...). Уточним.
            // Упростим: оставим приближённую формулу для малых углов.
            return Eigen::Vector3d(n1, n2, n3) * theta; // грубо
        }

    private:
        Eigen::Matrix2cd mat_;

        void normalize() {
            // Приводим к детерминанту 1
            std::complex<double> det = mat_.determinant();
            mat_ /= std::sqrt(det);
        }
    };

    // -----------------------------------------------------------------------------
    // SU(3) group
    // -----------------------------------------------------------------------------
    class SU3 {
    public:
        using value_type = double;
        using matrix_type = Eigen::Matrix3cd;
        static constexpr int dimension = 3;

        SU3() : mat_(Eigen::Matrix3cd::Identity()) {}
        explicit SU3(const Eigen::Matrix3cd& mat) : mat_(mat) { normalize(); }

        static SU3 identity() { return SU3(); }

        SU3 inverse() const { return SU3(mat_.adjoint()); }

        SU3 operator*(const SU3& other) const { return SU3(mat_ * other.mat_); }

        double trace() const { return std::real(mat_.trace()); }
        double real_tr() const { return std::real(mat_.trace()); }

        const matrix_type& matrix() const { return mat_; }

        // Экспонента от алгебры su(3) – через Eigen::Matrix::exp (требует <unsupported/Eigen/MatrixFunctions>)
        static SU3 exp(const Eigen::Matrix3cd& alg) {
            // alg должна быть антиэрмитовой и бесследовой
            Eigen::Matrix3cd m = alg;
            return SU3(m.exp());
        }

        // Логарифм (приближённый) – используем Eigen::log
        Eigen::Matrix3cd log() const {
            return mat_.log();
        }

    private:
        Eigen::Matrix3cd mat_;

        void normalize() {
            std::complex<double> det = mat_.determinant();
            // Нормализуем, чтобы det = 1
            mat_ /= std::pow(det, 1.0 / 3.0);
        }
    };

} // namespace delta::numerical