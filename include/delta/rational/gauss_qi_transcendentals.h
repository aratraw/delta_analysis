// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

#ifndef DELTA_COMPLEX_GAUSS_QI_TRANSCENDENTALS_H
#define DELTA_COMPLEX_GAUSS_QI_TRANSCENDENTALS_H

#include "gauss_qi.h"
#include "transcendentals.h"
#include "context.h"
#include <stdexcept>

namespace delta {

    // ----------------------------- atan2 (вещественный) -------------------------
    inline Rational atan2(const Rational& y, const Rational& x, const Rational& eps = default_eps()) {
        if (x == Rational(0) && y == Rational(0))
            throw std::domain_error("atan2(0,0) is undefined");

        if (x == Rational(0)) {
            return (y > Rational(0)) ? delta::pi(eps) / Rational(2) : -delta::pi(eps) / Rational(2);
        }

        Rational angle = delta::atan(y / x, eps);
        if (x > Rational(0)) return angle;
        // x < 0
        if (y >= Rational(0)) return angle + delta::pi(eps);
        else                 return angle - delta::pi(eps);
    }

    inline Rational arg(const GaussQi& z, const Rational& eps = default_eps()) {
        return atan2(z.imag(), z.real(), eps);
    }

    // ----------------------------- abs, sqrt ------------------------------------
    inline Rational abs(const GaussQi& z, const Rational& eps = default_eps()) {
        return delta::sqrt(z.norm(), eps);
    }

    inline GaussQi sqrt(const GaussQi& z, const Rational& eps = default_eps()) {
        if (z.real() == Rational(0) && z.imag() == Rational(0))
            return GaussQi(Rational(0));

        // Для двух вещественных корней закладываем запас 10
        Rational half_eps = eps / 10_r;

        Rational r = abs(z, half_eps);
        Rational re_part = delta::sqrt((r + z.real()) / Rational(2), half_eps);
        Rational im_part = delta::sqrt((r - z.real()) / Rational(2), half_eps);
        if (z.imag() < Rational(0)) im_part = -im_part;
        return GaussQi(re_part, im_part);
    }

    // ----------------------------- exp, log -------------------------------------
    inline GaussQi exp(const GaussQi& z, const Rational& eps = default_eps()) {
        Rational a = z.real(), b = z.imag();
        // Для больших |a| полагаемся на float-путь, иначе используем базовый налог
        Rational trig_eps = (a > 10_r || a < -10_r) ? eps / 10000_r : eps / 100_r;
        Rational exp_a = delta::exp(a, eps);
        Rational cos_b = delta::cos(b, trig_eps);
        Rational sin_b = delta::sin(b, trig_eps);
        return GaussQi(exp_a * cos_b, exp_a * sin_b);
    }

    inline GaussQi log(const GaussQi& z, const Rational& eps = default_eps()) {
        if (z.real() == Rational(0) && z.imag() == Rational(0))
            throw std::domain_error("log(0) is undefined");

        // Сумма погрешностей двух компонент -> делим eps на 2
        Rational half_eps = eps / 2_r;
        Rational ln_r = delta::log(abs(z, half_eps), half_eps);
        Rational arg_z = arg(z, half_eps);
        return GaussQi(ln_r, arg_z);
    }

    // ----------------------------- pow (общая и целая) --------------------------
    inline GaussQi pow(const GaussQi& z, const GaussQi& w, const Rational& eps = default_eps()) {
        if (z.real() == Rational(0) && z.imag() == Rational(0)) {
            if (w.real() == Rational(0) && w.imag() == Rational(0))
                throw std::domain_error("0^0 is undefined");
            if (w.real() < Rational(0))
                throw std::domain_error("0^negative is undefined");
            if (w.imag() != Rational(0))
                throw std::domain_error("0^(complex exponent) is not defined");
            return GaussQi(Rational(0));
        }
        // Два вложенных вызова: log и exp -> делим eps дважды
        Rational log_eps = eps / 100_r;
        GaussQi log_z = log(z, log_eps);
        GaussQi prod = w * log_z;
        return exp(prod, eps / 10_r);
    }

    inline GaussQi pow(const GaussQi& base, int exponent) {
        if (exponent == 0) return GaussQi(Rational(1));
        if (exponent < 0) return GaussQi(Rational(1)) / pow(base, -exponent);
        GaussQi result(1_r), b = base;
        int e = exponent;
        while (e) {
            if (e & 1) result *= b;
            e >>= 1;
            if (e) b = b * b;
        }
        return result;
    }

    // ----------------------------- sin, cos, tan --------------------------------
    inline GaussQi sin(const GaussQi& z, const Rational& eps = default_eps()) {
        Rational a = z.real(), b = z.imag();
        Rational sin_a = delta::sin(a, eps);
        Rational cos_a = delta::cos(a, eps);
        // Для гиперболических функций используем запас 100
        Rational hyp_eps = eps / 100_r;
        Rational exp_b = delta::exp(b, hyp_eps);
        Rational exp_neg_b = delta::exp(-b, hyp_eps);
        Rational sinh_b = (exp_b - exp_neg_b) / 2_r;
        Rational cosh_b = (exp_b + exp_neg_b) / 2_r;
        return GaussQi(sin_a * cosh_b, cos_a * sinh_b);
    }

    inline GaussQi cos(const GaussQi& z, const Rational& eps = default_eps()) {
        Rational a = z.real(), b = z.imag();
        Rational sin_a = delta::sin(a, eps);
        Rational cos_a = delta::cos(a, eps);
        Rational hyp_eps = eps / 100_r;
        Rational exp_b = delta::exp(b, hyp_eps);
        Rational exp_neg_b = delta::exp(-b, hyp_eps);
        Rational sinh_b = (exp_b - exp_neg_b) / 2_r;
        Rational cosh_b = (exp_b + exp_neg_b) / 2_r;
        return GaussQi(cos_a * cosh_b, -sin_a * sinh_b);
    }

    inline GaussQi tan(const GaussQi& z, const Rational& eps = default_eps()) {
        GaussQi s = sin(z, eps);
        GaussQi c = cos(z, eps);
        if (c.real() == 0_r && c.imag() == 0_r)
            throw std::domain_error("tan(z): cos(z) is zero");
        return s / c;
    }

    // ----------------------------- asin, acos, atan (с исправленными разрезами) --
    // Вспомогательная функция для грубой оценки расстояния до сингулярности
    static inline Rational rough_norm(const GaussQi& z) {
        Rational ax = z.real() > 0_r ? z.real() : -z.real();
        Rational ay = z.imag() > 0_r ? z.imag() : -z.imag();
        return ax + ay;   // норма L1, дешевая
    }

    inline GaussQi asin(const GaussQi& z, const Rational& eps = default_eps()) {
        if (z.real() < 0_r || (z.real() == 0_r && z.imag() < 0_r))
            return -asin(-z, eps);

        // Оценка близости к единичной окружности (особые точки ±1)
        Rational rd = rough_norm(z);
        Rational sqrt_eps = (rd > 9_r / 10 && rd < 11_r / 10) ? eps / 1000_r : eps / 100_r;

        GaussQi i(0, 1);
        GaussQi one(1, 0);
        GaussQi sqrt_term = delta::sqrt(one - z * z, sqrt_eps);
        GaussQi log_arg = i * z + sqrt_term;
        // Логарифм получает базовый налог 100
        return -i * delta::log(log_arg, eps / 100_r);
    }

    inline GaussQi acos(const GaussQi& z, const Rational& eps = default_eps()) {
        // Восстанавливаем симметрию относительно вещественной оси
        if (z.imag() < 0_r) {
            GaussQi conj_z(z.real(), -z.imag());
            GaussQi res = acos(conj_z, eps);
            return GaussQi(res.real(), -res.imag());
        }
        // Для верхней полуплоскости используем asin с тем же eps
        GaussQi half_pi(delta::pi(eps) / 2_r, 0);
        return half_pi - asin(z, eps);
    }

    inline GaussQi atan(const GaussQi& z, const Rational& eps = default_eps()) {
        if (z.real() == 0_r && (z.imag() == 1_r || z.imag() == -1_r))
            throw std::domain_error("atan(z) undefined at z = +-i");
        if (z.real() < 0_r || (z.real() == 0_r && z.imag() < 0_r))
            return -atan(-z, eps);

        // Производная atan ≤ 1, достаточно запаса 10
        Rational internal_eps = eps / 10_r;
        GaussQi i(0, 1), one(1, 0);
        GaussQi iz = i * z;
        GaussQi log1 = delta::log(one - iz, internal_eps);
        GaussQi log2 = delta::log(one + iz, internal_eps);
        return (i / 2_r) * (log1 - log2);
    }

} // namespace delta

#endif // DELTA_COMPLEX_GAUSS_QI_TRANSCENDENTALS_H