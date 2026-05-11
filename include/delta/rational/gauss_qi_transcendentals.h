// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

#ifndef DELTA_COMPLEX_GAUSS_QI_TRANSCENDENTALS_H
#define DELTA_COMPLEX_GAUSS_QI_TRANSCENDENTALS_H

#include "gauss_qi.h"
#include "transcendentals.h"   // delta::exp, delta::log, delta::sqrt, delta::sin, delta::cos, delta::atan, delta::pi
#include "context.h"            // delta::default_eps
#include <stdexcept>

namespace delta {

    // ----------------------------------------------------------------------------
    // atan2 для Rational – чисто рациональная реализация через atan(y/x)
    // Возвращает главное значение аргумента в (-π, π] с точностью eps.
    // ----------------------------------------------------------------------------
    inline Rational atan2(const Rational& y, const Rational& x, const Rational& eps = default_eps()) {
        if (x == Rational(0) && y == Rational(0)) {
            throw std::domain_error("atan2(0,0) is undefined");
        }

        if (x == Rational(0)) {
            // Точка на мнимой оси
            if (y > Rational(0)) {
                return delta::pi(eps) / Rational(2);
            }
            else {
                return -delta::pi(eps) / Rational(2);
            }
        }

        Rational angle = delta::atan(y / x, eps);

        if (x > Rational(0)) {
            return angle;
        }
        else { // x < 0
            if (y >= Rational(0)) {
                return angle + delta::pi(eps);
            }
            else {
                return angle - delta::pi(eps);
            }
        }
    }

    inline Rational arg(const GaussQi& z, const Rational& eps = default_eps()) {
        return atan2(z.imag(), z.real(), eps);
    }
    // ----------------------------------------------------------------------------
    // Модуль (абсолютное значение) комплексного числа – возвращает Rational
    // ----------------------------------------------------------------------------
    inline Rational abs(const GaussQi& z, const Rational& eps = default_eps()) {
        return delta::sqrt(z.norm(), eps);
    }

    // ----------------------------------------------------------------------------
    // Квадратный корень комплексного числа (главная ветвь)
    // sqrt(z) = sqrt((|z|+Re(z))/2) + i * sign(Im(z)) * sqrt((|z|-Re(z))/2)
    // ----------------------------------------------------------------------------
    inline GaussQi sqrt(const GaussQi& z, const Rational& eps = default_eps()) {
        if (z.real() == Rational(0) && z.imag() == Rational(0)) {
            return GaussQi(Rational(0));
        }

        Rational r = abs(z, eps);            
        Rational re_part = delta::sqrt((r + z.real()) / Rational(2), eps);
        Rational im_part = delta::sqrt((r - z.real()) / Rational(2), eps);

        if (z.imag() < Rational(0)) {
            im_part = -im_part;
        }
        return GaussQi(re_part, im_part);
    }

    // ----------------------------------------------------------------------------
    // Экспонента комплексного числа
    // exp(a+bi) = exp(a) * (cos(b) + i*sin(b))
    // ----------------------------------------------------------------------------
    inline GaussQi exp(const GaussQi& z, const Rational& eps = default_eps()) {
        Rational exp_a = delta::exp(z.real(), eps);
        Rational cos_b = delta::cos(z.imag(), eps);
        Rational sin_b = delta::sin(z.imag(), eps);
        return GaussQi(exp_a * cos_b, exp_a * sin_b);
    }

    // ----------------------------------------------------------------------------
    // Натуральный логарифм комплексного числа (главная ветвь)
    // log(z) = ln|z| + i * arg(z)
    // ----------------------------------------------------------------------------
    inline GaussQi log(const GaussQi& z, const Rational& eps = default_eps()) {
        if (z.real() == Rational(0) && z.imag() == Rational(0)) {
            throw std::domain_error("log(0) is undefined");
        }
        Rational ln_r = delta::log(abs(z, eps), eps);
        Rational arg_z = atan2(z.imag(), z.real(), eps);
        return GaussQi(ln_r, arg_z);
    }

    // ----------------------------------------------------------------------------
    // Степень комплексного числа (главная ветвь) – общая версия
    // pow(z, w) = exp(w * log(z))
    // ----------------------------------------------------------------------------
    inline GaussQi pow(const GaussQi& z, const GaussQi& w, const Rational& eps = default_eps()) {
        if (z.real() == Rational(0) && z.imag() == Rational(0)) {
            if (w.real() == Rational(0) && w.imag() == Rational(0)) {
                throw std::domain_error("0^0 is undefined");
            }
            if (w.real() < Rational(0)) {
                throw std::domain_error("0^negative is undefined");
            }
            if (w.imag() != Rational(0)) {
                // Для комплексного показателя результат не определён однозначно
                throw std::domain_error("0^(complex exponent) is not defined");
            }
            return GaussQi(Rational(0));
        }
        return exp(w * log(z, eps), eps);
    }

    // ----------------------------------------------------------------------------
    // Степень с целым показателем (без eps, точное возведение)
    // ----------------------------------------------------------------------------
    inline GaussQi pow(const GaussQi& base, int exponent) {
        if (exponent == 0) return GaussQi(Rational(1));
        if (exponent < 0) {
            GaussQi inv = GaussQi(Rational(1)) / base;
            return pow(inv, -exponent);
        }
        GaussQi result = GaussQi(Rational(1));
        GaussQi b = base;
        int e = exponent;
        while (e > 0) {
            if (e & 1) result *= b;
            e >>= 1;
            if (e) b = b * b;
        }
        return result;
    }
    // ============================================================================
// Trigonometric functions for GaussQi
// ============================================================================

    inline GaussQi sin(const GaussQi& z, const Rational& eps = default_eps()) {
        // sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        Rational a = z.real();
        Rational b = z.imag();
        Rational sin_a = delta::sin(a, eps);
        Rational cos_a = delta::cos(a, eps);
        Rational sinh_b = (delta::exp(b, eps) - delta::exp(-b, eps)) / 2_r;
        Rational cosh_b = (delta::exp(b, eps) + delta::exp(-b, eps)) / 2_r;
        return GaussQi(sin_a * cosh_b, cos_a * sinh_b);
    }

    inline GaussQi cos(const GaussQi& z, const Rational& eps = default_eps()) {
        // cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
        Rational a = z.real();
        Rational b = z.imag();
        Rational sin_a = delta::sin(a, eps);
        Rational cos_a = delta::cos(a, eps);
        Rational sinh_b = (delta::exp(b, eps) - delta::exp(-b, eps)) / 2_r;
        Rational cosh_b = (delta::exp(b, eps) + delta::exp(-b, eps)) / 2_r;
        return GaussQi(cos_a * cosh_b, -sin_a * sinh_b);
    }

    inline GaussQi tan(const GaussQi& z, const Rational& eps = default_eps()) {
        GaussQi s = sin(z, eps);
        GaussQi c = cos(z, eps);
        if (c.real() == 0_r && c.imag() == 0_r)
            throw std::domain_error("tan(z): cos(z) is zero");
        return s / c;
    }

    inline GaussQi asin(const GaussQi& z, const Rational& eps = default_eps()) {
        // asin(z) = -i * log(i*z + sqrt(1 - z^2))
        GaussQi i(0, 1);
        GaussQi one(1, 0);
        GaussQi iz = i * z;
        GaussQi sqrt_term = delta::sqrt(one - z * z, eps);
        GaussQi log_arg = iz + sqrt_term;
        GaussQi log_val = delta::log(log_arg, eps);
        return -i * log_val;
    }

    inline GaussQi acos(const GaussQi& z, const Rational& eps = default_eps()) {
        // acos(z) = π/2 - asin(z)
        GaussQi half_pi(delta::pi(eps) / 2_r, 0);
        return half_pi - asin(z, eps);
    }

    inline GaussQi atan(const GaussQi& z, const Rational& eps = default_eps()) {
        // atan(z) = (i/2) * (log(1 - i*z) - log(1 + i*z))
        GaussQi i(0, 1);
        GaussQi one(1, 0);
        GaussQi iz = i * z;
        GaussQi log1 = delta::log(one - iz, eps);
        GaussQi log2 = delta::log(one + iz, eps);
        return (i / 2_r) * (log1 - log2);
    }
} // namespace delta

#endif // DELTA_COMPLEX_GAUSS_QI_TRANSCENDENTALS_H