// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

#ifndef DELTA_COMPLEX_GAUSS_QI_TRANSCENDENTALS_H
#define DELTA_COMPLEX_GAUSS_QI_TRANSCENDENTALS_H

#include "gauss_qi.h"
#include "transcendentals.h"
#include "context.h"
#include <stdexcept>

namespace delta {

    /**
     * @file gauss_qi_transcendentals.h
     * @brief Complex elementary transcendental functions for the GaussQi type.
     *
     * All functions implement the principal branch with branch cuts as defined in standard
     * complex analysis (Abramowitz & Stegun, DLMF). The real-valued helper atan2 also follows
     * the IEEE convention for the two-argument arctangent.
     *
     * Epsilon parameters control the internal precision of the rational approximations.
     * They are scaled heuristically to account for error propagation in composite expressions.
     *
     * Special cases (e.g., log(0), 0^0) throw std::domain_error.
     */

     // ----------------------------- atan2 (real) ----------------------------------
     /**
      * @brief Two-argument arctangent for real numbers.
      *
      * @param y numerator (ordinate)
      * @param x denominator (abscissa)
      * @param eps precision for computing π
      * @return angle in (-π, π] such that x = r cos θ, y = r sin θ with r ≥ 0.
      *
      * Formula:
      *   - If x > 0:  θ = atan(y / x)
      *   - If x < 0:
      *       y ≥ 0  ⇒  θ = atan(y / x) + π
      *       y < 0  ⇒  θ = atan(y / x) - π
      *   - If x == 0:
      *       y > 0  ⇒  θ = +π/2
      *       y < 0  ⇒  θ = -π/2
      *       y == 0 ⇒  undefined (throws domain_error)
      *
      * This matches the standard IEEE 754 atan2 and places the branch cut on the negative
      * real axis, giving values in (-π, π].
      */
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

    // ----------------------------- arg -------------------------------------------
    /**
     * @brief Principal argument (phase) of a complex number.
     *
     * @param z non‑zero complex number
     * @param eps precision
     * @return Arg(z) ∈ (-π, π]
     *
     * Defined as atan2(Im(z), Re(z)).  The branch cut is the negative real axis: for
     * real negative z the result is +π.
     */
    inline Rational arg(const GaussQi& z, const Rational& eps = default_eps()) {
        return atan2(z.imag(), z.real(), eps);
    }

    // ----------------------------- abs, sqrt ------------------------------------
    /**
     * @brief Absolute value (modulus) of a complex number.
     *
     * @param z complex number
     * @param eps precision for the real square root
     * @return |z| = sqrt(Re(z)^2 + Im(z)^2)
     *
     * Always non‑negative real.
     */
    inline Rational abs(const GaussQi& z, const Rational& eps = default_eps()) {
        return delta::sqrt(z.norm(), eps);
    }

    /**
     * @brief Principal square root of a complex number.
     *
     * @param z complex number
     * @param eps precision
     * @return sqrt(z) with branch cut along the negative real axis.
     *
     * Formula (using the principal value):
     *   Let r = |z|.  Then
     *     Re(sqrt(z)) = sqrt((r + Re(z))/2)
     *     Im(sqrt(z)) = sign(Im(z)) * sqrt((r - Re(z))/2)
     *   where sign(0) is taken as +1 (the formula gives 0 for real non‑negative z).
     *
     * For z = 0 the result is exactly 0.
     *
     * For real negative z (Im(z) = 0, Re(z) < 0) this yields a pure imaginary number
     * with positive imaginary part (the principal branch).
     */
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
    /**
     * @brief Complex exponential function.
     *
     * @param z = a + i b
     * @param eps precision
     * @return exp(z) = e^a (cos b + i sin b)
     *
     * The exponential is an entire function; no branch cuts.
     * The trigonometric functions are evaluated with a separately scaled epsilon
     * (tighter for large |a|) to bound the absolute error.
     */
    inline GaussQi exp(const GaussQi& z, const Rational& eps = default_eps()) {
        Rational a = z.real(), b = z.imag();
        // Для больших |a| полагаемся на float-путь, иначе используем базовый налог
        Rational trig_eps = (a > 10_r || a < -10_r) ? eps / 10000_r : eps / 100_r;
        Rational exp_a = delta::exp(a, eps);
        Rational cos_b = delta::cos(b, trig_eps);
        Rational sin_b = delta::sin(b, trig_eps);
        return GaussQi(exp_a * cos_b, exp_a * sin_b);
    }

    /**
     * @brief Principal value of the complex natural logarithm.
     *
     * @param z ≠ 0 complex number
     * @param eps precision
     * @return log(z) = ln|z| + i Arg(z),  where Arg(z) ∈ (-π, π]
     *
     * Branch cut: the negative real axis.  For real negative z, Im(log(z)) = +π.
     *
     * Special case: log(0) throws std::domain_error.
     */
    inline GaussQi log(const GaussQi& z, const Rational& eps = default_eps()) {
        if (z.real() == Rational(0) && z.imag() == Rational(0))
            throw std::domain_error("log(0) is undefined");

        // Сумма погрешностей двух компонент -> делим eps на 2
        Rational half_eps = eps / 2_r;
        Rational ln_r = delta::log(abs(z, half_eps), half_eps);
        Rational arg_z = arg(z, half_eps);
        return GaussQi(ln_r, arg_z);
    }

    // ----------------------------- pow (general and integer) --------------------
    /**
     * @brief Complex power function (principal branch).
     *
     * @param z base
     * @param w exponent
     * @param eps precision
     * @return z^w = exp(w * log(z))
     *
     * Branch cut inherited from log(z): the negative real axis.
     * The principal branch is chosen: for a negative real base and non‑integer exponent
     * the result lies on the principal sheet.
     *
     * Special cases:
     *   - 0^0  → domain_error
     *   - 0^{negative real} → domain_error
     *   - 0^{complex with non‑zero imaginary part} → domain_error
     *   - 0^{positive real} → 0
     *
     * The exponent w is allowed to be any complex number.
     */
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

    /**
     * @brief Integer power of a complex number (exact arithmetic).
     *
     * @param base complex number
     * @param exponent integer
     * @return base^exponent computed by exponentiation by squaring.
     *
     * This overload does not use log/exp, so it avoids branch cut issues and is
     * exact up to rational multiplication error.  It follows the algebraic convention
     * that 0^0 = 1 (as a discrete limit of x^0), though the general complex pow()
     * treats 0^0 as an error.
     */
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
    /**
     * @brief Complex sine.
     *
     * @param z = a + i b
     * @param eps precision
     * @return sin(z) = sin a cosh b + i cos a sinh b
     *
     * Entire function; no branch cuts.
     */
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

    /**
     * @brief Complex cosine.
     *
     * @param z = a + i b
     * @param eps precision
     * @return cos(z) = cos a cosh b - i sin a sinh b
     *
     * Entire function; no branch cuts.
     */
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

    /**
     * @brief Complex tangent.
     *
     * @param z complex number
     * @param eps precision
     * @return tan(z) = sin(z) / cos(z)
     *
     * Singularities at z = π/2 + kπ (where cos(z)=0).  If cos(z) evaluates exactly to
     * 0+0i, a domain_error is thrown.
     */
    inline GaussQi tan(const GaussQi& z, const Rational& eps = default_eps()) {
        GaussQi s = sin(z, eps);
        GaussQi c = cos(z, eps);
        if (c.real() == 0_r && c.imag() == 0_r)
            throw std::domain_error("tan(z): cos(z) is zero");
        return s / c;
    }

    // ----------------------------- asin, acos, atan (with corrected branch cuts) --
    /**
     * @brief Rough L1 norm used for heuristic epsilon scaling near singularities.
     *
     * Returns |Re(z)| + |Im(z)|, cheaply estimating the distance to the unit circle.
     */
    static inline Rational rough_norm(const GaussQi& z) {
        Rational ax = z.real() > 0_r ? z.real() : -z.real();
        Rational ay = z.imag() > 0_r ? z.imag() : -z.imag();
        return ax + ay;   // норма L1, дешевая
    }

    /**
     * @brief Principal value of the inverse sine (arcsine).
     *
     * @param z complex number
     * @param eps precision
     * @return asin(z) = -i log(i z + sqrt(1 - z^2))
     *
     * Branch cuts: (-∞, -1] and [1, ∞) on the real axis.
     * The function satisfies asin(-z) = -asin(z) (odd symmetry).
     *
     * Implementation:
     *   - For Re(z) < 0 or (Re(z)==0 and Im(z) < 0), the oddness is used to map to
     *     the right half‑plane, ensuring the principal branch.
     *   - Near the unit circle (rough_norm ≈ 1) the epsilon for sqrt is tightened
     *     because the derivative blows up.
     *
     * Special values: asin(±1) = ±π/2; asin(0) = 0.
     */
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

    /**
     * @brief Principal value of the inverse cosine (arccosine).
     *
     * @param z complex number
     * @param eps precision
     * @return acos(z) = π/2 - asin(z)
     *
     * Branch cuts: (-∞, -1] and [1, ∞) on the real axis.
     * The principal branch satisfies Re(acos(z)) ∈ [0, π].
     *
     * Implementation uses the identity acos(z) = π/2 - asin(z) and ensures the
     * symmetry acos(z̅) = acos(z)̅ by explicitly conjugating when Im(z) < 0.
     * This guarantees the correct branch without extra computations.
     */
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

    /**
     * @brief Principal value of the inverse tangent (arctangent).
     *
     * @param z complex number
     * @param eps precision
     * @return atan(z) = (i/2) [log(1 - i z) - log(1 + i z)]
     *
     * Branch cuts: the imaginary axis from i to i∞ and from -i to -i∞.
     * The principal branch satisfies Re(atan(z)) ∈ (-π/2, π/2) and atan(-z) = -atan(z).
     *
     * Special points:
     *   - atan(±i) are branch points and throw domain_error.
     *
     * Implementation uses the oddness to map left half‑plane (Re(z)<0) to the right,
     * and then applies the standard logarithmic formula with a moderately scaled epsilon.
     */
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