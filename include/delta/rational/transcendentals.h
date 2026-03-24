#pragma once

#include "rational_class.h"
#include "context.h"
#include "eager.h"
#include "expression_root_factories.h"
#include <stdexcept>

namespace delta {

    inline Rational sqrt(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return internal::eager_sqrt(x, eps);
        }
        if (x.approx_interval().lower() < 0) {
            throw std::domain_error("sqrt of negative number");
        }
        auto root = internal::make_sqrt(x, eps);
        return Rational(std::make_shared<const internal::ExpressionRoot>(root));
    }

    inline Rational exp(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return internal::eager_exp(x, eps);
        }
        auto root = internal::make_exp(x, eps);
        return Rational(std::make_shared<const internal::ExpressionRoot>(root));
    }

    inline Rational log(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return internal::eager_log(x, eps);
        }
        if (x.approx_interval().upper() <= 0) {
            throw std::domain_error("log of non-positive number");
        }
        auto root = internal::make_log(x, eps);
        return Rational(std::make_shared<const internal::ExpressionRoot>(root));
    }

    inline Rational sin(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return internal::eager_sin(x, eps);
        }
        auto root = internal::make_sin(x, eps);
        return Rational(std::make_shared<const internal::ExpressionRoot>(root));
    }

    inline Rational cos(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return internal::eager_cos(x, eps);
        }
        auto root = internal::make_cos(x, eps);
        return Rational(std::make_shared<const internal::ExpressionRoot>(root));
    }

    inline Rational acos(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return internal::eager_acos(x, eps);
        }
        auto interval = x.approx_interval();
        if (interval.upper() < -1 || interval.lower() > 1) {
            throw std::domain_error("acos argument out of [-1,1]");
        }
        auto root = internal::make_acos(x, eps);
        return Rational(std::make_shared<const internal::ExpressionRoot>(root));
    }

    inline Rational pi(const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return internal::eager_pi(eps);
        }
        auto root = internal::make_pi(eps);
        return Rational(std::make_shared<const internal::ExpressionRoot>(root));
    }

    inline Rational e(const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return internal::eager_e(eps);
        }
        auto root = internal::make_e(eps);
        return Rational(std::make_shared<const internal::ExpressionRoot>(root));
    }

    inline Rational pow(const Rational& base, int exponent) {
        if (exponent == 0) return Rational(1);
        if (exponent < 0) return Rational(1) / pow(base, -exponent);
        Rational result = Rational(1);
        Rational b = base;
        int e = exponent;
        while (e > 0) {
            if (e & 1) result = result * b;
            b = b * b;
            e >>= 1;
        }
        return result;
    }

} // namespace delta