// transcendentals.h
#pragma once

#include "rational_class.h"
#include "context.h"
#include "expression_root.h"
#include <stdexcept>

namespace delta {

    inline Rational sqrt(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return eager_sqrt(x, eps);
        }
        if (x.approx_interval().lower() < 0) {
            throw std::domain_error("sqrt of negative number");
        }
        ExpressionRoot root = x.is_lazy() ? ExpressionRoot(x.root_index())
            : ExpressionRoot::make_const(x.to_value());
        return Rational(static_cast<std::size_t>(root.sqrt(eps).root_index()));
    }

    inline Rational exp(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return eager_exp(x, eps);
        }
        ExpressionRoot root = x.is_lazy() ? ExpressionRoot(x.root_index())
            : ExpressionRoot::make_const(x.to_value());
        return Rational(static_cast<std::size_t>(root.exp(eps).root_index()));
    }

    inline Rational log(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return eager_log(x, eps);
        }
        if (x.approx_interval().upper() <= 0) {
            throw std::domain_error("log of non-positive number");
        }
        ExpressionRoot root = x.is_lazy() ? ExpressionRoot(x.root_index())
            : ExpressionRoot::make_const(x.to_value());
        return Rational(static_cast<std::size_t>(root.log(eps).root_index()));
    }

    inline Rational sin(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return eager_sin(x, eps);
        }
        ExpressionRoot root = x.is_lazy() ? ExpressionRoot(x.root_index())
            : ExpressionRoot::make_const(x.to_value());
        return Rational(static_cast<std::size_t>(root.sin(eps).root_index()));
    }

    inline Rational cos(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return eager_cos(x, eps);
        }
        ExpressionRoot root = x.is_lazy() ? ExpressionRoot(x.root_index())
            : ExpressionRoot::make_const(x.to_value());
        return Rational(static_cast<std::size_t>(root.cos(eps).root_index()));
    }

    inline Rational acos(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return eager_acos(x, eps);
        }
        auto interval = x.approx_interval();
        if (interval.upper() < -1 || interval.lower() > 1) {
            throw std::domain_error("acos argument out of [-1,1]");
        }
        ExpressionRoot root = x.is_lazy() ? ExpressionRoot(x.root_index())
            : ExpressionRoot::make_const(x.to_value());
        return Rational(static_cast<std::size_t>(root.acos(eps).root_index()));
    }

    inline Rational pi(const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return eager_pi(eps);
        }
        return Rational(static_cast<std::size_t>(ExpressionRoot::pi(eps).root_index()));
    }

    inline Rational e(const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return eager_e(eps);
        }
        return Rational(static_cast<std::size_t>(ExpressionRoot::e(eps).root_index()));
    }

    inline Rational pow(const Rational& base, int exponent) {
        if (exponent == 0) return Rational(static_cast<absl::int128>(1));
        if (base.is_immediate()) {
            if (exponent < 0) {
                Rational pos = pow(base, -exponent);
                return eager_div(Rational(static_cast<absl::int128>(1)), pos);
            }
            Rational result = Rational(static_cast<absl::int128>(1));
            Rational b = base;
            int e = exponent;
            while (e > 0) {
                if (e & 1) result = eager_mul(result, b);
                e >>= 1;
                if (e != 0) b = eager_mul(b, b);
            }
            return result;
        }
        // lazy base
        if (exponent < 0) {
            return Rational(static_cast<absl::int128>(1)) / pow(base, -exponent);
        }
        Rational result = Rational(static_cast<absl::int128>(1));
        Rational b = base;
        int e = exponent;
        while (e > 0) {
            if (e & 1) result = result * b;
            e >>= 1;
            if (e != 0) b = b * b;
        }
        return result;
    }

} // namespace delta