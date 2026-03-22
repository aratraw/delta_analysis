// include/delta/rational/transcendentals.h
#pragma once

#include "delta/rational/rational_class.h"
#include "delta/rational/context.h"
#include "delta/rational/evaluation.h"   // for eager_* functions
#include "delta/rational/simplify.h"     // for simplify
#include <memory>
#include <vector>

namespace delta {

    // -------------------------------------------------------------------------
    // sqrt(x, eps)
    // -------------------------------------------------------------------------
    inline Rational sqrt(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return internal::eager_sqrt(x, eps);
        }
        auto node = std::make_shared<internal::LazyNode>(
            internal::LazyOp::SQRT,
            std::vector<std::shared_ptr<const Rational>>{std::make_shared<Rational>(x)},
            eps
        );
        Rational result(node);
        return internal::simplify(result);
    }

    // -------------------------------------------------------------------------
    // exp(x, eps)
    // -------------------------------------------------------------------------
    inline Rational exp(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return internal::eager_exp(x, eps);
        }
        auto node = std::make_shared<internal::LazyNode>(
            internal::LazyOp::EXP,
            std::vector<std::shared_ptr<const Rational>>{std::make_shared<Rational>(x)},
            eps
        );
        Rational result(node);
        return internal::simplify(result);
    }

    // -------------------------------------------------------------------------
    // log(x, eps)
    // -------------------------------------------------------------------------
    inline Rational log(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return internal::eager_log(x, eps);
        }
        auto node = std::make_shared<internal::LazyNode>(
            internal::LazyOp::LOG,
            std::vector<std::shared_ptr<const Rational>>{std::make_shared<Rational>(x)},
            eps
        );
        Rational result(node);
        return internal::simplify(result);
    }

    // -------------------------------------------------------------------------
    // sin(x, eps)
    // -------------------------------------------------------------------------
    inline Rational sin(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return internal::eager_sin(x, eps);
        }
        auto node = std::make_shared<internal::LazyNode>(
            internal::LazyOp::SIN,
            std::vector<std::shared_ptr<const Rational>>{std::make_shared<Rational>(x)},
            eps
        );
        Rational result(node);
        return internal::simplify(result);
    }

    // -------------------------------------------------------------------------
    // cos(x, eps)
    // -------------------------------------------------------------------------
    inline Rational cos(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return internal::eager_cos(x, eps);
        }
        auto node = std::make_shared<internal::LazyNode>(
            internal::LazyOp::COS,
            std::vector<std::shared_ptr<const Rational>>{std::make_shared<Rational>(x)},
            eps
        );
        Rational result(node);
        return internal::simplify(result);
    }

    // -------------------------------------------------------------------------
    // acos(x, eps)
    // -------------------------------------------------------------------------
    inline Rational acos(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return internal::eager_acos(x, eps);
        }
        auto node = std::make_shared<internal::LazyNode>(
            internal::LazyOp::ACOS,
            std::vector<std::shared_ptr<const Rational>>{std::make_shared<Rational>(x)},
            eps
        );
        Rational result(node);
        return internal::simplify(result);
    }

    // -------------------------------------------------------------------------
    // pi(eps)
    // -------------------------------------------------------------------------
    inline Rational pi(const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return internal::eager_pi(eps);
        }
        auto node = std::make_shared<internal::LazyNode>(
            internal::LazyOp::PI,
            std::vector<std::shared_ptr<const Rational>>{},
            eps
        );
        Rational result(node);
        return internal::simplify(result);
    }

    // -------------------------------------------------------------------------
    // e(eps)
    // -------------------------------------------------------------------------
    inline Rational e(const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return internal::eager_e(eps);
        }
        auto node = std::make_shared<internal::LazyNode>(
            internal::LazyOp::E,
            std::vector<std::shared_ptr<const Rational>>{},
            eps
        );
        Rational result(node);
        return internal::simplify(result);
    }

} // namespace delta