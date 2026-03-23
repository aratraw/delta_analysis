// include/delta/rational/transcendentals.h
#pragma once

#include "delta/rational/literals.h"
#include "delta/rational/rational_class.h"
#include "delta/rational/context.h"
#include "delta/rational/evaluation.h"   // for eager_* functions
#include "delta/rational/simplify.h"     // for simplify
#include <absl/container/inlined_vector.h>
#include <memory>

namespace delta {

    // -------------------------------------------------------------------------
    // sqrt(x, eps)
    // -------------------------------------------------------------------------
    inline Rational sqrt(const Rational& x, const Rational& eps = default_eps()) {
        if (internal::global_eager_mode) {
            return internal::eager_sqrt(x, eps);
        }
        absl::InlinedVector<std::shared_ptr<const Rational>, 2> args;
        args.emplace_back(std::make_shared<Rational>(x));
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::SQRT, std::move(args), std::make_shared<const Rational>(eps));
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
        absl::InlinedVector<std::shared_ptr<const Rational>, 2> args;
        args.emplace_back(std::make_shared<Rational>(x));
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::EXP, std::move(args), std::make_shared<const Rational>(eps));
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
        absl::InlinedVector<std::shared_ptr<const Rational>, 2> args;
        args.emplace_back(std::make_shared<Rational>(x));
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::LOG, std::move(args), std::make_shared<const Rational>(eps));
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
        absl::InlinedVector<std::shared_ptr<const Rational>, 2> args;
        args.emplace_back(std::make_shared<Rational>(x));
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::SIN, std::move(args), std::make_shared<const Rational>(eps));
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
        absl::InlinedVector<std::shared_ptr<const Rational>, 2> args;
        args.emplace_back(std::make_shared<Rational>(x));
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::COS, std::move(args), std::make_shared<const Rational>(eps));
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
        absl::InlinedVector<std::shared_ptr<const Rational>, 2> args;
        args.emplace_back(std::make_shared<Rational>(x));
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::ACOS, std::move(args), std::make_shared<const Rational>(eps));
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
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::PI, std::make_shared<const Rational>(eps));
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
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::E, std::make_shared<const Rational>(eps));
        Rational result(node);
        return internal::simplify(result);
    }

    // -------------------------------------------------------------------------
    // pow (integer exponent)
    // -------------------------------------------------------------------------
    inline Rational pow(const Rational& base, int exponent) {
        if (exponent == 0) return Rational(1);
        if (exponent < 0) return Rational(1) / pow(base, -exponent);
        Rational result = 1_r;
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