#pragma once

#include "rational_class.h"
#include "expression_root.h"
#include "context.h"

#include <memory>
#include <variant>

namespace delta::internal {

    // ----------------------------------------------------------------------------
    //  Convert a Rational to a shared pointer to a lazy ExpressionRoot
    // ----------------------------------------------------------------------------
    inline std::shared_ptr<const ExpressionRoot> to_lazy_root(const Rational& r) {
        if (r.is_lazy()) {
            return r.as_lazy();
        }
        // Immediate -> create a CONST node
        Value v = r.to_value();
        return std::make_shared<const ExpressionRoot>(v);
    }

    // ----------------------------------------------------------------------------
    //  Binary operations
    // ----------------------------------------------------------------------------
    inline ExpressionRoot make_add(const Rational& a, const Rational& b) {
        auto root_a = to_lazy_root(a);
        auto root_b = to_lazy_root(b);
        return root_a->add(*root_b);
    }

    inline ExpressionRoot make_sub(const Rational& a, const Rational& b) {
        auto root_a = to_lazy_root(a);
        auto root_b = to_lazy_root(b);
        return root_a->sub(*root_b);
    }

    inline ExpressionRoot make_mul(const Rational& a, const Rational& b) {
        auto root_a = to_lazy_root(a);
        auto root_b = to_lazy_root(b);
        return root_a->mul(*root_b);
    }

    inline ExpressionRoot make_div(const Rational& a, const Rational& b) {
        auto root_a = to_lazy_root(a);
        auto root_b = to_lazy_root(b);
        return root_a->div(*root_b);
    }

    // ----------------------------------------------------------------------------
    //  Unary operations
    // ----------------------------------------------------------------------------
    inline ExpressionRoot make_neg(const Rational& a) {
        auto root = to_lazy_root(a);
        return root->neg();
    }

    // ----------------------------------------------------------------------------
    //  Transcendental functions (each takes an epsilon)
    // ----------------------------------------------------------------------------
    inline ExpressionRoot make_sqrt(const Rational& x, const Rational& eps) {
        auto root = to_lazy_root(x);
        return root->sqrt(eps);
    }

    inline ExpressionRoot make_exp(const Rational& x, const Rational& eps) {
        auto root = to_lazy_root(x);
        return root->exp(eps);
    }

    inline ExpressionRoot make_log(const Rational& x, const Rational& eps) {
        auto root = to_lazy_root(x);
        return root->log(eps);
    }

    inline ExpressionRoot make_sin(const Rational& x, const Rational& eps) {
        auto root = to_lazy_root(x);
        return root->sin(eps);
    }

    inline ExpressionRoot make_cos(const Rational& x, const Rational& eps) {
        auto root = to_lazy_root(x);
        return root->cos(eps);
    }

    inline ExpressionRoot make_acos(const Rational& x, const Rational& eps) {
        auto root = to_lazy_root(x);
        return root->acos(eps);
    }

    // ----------------------------------------------------------------------------
    //  Constants (π and e)
    // ----------------------------------------------------------------------------
    inline ExpressionRoot make_pi(const Rational& eps) {
        return ExpressionRoot::pi(eps);
    }

    inline ExpressionRoot make_e(const Rational& eps) {
        return ExpressionRoot::e(eps);
    }

} // namespace delta::internal