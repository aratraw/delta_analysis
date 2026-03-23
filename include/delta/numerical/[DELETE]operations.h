// include/delta/rational/operations.h
#pragma once

#include "delta/rational/rational_class.h"
#include "delta/rational/context.h"
#include "delta/rational/simplify.h"
#include <absl/container/inlined_vector.h>
#include "delta/rational/rational_fwd.h"
#include <memory>

namespace delta {

    // Forward declarations of eager arithmetic functions (defined in evaluation.h)
    namespace internal {
        Rational eager_add(const Rational& a, const Rational& b);
        Rational eager_sub(const Rational& a, const Rational& b);
        Rational eager_mul(const Rational& a, const Rational& b);
        Rational eager_div(const Rational& a, const Rational& b);
        Rational eager_neg(const Rational& a);

        Rational simplify(const Rational& r);
    }

    // -------------------------------------------------------------------------
    // Arithmetic operators (create lazy nodes or evaluate eagerly)
    // -------------------------------------------------------------------------

    inline Rational operator+(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode) {
            return internal::eager_add(a, b);
        }
        absl::InlinedVector<std::shared_ptr<const Rational>, 2> args;
        args.emplace_back(std::make_shared<Rational>(a));
        args.emplace_back(std::make_shared<Rational>(b));
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::ADD, std::move(args));
        Rational result(node);
        return internal::simplify(result);
    }

    inline Rational operator-(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode) {
            return internal::eager_sub(a, b);
        }
        absl::InlinedVector<std::shared_ptr<const Rational>, 2> args;
        args.emplace_back(std::make_shared<Rational>(a));
        args.emplace_back(std::make_shared<Rational>(b));
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::SUB, std::move(args));
        Rational result(node);
        return internal::simplify(result);
    }

    inline Rational operator*(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode) {
            return internal::eager_mul(a, b);
        }
        absl::InlinedVector<std::shared_ptr<const Rational>, 2> args;
        args.emplace_back(std::make_shared<Rational>(a));
        args.emplace_back(std::make_shared<Rational>(b));
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::MUL, std::move(args));
        Rational result(node);
        return internal::simplify(result);
    }

    inline Rational operator/(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode) {
            return internal::eager_div(a, b);
        }
        absl::InlinedVector<std::shared_ptr<const Rational>, 2> args;
        args.emplace_back(std::make_shared<Rational>(a));
        args.emplace_back(std::make_shared<Rational>(b));
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::DIV, std::move(args));
        Rational result(node);
        return internal::simplify(result);
    }

    inline Rational operator-(const Rational& a) {
        if (internal::global_eager_mode) {
            return internal::eager_neg(a);
        }
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::NEG, std::make_shared<Rational>(a));
        Rational result(node);
        return internal::simplify(result);
    }

    // -------------------------------------------------------------------------
    // Compound assignment operators (use the above operators)
    // -------------------------------------------------------------------------
    inline Rational& operator+=(Rational& a, const Rational& b) {
        a = a + b;
        return a;
    }
    inline Rational& operator-=(Rational& a, const Rational& b) {
        a = a - b;
        return a;
    }
    inline Rational& operator*=(Rational& a, const Rational& b) {
        a = a * b;
        return a;
    }
    inline Rational& operator/=(Rational& a, const Rational& b) {
        a = a / b;
        return a;
    }

} // namespace delta