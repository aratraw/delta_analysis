// include/delta/rational/operations.h
#pragma once

#include "delta/rational/rational_class.h"
#include "delta/rational/context.h"
#include <memory>
#include <vector>
#include "delta/rational/evaluation.h"
#include "delta/rational/simplify.h"

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
        auto node = std::make_shared<internal::LazyNode>(
            internal::LazyOp::ADD,
            std::vector<std::shared_ptr<const Rational>>{
            std::make_shared<Rational>(a),
                std::make_shared<Rational>(b)
        },
            Rational()  // default precision (ignored for arithmetic)
        );
        Rational result(node);
        return internal::simplify(result);
    }

    inline Rational operator-(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode) {
            return internal::eager_sub(a, b);
        }
        auto node = std::make_shared<internal::LazyNode>(
            internal::LazyOp::SUB,
            std::vector<std::shared_ptr<const Rational>>{
            std::make_shared<Rational>(a),
                std::make_shared<Rational>(b)
        }
        );
        Rational result(node);
        return internal::simplify(result);
    }

    inline Rational operator*(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode) {
            return internal::eager_mul(a, b);
        }
        auto node = std::make_shared<internal::LazyNode>(
            internal::LazyOp::MUL,
            std::vector<std::shared_ptr<const Rational>>{
            std::make_shared<Rational>(a),
                std::make_shared<Rational>(b)
        }
        );
        Rational result(node);
        return internal::simplify(result);
    }

    inline Rational operator/(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode) {
            return internal::eager_div(a, b);
        }
        auto node = std::make_shared<internal::LazyNode>(
            internal::LazyOp::DIV,
            std::vector<std::shared_ptr<const Rational>>{
            std::make_shared<Rational>(a),
                std::make_shared<Rational>(b)
        }
        );
        Rational result(node);
        return internal::simplify(result);
    }

    inline Rational operator-(const Rational& a) {
        if (internal::global_eager_mode) {
            return internal::eager_neg(a);
        }
        auto node = std::make_shared<internal::LazyNode>(
            internal::LazyOp::NEG,
            std::vector<std::shared_ptr<const Rational>>{
            std::make_shared<Rational>(a)
        }
        );
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