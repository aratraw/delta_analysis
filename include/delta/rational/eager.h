#pragma once

#include "rational_class.h"
#include "evaluation_core.h"

namespace delta::internal {

    inline Rational eager_add(const Rational& a, const Rational& b) {
        Value va = a.eval();
        Value vb = b.eval();
        Value res = eager_add(va, vb);
        return Rational(res);
    }

    inline Rational eager_sub(const Rational& a, const Rational& b) {
        Value va = a.eval();
        Value vb = b.eval();
        Value res = eager_sub(va, vb);
        return Rational(res);
    }

    inline Rational eager_mul(const Rational& a, const Rational& b) {
        Value va = a.eval();
        Value vb = b.eval();
        Value res = eager_mul(va, vb);
        return Rational(res);
    }

    inline Rational eager_div(const Rational& a, const Rational& b) {
        Value va = a.eval();
        Value vb = b.eval();
        Value res = eager_div(va, vb);
        return Rational(res);
    }

    inline Rational eager_neg(const Rational& a) {
        Value va = a.eval();
        Value res = eager_neg(va);
        return Rational(res);
    }

    inline Rational eager_sqrt(const Rational& x, const Rational& eps) {
        Value vx = x.eval();
        Value veps = eps.eval();
        Value res = eager_sqrt(vx, veps);
        return Rational(res);
    }

    inline Rational eager_exp(const Rational& x, const Rational& eps) {
        Value vx = x.eval();
        Value veps = eps.eval();
        Value res = eager_exp(vx, veps);
        return Rational(res);
    }

    inline Rational eager_log(const Rational& x, const Rational& eps) {
        Value vx = x.eval();
        Value veps = eps.eval();
        Value res = eager_log(vx, veps);
        return Rational(res);
    }

    inline Rational eager_sin(const Rational& x, const Rational& eps) {
        Value vx = x.eval();
        Value veps = eps.eval();
        Value res = eager_sin(vx, veps);
        return Rational(res);
    }

    inline Rational eager_cos(const Rational& x, const Rational& eps) {
        Value vx = x.eval();
        Value veps = eps.eval();
        Value res = eager_cos(vx, veps);
        return Rational(res);
    }

    inline Rational eager_acos(const Rational& x, const Rational& eps) {
        Value vx = x.eval();
        Value veps = eps.eval();
        Value res = eager_acos(vx, veps);
        return Rational(res);
    }

    inline Rational eager_pi(const Rational& eps) {
        Value veps = eps.eval();
        Value res = eager_pi(veps);
        return Rational(res);
    }

    inline Rational eager_e(const Rational& eps) {
        Value veps = eps.eval();
        Value res = eager_e(veps);
        return Rational(res);
    }

} // namespace delta::internal