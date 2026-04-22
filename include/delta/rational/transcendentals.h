// transcendentals.h
#pragma once

#include "rational_class.h"
#include "lazy_rational.h"
#include "context.h"

namespace delta {

    // ----------------------------------------------------------------------------
    // Eager версии (возвращают Rational)
    // ----------------------------------------------------------------------------
    inline Rational sqrt(const Rational& x, const Rational& eps = default_eps()) {
        return eager_sqrt(x, eps);
    }
    inline Rational exp(const Rational& x, const Rational& eps = default_eps()) {
        return eager_exp(x, eps);
    }
    inline Rational log(const Rational& x, const Rational& eps = default_eps()) {
        return eager_log(x, eps);
    }
    inline Rational sin(const Rational& x, const Rational& eps = default_eps()) {
        return eager_sin(x, eps);
    }
    inline Rational cos(const Rational& x, const Rational& eps = default_eps()) {
        return eager_cos(x, eps);
    }
    inline Rational acos(const Rational& x, const Rational& eps = default_eps()) {
        return eager_acos(x, eps);
    }
    inline Rational pi(const Rational& eps = default_eps()) {
        return eager_pi(eps);
    }
    inline Rational e(const Rational& eps = default_eps()) {
        return eager_e(eps);
    }
    inline Rational pow(const Rational& base, const Rational& exponent, const Rational& eps = default_eps()) {
        return eager_pow(base, exponent, eps);
    }
    inline Rational pow(const Rational& base, int exponent) {
        if (exponent == 0) return Rational(1);
        if (exponent < 0) {
            Rational pos = pow(base, -exponent);
            return 1 / pos;
        }
        Rational result = Rational(1);
        Rational b = base;
        int e = exponent;
        while (e > 0) {
            if (e & 1) result *= b;
            e >>= 1;
            if (e != 0) b = b * b;
        }
        return result;
    }

    // ----------------------------------------------------------------------------
    // Lazy версии (возвращают LazyRational) с LazyRational аргументом
    // ----------------------------------------------------------------------------
    inline LazyRational lazy_sqrt(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::SQRT, { child }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_exp(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::EXP, { child }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_log(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::LOG, { child }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_sin(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::SIN, { child }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_cos(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::COS, { child }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_acos(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::ACOS, { child }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_pow(const LazyRational& base, const LazyRational& exponent, const Rational& eps = default_eps()) {
        LazyRational result = base.clone();
        result.ensure_dirty();
        int exp_root = result.import_tree(exponent);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::POW, { result.root_, exp_root }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    // ----------------------------------------------------------------------------
    // Lazy версии с аргументом Rational (без лишнего создания LazyRational)
    // ----------------------------------------------------------------------------
    inline LazyRational lazy_sqrt(const Rational& x, const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int child_const = result.add_constant(x.value());
        int child_node = result.new_dirty_node(internal::LazyOp::CONST, {}, child_const, -1);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::SQRT, { child_node }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_exp(const Rational& x, const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int child_const = result.add_constant(x.value());
        int child_node = result.new_dirty_node(internal::LazyOp::CONST, {}, child_const, -1);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::EXP, { child_node }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_log(const Rational& x, const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int child_const = result.add_constant(x.value());
        int child_node = result.new_dirty_node(internal::LazyOp::CONST, {}, child_const, -1);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::LOG, { child_node }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_sin(const Rational& x, const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int child_const = result.add_constant(x.value());
        int child_node = result.new_dirty_node(internal::LazyOp::CONST, {}, child_const, -1);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::SIN, { child_node }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_cos(const Rational& x, const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int child_const = result.add_constant(x.value());
        int child_node = result.new_dirty_node(internal::LazyOp::CONST, {}, child_const, -1);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::COS, { child_node }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_acos(const Rational& x, const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int child_const = result.add_constant(x.value());
        int child_node = result.new_dirty_node(internal::LazyOp::CONST, {}, child_const, -1);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::ACOS, { child_node }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_pow(const Rational& base, const LazyRational& exponent, const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int base_const = result.add_constant(base.value());
        int base_node = result.new_dirty_node(internal::LazyOp::CONST, {}, base_const, -1);
        int exp_root = result.import_tree(exponent);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::POW, { base_node, exp_root }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_pow(const Rational& base, const Rational& exponent, const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int base_const = result.add_constant(base.value());
        int base_node = result.new_dirty_node(internal::LazyOp::CONST, {}, base_const, -1);
        int exp_const = result.add_constant(exponent.value());
        int exp_node = result.new_dirty_node(internal::LazyOp::CONST, {}, exp_const, -1);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::POW, { base_node, exp_node }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_pow(const LazyRational& base, const Rational& exponent, const Rational& eps = default_eps()) {
        LazyRational result = base.clone();
        result.ensure_dirty();
        int exp_const = result.add_constant(exponent.value());
        int exp_node = result.new_dirty_node(internal::LazyOp::CONST, {}, exp_const, -1);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::POW, { result.root_, exp_node }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_pow(const LazyRational& base, int exponent) {
        return lazy_pow(base, Rational(exponent), default_eps());
    }

    // Статические фабрики для констант
    inline LazyRational lazy_pi(const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::PI, {}, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_e(const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::E, {}, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    // ----------------------------------------------------------------------------
    // Удобные короткие имена для lazy-вычислений (заглавные буквы)
    // ----------------------------------------------------------------------------
    inline LazyRational Sqrt(const LazyRational& x, const Rational& eps = default_eps()) {
        return lazy_sqrt(x, eps);
    }
    inline LazyRational Sqrt(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_sqrt(x, eps);
    }

    inline LazyRational Exp(const LazyRational& x, const Rational& eps = default_eps()) {
        return lazy_exp(x, eps);
    }
    inline LazyRational Exp(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_exp(x, eps);
    }

    inline LazyRational Log(const LazyRational& x, const Rational& eps = default_eps()) {
        return lazy_log(x, eps);
    }
    inline LazyRational Log(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_log(x, eps);
    }

    inline LazyRational Sin(const LazyRational& x, const Rational& eps = default_eps()) {
        return lazy_sin(x, eps);
    }
    inline LazyRational Sin(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_sin(x, eps);
    }

    inline LazyRational Cos(const LazyRational& x, const Rational& eps = default_eps()) {
        return lazy_cos(x, eps);
    }
    inline LazyRational Cos(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_cos(x, eps);
    }

    inline LazyRational Acos(const LazyRational& x, const Rational& eps = default_eps()) {
        return lazy_acos(x, eps);
    }
    inline LazyRational Acos(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_acos(x, eps);
    }

    inline LazyRational Pi(const Rational& eps = default_eps()) {
        return lazy_pi(eps);
    }

    inline LazyRational E(const Rational& eps = default_eps()) {
        return lazy_e(eps);
    }

    inline LazyRational Pow(const LazyRational& base, const LazyRational& exponent, const Rational& eps = default_eps()) {
        return lazy_pow(base, exponent, eps);
    }
    inline LazyRational Pow(const Rational& base, const LazyRational& exponent, const Rational& eps = default_eps()) {
        return lazy_pow(base, exponent, eps);
    }
    inline LazyRational Pow(const LazyRational& base, const Rational& exponent, const Rational& eps = default_eps()) {
        return lazy_pow(base, exponent, eps);
    }
    inline LazyRational Pow(const Rational& base, const Rational& exponent, const Rational& eps = default_eps()) {
        return lazy_pow(base, exponent, eps);
    }
    inline LazyRational Pow(const LazyRational& base, int exponent) {
        return lazy_pow(base, exponent);
    }

} // namespace delta