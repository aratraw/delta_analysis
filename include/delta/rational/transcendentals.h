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
            return eager_div(Rational(1), pos);
        }
        Rational result = Rational(1);
        Rational b = base;
        int e = exponent;
        while (e > 0) {
            if (e & 1) result = eager_mul(result, b);
            e >>= 1;
            if (e != 0) b = eager_mul(b, b);
        }
        return result;
    }

    // ----------------------------------------------------------------------------
    // Lazy версии (возвращают LazyRational)
    // ----------------------------------------------------------------------------
    // Базовые перегрузки, принимающие LazyRational
    inline LazyRational lazy_sqrt(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::SQRT, { child }, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_exp(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::EXP, { child }, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_log(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::LOG, { child }, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_sin(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::SIN, { child }, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_cos(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::COS, { child }, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_acos(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::ACOS, { child }, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_pi(const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::PI, {}, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_e(const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::E, {}, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_pow(const LazyRational& base, const LazyRational& exponent, const Rational& eps = default_eps()) {
        LazyRational result = base.clone();
        result.ensure_dirty();
        int exp_root = result.import_tree(exponent);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::POW, { result.root_, exp_root }, eps_idx);
        result.root_ = node;
        return result;
    }

    // ----------------------------------------------------------------------------
    // Дополнительные перегрузки для удобства: принимают Rational, преобразуют в LazyRational
    // ----------------------------------------------------------------------------
    inline LazyRational lazy_sqrt(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_sqrt(LazyRational(x), eps);
    }
    inline LazyRational lazy_exp(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_exp(LazyRational(x), eps);
    }
    inline LazyRational lazy_log(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_log(LazyRational(x), eps);
    }
    inline LazyRational lazy_sin(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_sin(LazyRational(x), eps);
    }
    inline LazyRational lazy_cos(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_cos(LazyRational(x), eps);
    }
    inline LazyRational lazy_acos(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_acos(LazyRational(x), eps);
    }
    inline LazyRational lazy_pow(const Rational& base, const LazyRational& exponent, const Rational& eps = default_eps()) {
        return lazy_pow(LazyRational(base), exponent, eps);
    }
    inline LazyRational lazy_pow(const Rational& base, const Rational& exponent, const Rational& eps = default_eps()) {
        return lazy_pow(LazyRational(base), LazyRational(exponent), eps);
    }
    inline LazyRational lazy_pow(const LazyRational& base, const Rational& exponent, const Rational& eps = default_eps()) {
        return lazy_pow(base, LazyRational(exponent), eps);
    }
    inline LazyRational lazy_pow(const LazyRational& base, int exponent) {
        // Просто конвертируем int -> Rational и используем дефолтную точность.
        // Общая логика lazy_pow сама создаст узел POW с корректным eps_idx,
        // а eager_pow при вычислении сам обработает отрицательный показатель.
        return lazy_pow(base, Rational(exponent), default_eps());
    }

} // namespace delta