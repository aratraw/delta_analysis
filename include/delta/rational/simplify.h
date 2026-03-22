// include/delta/rational/simplify.h
#pragma once

#include "delta/rational/rational_class.h"
#include "delta/rational/storage.h"
#include "delta/rational/evaluation.h"   // for eager_* functions (constant folding)
#include <algorithm>
#include <cmath>
#include <limits>

namespace delta::internal {

    // -------------------------------------------------------------------------
    // Helper: get interval from a Rational (may be lazy)
    // -------------------------------------------------------------------------
    inline Interval get_interval(const Rational& r) {
        return r.approx_interval();
    }

    // -------------------------------------------------------------------------
    // Helper: check if a rational is zero (non‑lazy only)
    // -------------------------------------------------------------------------
    inline bool is_exact_zero(const Rational& r) {
        if (r.is_small()) {
            const auto& s = *r.as_small();
            return s.num == 0;
        }
        if (r.is_big()) {
            const auto& b = *r.as_big();
            return b.num == 0;
        }
        return false;
    }

    // -------------------------------------------------------------------------
    // Helper: check if a rational is one (non‑lazy only)
    // -------------------------------------------------------------------------
    inline bool is_exact_one(const Rational& r) {
        if (r.is_small()) {
            SmallStorage s = *r.as_small();
            s.normalize();
            return s.num == 1 && s.den == 1;
        }
        if (r.is_big()) {
            const auto& b = *r.as_big();
            return b.num == 1 && b.den == 1;
        }
        return false;
    }

    // -------------------------------------------------------------------------
    // Helper: check if two rationals are structurally equal (for cancellation)
    // -------------------------------------------------------------------------
    inline bool structurally_equal(const Rational& a, const Rational& b) {
        if (&a == &b) return true;
        if (a.is_lazy() && b.is_lazy()) {
            return structurally_equal(*a.as_lazy(), *b.as_lazy());
        }
        // If one is lazy and the other is not, they can't be structurally equal.
        // If both are non‑lazy, we could check exact values, but for simplification
        // we only need to catch cases like x - x. For that, we need identity,
        // not value equality. So we use pointer equality.
        return false;
    }

    // -------------------------------------------------------------------------
    // Interval computation for a lazy node (robust, using interval arithmetic)
    // -------------------------------------------------------------------------
    inline Interval compute_approx(LazyOp op, const std::vector<std::shared_ptr<const Rational>>& args) {
        switch (op) {
        case LazyOp::ADD: {
            Interval ia = get_interval(*args[0]);
            Interval ib = get_interval(*args[1]);
            return ia + ib;
        }
        case LazyOp::SUB: {
            Interval ia = get_interval(*args[0]);
            Interval ib = get_interval(*args[1]);
            return ia - ib;
        }
        case LazyOp::MUL: {
            Interval ia = get_interval(*args[0]);
            Interval ib = get_interval(*args[1]);
            return ia * ib;
        }
        case LazyOp::DIV: {
            Interval ia = get_interval(*args[0]);
            Interval ib = get_interval(*args[1]);
            return ia / ib;
        }
        case LazyOp::NEG: {
            Interval ia = get_interval(*args[0]);
            return -ia;
        }
        case LazyOp::SQRT: {
            Interval ia = get_interval(*args[0]);
            double l = std::sqrt(ia.lower());
            double h = std::sqrt(ia.upper());
            if (l > h) std::swap(l, h);
            // Expand outward by one ulp
            l = std::nextafter(l, -std::numeric_limits<double>::infinity());
            h = std::nextafter(h, std::numeric_limits<double>::infinity());
            return Interval(l, h);
        }
        case LazyOp::EXP: {
            Interval ia = get_interval(*args[0]);
            double l = std::exp(ia.lower());
            double h = std::exp(ia.upper());
            if (l > h) std::swap(l, h);
            l = std::nextafter(l, -std::numeric_limits<double>::infinity());
            h = std::nextafter(h, std::numeric_limits<double>::infinity());
            return Interval(l, h);
        }
        case LazyOp::LOG: {
            Interval ia = get_interval(*args[0]);
            // log is monotonic for positive arguments
            double l = std::log(ia.lower());
            double h = std::log(ia.upper());
            if (l > h) std::swap(l, h);
            l = std::nextafter(l, -std::numeric_limits<double>::infinity());
            h = std::nextafter(h, std::numeric_limits<double>::infinity());
            return Interval(l, h);
        }
        case LazyOp::SIN: {
            Interval ia = get_interval(*args[0]);
            // sin is periodic and non‑monotonic; we need to find min and max over the interval.
            // For simplicity, we sample at the endpoints and at points where derivative is zero
            // (multiples of π/2) that lie within the interval.
            double lo = ia.lower();
            double hi = ia.upper();
            double pi = 3.141592653589793;
            double min_val = std::sin(lo);
            double max_val = std::sin(hi);
            if (min_val > max_val) std::swap(min_val, max_val);
            // Find the nearest multiples of π/2 within [lo, hi]
            double start = std::floor(lo / (pi / 2)) * (pi / 2);
            for (double t = start; t <= hi + 1e-12; t += pi / 2) {
                if (t >= lo && t <= hi) {
                    double val = std::sin(t);
                    if (val < min_val) min_val = val;
                    if (val > max_val) max_val = val;
                }
            }
            // Expand by one ulp
            min_val = std::nextafter(min_val, -std::numeric_limits<double>::infinity());
            max_val = std::nextafter(max_val, std::numeric_limits<double>::infinity());
            return Interval(min_val, max_val);
        }
        case LazyOp::COS: {
            Interval ia = get_interval(*args[0]);
            double lo = ia.lower();
            double hi = ia.upper();
            double pi = 3.141592653589793;
            double min_val = std::cos(lo);
            double max_val = std::cos(hi);
            if (min_val > max_val) std::swap(min_val, max_val);
            double start = std::floor(lo / (pi / 2)) * (pi / 2);
            for (double t = start; t <= hi + 1e-12; t += pi / 2) {
                if (t >= lo && t <= hi) {
                    double val = std::cos(t);
                    if (val < min_val) min_val = val;
                    if (val > max_val) max_val = val;
                }
            }
            min_val = std::nextafter(min_val, -std::numeric_limits<double>::infinity());
            max_val = std::nextafter(max_val, std::numeric_limits<double>::infinity());
            return Interval(min_val, max_val);
        }
        case LazyOp::ACOS: {
            Interval ia = get_interval(*args[0]);
            // acos is decreasing on [-1,1]
            double l = std::acos(ia.upper()); // note: because decreasing
            double h = std::acos(ia.lower());
            if (l > h) std::swap(l, h);
            l = std::nextafter(l, -std::numeric_limits<double>::infinity());
            h = std::nextafter(h, std::numeric_limits<double>::infinity());
            return Interval(l, h);
        }
        case LazyOp::PI:
            return Interval(3.141592653589793, 3.141592653589793);
        case LazyOp::E:
            return Interval(2.718281828459045, 2.718281828459045);
        default:
            return Interval::zero();
        }
    }

    // -------------------------------------------------------------------------
    // Symbolic simplification of lazy nodes
    // -------------------------------------------------------------------------
    inline Rational simplify(const Rational& r) {
        if (!r.is_lazy()) return r;
        auto node = r.as_lazy();

        // First, simplify the arguments recursively
        std::vector<std::shared_ptr<const Rational>> new_args;
        bool changed = false;
        for (const auto& arg : node->args) {
            Rational simp = simplify(*arg);
            if (&simp != arg.get()) changed = true;
            new_args.push_back(std::make_shared<Rational>(std::move(simp)));
        }

        // If any argument changed, create a new node (with same op and precision)
        // and continue simplification (maybe more rules apply).
        if (changed) {
            auto new_node = std::make_shared<LazyNode>(node->op, std::move(new_args), node->precision);
            return simplify(Rational(new_node));
        }

        // Now apply algebraic rules
        auto& a = new_args[0];
        auto& b = (new_args.size() > 1) ? new_args[1] : nullptr;

        switch (node->op) {
        case LazyOp::NEG: {
            // -(-x) -> x
            if (a->is_lazy()) {
                const auto& sub = *a->as_lazy();
                if (sub.op == LazyOp::NEG && sub.args.size() == 1) {
                    return *sub.args[0];
                }
            }
            break;
        }

        case LazyOp::ADD: {
            // 0 + x -> x
            if (is_exact_zero(*a)) return *b;
            if (is_exact_zero(*b)) return *a;
            // x + x -> 2*x (optional)
            // not required but could help
            break;
        }

        case LazyOp::SUB: {
            // x - 0 -> x
            if (is_exact_zero(*b)) return *a;
            // 0 - x -> -x
            if (is_exact_zero(*a)) return -(*b);
            // x - x -> 0
            if (structurally_equal(*a, *b)) return Rational(0);
            break;
        }

        case LazyOp::MUL: {
            // 0 * x -> 0
            if (is_exact_zero(*a) || is_exact_zero(*b)) return Rational(0);
            // 1 * x -> x
            if (is_exact_one(*a)) return *b;
            if (is_exact_one(*b)) return *a;
            break;
        }

        case LazyOp::DIV: {
            // x / 1 -> x
            if (is_exact_one(*b)) return *a;
            // 0 / x -> 0 (if x != 0)
            if (is_exact_zero(*a)) return Rational(0);
            // x / x -> 1
            if (structurally_equal(*a, *b)) return Rational(1);
            break;
        }

        case LazyOp::SQRT: {
            // sqrt(0) -> 0
            if (is_exact_zero(*a)) return Rational(0);
            // sqrt(1) -> 1
            if (is_exact_one(*a)) return Rational(1);
            // sqrt(exp(x)) -> exp(x/2)  (optional)
            if (a->is_lazy() && a->as_lazy()->op == LazyOp::EXP) {
                const auto& exp_args = a->as_lazy()->args;
                Rational half = Rational(1, 2);
                Rational new_arg = *exp_args[0] * half;
                auto new_node = std::make_shared<LazyNode>(LazyOp::EXP, new_arg, node->precision);
                return simplify(Rational(new_node));
            }
            break;
        }

        case LazyOp::EXP: {
            // exp(0) -> 1
            if (is_exact_zero(*a)) return Rational(1);
            // exp(ln(x)) -> x
            if (a->is_lazy() && a->as_lazy()->op == LazyOp::LOG) {
                const auto& log_args = a->as_lazy()->args;
                return *log_args[0];
            }
            break;
        }

        case LazyOp::LOG: {
            // log(1) -> 0
            if (is_exact_one(*a)) return Rational(0);
            // log(exp(x)) -> x
            if (a->is_lazy() && a->as_lazy()->op == LazyOp::EXP) {
                const auto& exp_args = a->as_lazy()->args;
                return *exp_args[0];
            }
            break;
        }

        case LazyOp::SIN: {
            // sin(0) -> 0
            if (is_exact_zero(*a)) return Rational(0);
            // sin(pi) -> 0
            // Would need pi constant. Not implemented.
            break;
        }

        case LazyOp::COS: {
            // cos(0) -> 1
            if (is_exact_zero(*a)) return Rational(1);
            break;
        }

        case LazyOp::ACOS: {
            // acos(1) -> 0
            if (is_exact_one(*a)) return Rational(0);
            // acos(0) -> pi/2
            if (is_exact_zero(*a)) {
                Rational pi_half = eager_pi(node->precision) / Rational(2);
                return pi_half;
            }
            break;
        }

        default:
            break;
        }

        // If we can't simplify further, we may constant-fold if all arguments are non-lazy.
        bool all_non_lazy = true;
        for (const auto& arg : new_args) {
            if (arg->is_lazy()) { all_non_lazy = false; break; }
        }
        if (all_non_lazy) {
            // All arguments are computed numbers; we can compute the result eagerly.
            // This avoids building deep trees for constants.
            // We'll use the eager functions from evaluation.h.
            switch (node->op) {
            case LazyOp::ADD: return eager_add(*a, *b);
            case LazyOp::SUB: return eager_sub(*a, *b);
            case LazyOp::MUL: return eager_mul(*a, *b);
            case LazyOp::DIV: return eager_div(*a, *b);
            case LazyOp::NEG: return eager_neg(*a);
            case LazyOp::SQRT: return eager_sqrt(*a, node->precision);
            case LazyOp::EXP:  return eager_exp(*a, node->precision);
            case LazyOp::LOG:  return eager_log(*a, node->precision);
            case LazyOp::SIN:  return eager_sin(*a, node->precision);
            case LazyOp::COS:  return eager_cos(*a, node->precision);
            case LazyOp::ACOS: return eager_acos(*a, node->precision);
            case LazyOp::PI:   return eager_pi(node->precision);
            case LazyOp::E:    return eager_e(node->precision);
            default: break;
            }
        }

        // No simplification, return the (possibly modified) node
        auto new_node = std::make_shared<LazyNode>(node->op, std::move(new_args), node->precision);
        return Rational(new_node);
    }

} // namespace delta::internal