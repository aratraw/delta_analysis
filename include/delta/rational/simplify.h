// include/delta/rational/simplify.h
#pragma once

#include "delta/rational/rational_class.h"
#include "delta/rational/storage.h"
#include "delta/rational/evaluation.h"
#include "delta/rational/context.h"
#include "delta/rational/batch_arithmetic.h"
#include <absl/container/inlined_vector.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace delta::internal {

    // -------------------------------------------------------------------------
    // Helper: get interval from a Rational (may be lazy)
    // -------------------------------------------------------------------------
    inline Interval get_interval(const Rational& r) {
        return r.approx_interval();
    }

    // -------------------------------------------------------------------------
    // Interval computation for a lazy node (robust, using interval arithmetic)
    // -------------------------------------------------------------------------
    inline Interval compute_approx(LazyOp op, const absl::InlinedVector<std::shared_ptr<const Rational>, 2>& args) {
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
            double l = std::log(ia.lower());
            double h = std::log(ia.upper());
            if (l > h) std::swap(l, h);
            l = std::nextafter(l, -std::numeric_limits<double>::infinity());
            h = std::nextafter(h, std::numeric_limits<double>::infinity());
            return Interval(l, h);
        }
        case LazyOp::SIN: {
            Interval ia = get_interval(*args[0]);
            double lo = ia.lower();
            double hi = ia.upper();
            double pi = 3.141592653589793;
            double min_val = std::sin(lo);
            double max_val = std::sin(hi);
            if (min_val > max_val) std::swap(min_val, max_val);
            double start = std::floor(lo / (pi / 2)) * (pi / 2);
            for (double t = start; t <= hi + 1e-12; t += pi / 2) {
                if (t >= lo && t <= hi) {
                    double val = std::sin(t);
                    if (val < min_val) min_val = val;
                    if (val > max_val) max_val = val;
                }
            }
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
            double l = std::acos(ia.upper());
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
    // Definitions of LazyNode constructors (now using shared_ptr for precision)
    // -------------------------------------------------------------------------
    inline LazyNode::LazyNode(LazyOp o, absl::InlinedVector<std::shared_ptr<const Rational>, 2>&& a, std::shared_ptr<const Rational> eps)
        : op(o), args(std::move(a)), precision(eps), cached_value(std::nullopt),
        approx(compute_approx(op, this->args)), depth(0) {
        depth = 1;
        for (const auto& arg : args) {
            if (arg && arg->is_lazy()) {
                int d = arg->as_lazy()->depth;
                if (d + 1 > depth) depth = d + 1;
            }
        }
    }

    inline LazyNode::LazyNode(LazyOp o, std::shared_ptr<const Rational> a, std::shared_ptr<const Rational> eps)
        : LazyNode(o, absl::InlinedVector<std::shared_ptr<const Rational>, 2>{std::move(a)}, eps) {
    }

    inline LazyNode::LazyNode(LazyOp o, std::shared_ptr<const Rational> eps)
        : op(o), args(), precision(eps), cached_value(std::nullopt),
        approx(compute_approx(o, args)), depth(1) {
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
    // Structural equality for lazy nodes
    // -------------------------------------------------------------------------
    inline bool structurally_equal(const LazyNode& a, const LazyNode& b) {
        if (&a == &b) return true;
        if (a.op != b.op) return false;
        if (a.args.size() != b.args.size()) return false;
        for (size_t i = 0; i < a.args.size(); ++i) {
            if (!structurally_equal(*a.args[i], *b.args[i])) return false;
        }
        return true;
    }

    // -------------------------------------------------------------------------
    // Structural equality for Rational
    // -------------------------------------------------------------------------
    inline bool structurally_equal(const Rational& a, const Rational& b) {
        if (&a == &b) return true;
        if (a.is_lazy() && b.is_lazy()) {
            return structurally_equal(*a.as_lazy(), *b.as_lazy());
        }
        return false;
    }

    // -------------------------------------------------------------------------
    // Symbolic simplification of lazy nodes
    // -------------------------------------------------------------------------
    inline Rational simplify(const Rational& r) {
        if (!r.is_lazy()) return r;
        auto node = r.as_lazy();

        // 1. Simplify arguments recursively
        absl::InlinedVector<std::shared_ptr<const Rational>, 2> new_args;
        bool changed = false;
        for (const auto& arg : node->args) {
            Rational simp = simplify(*arg);
            if (&simp != arg.get()) changed = true;
            new_args.push_back(std::make_shared<Rational>(std::move(simp)));
        }

        if (changed) {
            auto new_node = std::make_shared<LazyNode>(node->op, std::move(new_args), node->precision);
            return simplify(Rational(new_node));
        }

        // 2. Depth overflow: force evaluation
        if (node->depth > MAX_LAZY_DEPTH) {
            Rational ev = evaluate(r);
            return ev;
        }

        // 3. Apply algebraic simplification rules
        auto& a = new_args[0];
        auto b = (new_args.size() > 1) ? new_args[1] : nullptr;   // removed reference

        switch (node->op) {
        case LazyOp::NEG: {
            if (a->is_lazy()) {
                const auto& sub = *a->as_lazy();
                if (sub.op == LazyOp::NEG && sub.args.size() == 1) {
                    return *sub.args[0];
                }
            }
            break;
        }
        case LazyOp::ADD: {
            if (is_exact_zero(*a)) return *b;
            if (is_exact_zero(*b)) return *a;
            break;
        }
        case LazyOp::SUB: {
            if (is_exact_zero(*b)) return *a;
            if (is_exact_zero(*a)) return -(*b);
            if (structurally_equal(*a, *b)) return Rational(0);
            break;
        }
        case LazyOp::MUL: {
            if (is_exact_zero(*a) || is_exact_zero(*b)) return Rational(0);
            if (is_exact_one(*a)) return *b;
            if (is_exact_one(*b)) return *a;
            break;
        }
        case LazyOp::DIV: {
            if (is_exact_one(*b)) return *a;
            if (is_exact_zero(*a)) return Rational(0);
            if (structurally_equal(*a, *b)) return Rational(1);
            break;
        }
        case LazyOp::SQRT: {
            if (is_exact_zero(*a)) return Rational(0);
            if (is_exact_one(*a)) return Rational(1);
            if (a->is_lazy() && a->as_lazy()->op == LazyOp::EXP) {
                const auto& exp_args = a->as_lazy()->args;
                Rational half = Rational(1, 2);
                Rational new_arg = *exp_args[0] * half;
                auto new_node = std::make_shared<LazyNode>(LazyOp::EXP, std::make_shared<Rational>(new_arg), node->precision);
                return simplify(Rational(new_node));
            }
            break;
        }
        case LazyOp::EXP: {
            if (is_exact_zero(*a)) return Rational(1);
            if (a->is_lazy() && a->as_lazy()->op == LazyOp::LOG) {
                const auto& log_args = a->as_lazy()->args;
                return *log_args[0];
            }
            break;
        }
        case LazyOp::LOG: {
            if (is_exact_one(*a)) return Rational(0);
            if (a->is_lazy() && a->as_lazy()->op == LazyOp::EXP) {
                const auto& exp_args = a->as_lazy()->args;
                return *exp_args[0];
            }
            break;
        }
        case LazyOp::SIN: {
            if (is_exact_zero(*a)) return Rational(0);
            break;
        }
        case LazyOp::COS: {
            if (is_exact_zero(*a)) return Rational(1);
            break;
        }
        case LazyOp::ACOS: {
            if (is_exact_one(*a)) return Rational(0);
            if (is_exact_zero(*a)) {
                Rational pi_half = eager_pi(*node->precision) / Rational(2);
                return pi_half;
            }
            break;
        }
        default: break;
        }

        // 4. Batch addition
        if (node->op == LazyOp::ADD && new_args.size() >= 2) {
            std::vector<Rational> terms;
            bool all_non_lazy = true;
            for (const auto& arg : new_args) {
                if (arg->is_lazy()) {
                    all_non_lazy = false;
                    break;
                }
                terms.push_back(*arg);
            }
            if (all_non_lazy) {
                return batch_add(terms);
            }
        }

        // 5. Constant folding
        bool all_non_lazy = true;
        for (const auto& arg : new_args) {
            if (arg->is_lazy()) { all_non_lazy = false; break; }
        }
        if (all_non_lazy) {
            switch (node->op) {
            case LazyOp::ADD: return eager_add(*a, *b);
            case LazyOp::SUB: return eager_sub(*a, *b);
            case LazyOp::MUL: return eager_mul(*a, *b);
            case LazyOp::DIV: return eager_div(*a, *b);
            case LazyOp::NEG: return eager_neg(*a);
            case LazyOp::SQRT: return eager_sqrt(*a, *node->precision);
            case LazyOp::EXP:  return eager_exp(*a, *node->precision);
            case LazyOp::LOG:  return eager_log(*a, *node->precision);
            case LazyOp::SIN:  return eager_sin(*a, *node->precision);
            case LazyOp::COS:  return eager_cos(*a, *node->precision);
            case LazyOp::ACOS: return eager_acos(*a, *node->precision);
            case LazyOp::PI:   return eager_pi(*node->precision);
            case LazyOp::E:    return eager_e(*node->precision);
            default: break;
            }
        }

        auto new_node = std::make_shared<LazyNode>(node->op, std::move(new_args), node->precision);
        return Rational(new_node);
    }

} // namespace delta::internal