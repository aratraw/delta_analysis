// expression_root.h
#pragma once

#include "rational_fwd.h"
#include "node_pool.h"
#include "evaluate_impl.h"
#include "simplify_impl.h"

namespace delta {

    class ExpressionRoot {
        int root_idx_;

    public:
        explicit ExpressionRoot(int idx) : root_idx_(idx) {}
        int root_index() const { return root_idx_; }

        int depth() const {
            return internal::pool.nodes[root_idx_].depth;
        }

        uint64_t hash() const {
            return internal::pool.nodes[root_idx_].hash;
        }

        // ------------------------------------------------------------------------
        // Фабричные методы (создают узлы через пул)
        // ------------------------------------------------------------------------
        static ExpressionRoot make_const(const internal::Value& v) {
            return ExpressionRoot(internal::add_const(v));
        }

        // Унарная операция без eps (NEG, RECIP)
        static ExpressionRoot make_unary(internal::LazyOp op, const ExpressionRoot& child) {
            int child_idx = child.root_index();
            int node_idx = internal::get_unary_node(op, child_idx, -1);
            return ExpressionRoot(node_idx);
        }

        // Унарная операция с eps (трансцендентные)
        static ExpressionRoot make_unary(internal::LazyOp op, const ExpressionRoot& child,
            const internal::Value& eps) {
            int val_idx = internal::pool.add_value(eps);
            int child_idx = child.root_index();
            int node_idx = internal::get_unary_node(op, child_idx, val_idx);
            return ExpressionRoot(node_idx);
        }

        static ExpressionRoot make_binary(internal::LazyOp op, const ExpressionRoot& left,
            const ExpressionRoot& right) {
            int left_idx = left.root_index();
            int right_idx = right.root_index();
            int node_idx = internal::get_binary_node(op, left_idx, right_idx);
            return ExpressionRoot(node_idx);
        }

        //generally for POW
        static ExpressionRoot make_binary_with_eps(internal::LazyOp op, const ExpressionRoot& left,
            const ExpressionRoot& right, const internal::Value& eps) {
            int val_idx = internal::pool.add_value(eps);
            int left_idx = left.root_index();
            int right_idx = right.root_index();
            int node_idx = internal::get_binary_node(op, left_idx, right_idx, val_idx);
            return ExpressionRoot(node_idx);
        }
        // ------------------------------------------------------------------------
        // Арифметические операции
        // ------------------------------------------------------------------------
        ExpressionRoot add(const ExpressionRoot& other) const {
            return make_binary(internal::LazyOp::ADD, *this, other);
        }

        ExpressionRoot mul(const ExpressionRoot& other) const {
            return make_binary(internal::LazyOp::MUL, *this, other);
        }

        ExpressionRoot sub(const ExpressionRoot& other) const {
            return add(other.neg());
        }

        ExpressionRoot div(const ExpressionRoot& other) const {
            return mul(other.recip());
        }

        ExpressionRoot neg() const {
            return make_unary(internal::LazyOp::NEG, *this);
        }

        ExpressionRoot recip() const {
            return make_unary(internal::LazyOp::RECIP, *this);
        }

        ExpressionRoot pow(const ExpressionRoot& exponent, const Rational& eps) const;
        // ------------------------------------------------------------------------
        // Трансцендентные функции и константы (объявления)
        // ------------------------------------------------------------------------
        ExpressionRoot sqrt(const Rational& eps) const;
        ExpressionRoot exp(const Rational& eps) const;
        ExpressionRoot log(const Rational& eps) const;
        ExpressionRoot sin(const Rational& eps) const;
        ExpressionRoot cos(const Rational& eps) const;
        ExpressionRoot acos(const Rational& eps) const;
 

        static ExpressionRoot pi(const Rational& eps);
        static ExpressionRoot e(const Rational& eps);

        // ------------------------------------------------------------------------
        // Вычисление и упрощение
        // ------------------------------------------------------------------------
        internal::Value eval() const {
            return internal::evaluate(root_idx_);
        }

        ExpressionRoot simplify() const {
            int new_root = internal::simplify_impl(root_idx_);
            return ExpressionRoot(new_root);
        }
    };

} // namespace delta