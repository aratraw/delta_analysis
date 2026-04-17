// lazy_rational_impl.h
#pragma once

#include "node_pool.h"
#include "evaluate_impl.h"
#include "gc.h"
#include "lazy_nodes.h"        // для DirtyNode, TempNode
#include "simplify_impl.h"     // для internal::simplify_tree
#include "interval.h"
#include <stack>
#include <cassert>
#include <algorithm>
#include <optional>

namespace delta {

    // ============================================================================
    // Вспомогательные функции для работы с грязным деревом (интервалы)
    // ============================================================================

    // Вычисление интервала для грязного дерева (итеративно) – теперь в пространстве delta,
    // чтобы иметь доступ к приватным полям LazyRational через дружбу
    inline internal::Interval compute_interval_dirty(const LazyRational& lr) {
        assert(lr.is_dirty());
        const auto& nodes = lr.nodes_;
        const auto& constants = lr.constants_;
        std::vector<internal::Interval> intervals(nodes.size());

        // Пост-порядный обход (стек)
        std::stack<int> st;
        st.push(lr.root_);
        std::vector<int> postorder;
        while (!st.empty()) {
            int idx = st.top(); st.pop();
            postorder.push_back(idx);
            for (int child : nodes[idx].children) {
                st.push(child);
            }
        }
        // Обрабатываем в обратном порядке
        for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
            int idx = *it;
            const auto& dn = nodes[idx];
            switch (dn.op) {
            case internal::LazyOp::CONST: {
                double val = internal::to_double(constants[dn.const_index]);
                intervals[idx] = internal::Interval(val);
                break;
            }
            case internal::LazyOp::SUM: {
                internal::Interval sum = internal::Interval::zero();
                for (int child : dn.children) sum = sum + intervals[child];
                intervals[idx] = sum;
                break;
            }
            case internal::LazyOp::PRODUCT: {
                internal::Interval prod = internal::Interval::one();
                for (int child : dn.children) prod = prod * intervals[child];
                intervals[idx] = prod;
                break;
            }
            case internal::LazyOp::NEG:
                intervals[idx] = -intervals[dn.children[0]];
                break;
            case internal::LazyOp::RECIP: {
                const auto& child_int = intervals[dn.children[0]];
                if (child_int.lower() <= 0.0 && child_int.upper() >= 0.0)
                    intervals[idx] = internal::Interval(-std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity());
                else {
                    double lo = 1.0 / child_int.upper();
                    double hi = 1.0 / child_int.lower();
                    if (lo > hi) std::swap(lo, hi);
                    intervals[idx] = internal::Interval(lo, hi);
                }
                break;
            }
            case internal::LazyOp::SQRT: {
                const auto& child_int = intervals[dn.children[0]];
                if (child_int.upper() < 0) intervals[idx] = internal::Interval();
                else {
                    double lo = child_int.lower() < 0 ? 0.0 : std::sqrt(child_int.lower());
                    double hi = std::sqrt(child_int.upper());
                    intervals[idx] = internal::Interval(lo, hi);
                }
                break;
            }
            case internal::LazyOp::EXP: {
                const auto& child_int = intervals[dn.children[0]];
                intervals[idx] = internal::Interval(std::exp(child_int.lower()), std::exp(child_int.upper()));
                break;
            }
            case internal::LazyOp::LOG: {
                const auto& child_int = intervals[dn.children[0]];
                if (child_int.upper() <= 0)
                    intervals[idx] = internal::Interval(-std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity());
                else {
                    double lo = std::log(child_int.lower());
                    double hi = std::log(child_int.upper());
                    intervals[idx] = internal::Interval(lo, hi);
                }
                break;
            }
            case internal::LazyOp::SIN:
            case internal::LazyOp::COS:
                intervals[idx] = internal::Interval(-1.0, 1.0);
                break;
            case internal::LazyOp::ACOS: {
                const auto& child_int = intervals[dn.children[0]];
                if (child_int.lower() < -1 || child_int.upper() > 1)
                    intervals[idx] = internal::Interval(-std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity());
                else {
                    double lo = std::acos(child_int.upper());
                    double hi = std::acos(child_int.lower());
                    intervals[idx] = internal::Interval(lo, hi);
                }
                break;
            }
            case internal::LazyOp::PI:
                intervals[idx] = internal::Interval(3.14159265358979323846);
                break;
            case internal::LazyOp::E:
                intervals[idx] = internal::Interval(2.71828182845904523536);
                break;
            case internal::LazyOp::POW: {
                const auto& base_int = intervals[dn.children[0]];
                const auto& exp_int = intervals[dn.children[1]];
                double lo = std::pow(base_int.lower(), exp_int.lower());
                double hi = std::pow(base_int.upper(), exp_int.upper());
                intervals[idx] = internal::Interval(lo, hi);
                break;
            }
            default:
                throw std::logic_error("compute_interval_dirty: unknown op");
            }
        }
        return intervals[lr.root_];
    }

    // ============================================================================
    // Конструкторы
    // ============================================================================
    inline LazyRational::LazyRational() : state_(State::Dirty) {
        int const_idx = add_constant(internal::Value(internal::SmallStorage(0)));
        nodes_.emplace_back(internal::LazyOp::CONST, absl::InlinedVector<int, 2>{}, const_idx);
        root_ = 0;
    }

    inline LazyRational::LazyRational(const Rational& r) : state_(State::Dirty) {
        int const_idx = add_constant(r.value());
        nodes_.emplace_back(internal::LazyOp::CONST, absl::InlinedVector<int, 2>{}, const_idx);
        root_ = 0;
    }

    inline LazyRational::LazyRational(Rational&& r) : state_(State::Dirty) {
        int const_idx = add_constant(std::move(r.value()));
        nodes_.emplace_back(internal::LazyOp::CONST, absl::InlinedVector<int, 2>{}, const_idx);
        root_ = 0;
    }

    // Перемещение
    inline LazyRational::LazyRational(LazyRational&& other) noexcept
        : state_(other.state_),
        nodes_(std::move(other.nodes_)),
        constants_(std::move(other.constants_)),
        root_(other.root_),
        clean_index_(other.clean_index_),
        cached_interval_(std::move(other.cached_interval_))
    {
        other.state_ = State::Dirty;
        other.root_ = -1;
        other.clean_index_ = -1;
        other.cached_interval_.reset();
    }

    inline LazyRational& LazyRational::operator=(LazyRational&& other) noexcept {
        if (this != &other) {
            this->~LazyRational();
            state_ = other.state_;
            nodes_ = std::move(other.nodes_);
            constants_ = std::move(other.constants_);
            root_ = other.root_;
            clean_index_ = other.clean_index_;
            cached_interval_ = std::move(other.cached_interval_);
            other.state_ = State::Dirty;
            other.root_ = -1;
            other.clean_index_ = -1;
            other.cached_interval_.reset();
        }
        return *this;
    }

    inline LazyRational::~LazyRational() {
        if (state_ == State::Clean) {
            internal::decrement_ref(clean_index_);
        }
    }

    // ============================================================================
    // import_tree (копирование дерева в грязное состояние)
    // ============================================================================
    inline int LazyRational::import_tree(const LazyRational& other) {
        assert(state_ == State::Dirty);

        // 🔧 ОБРАБОТКА SELF-IMPORT: создаём снапшот перед мутацией
        if (this == &other) {
            // Снапшот изолирует источник от мутаций
            std::vector<internal::DirtyNode> nodes_snapshot = nodes_;
            std::vector<internal::Value> constants_snapshot = constants_;
            int root_snapshot = root_;

            std::vector<int> old_to_new(nodes_snapshot.size(), -1);
            std::stack<int> st;
            st.push(root_snapshot);
            std::vector<int> postorder;

            // Пост-порядный обход снапшота
            while (!st.empty()) {
                int idx = st.top(); st.pop();
                postorder.push_back(idx);
                for (int child : nodes_snapshot[idx].children) {
                    st.push(child);
                }
            }

            // Создаём новые узлы из снапшота
            for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
                int old_idx = *it;
                const auto& old_node = nodes_snapshot[old_idx];
                absl::InlinedVector<int, 2> new_children;
                for (int child : old_node.children) {
                    new_children.push_back(old_to_new[child]);
                }
                int new_const = -1;
                if (old_node.const_index != -1) {
                    new_const = add_constant(constants_snapshot[old_node.const_index]);
                }
                int new_idx = new_dirty_node(old_node.op, std::move(new_children), new_const);
                old_to_new[old_idx] = new_idx;
            }
            return old_to_new[root_snapshot];
        }

        // Оригинальный код для обычного импорта (разные объекты)
        if (other.state_ == State::Dirty) {
            std::vector<int> old_to_new(other.nodes_.size(), -1);
            std::stack<int> st;
            st.push(other.root_);
            std::vector<int> postorder;
            while (!st.empty()) {
                int idx = st.top(); st.pop();
                postorder.push_back(idx);
                for (int child : other.nodes_[idx].children) {
                    st.push(child);
                }
            }
            for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
                int old_idx = *it;
                const auto& old_node = other.nodes_[old_idx];
                absl::InlinedVector<int, 2> new_children;
                for (int child : old_node.children) {
                    new_children.push_back(old_to_new[child]);
                }
                int new_const = -1;
                if (old_node.const_index != -1) {
                    new_const = add_constant(other.constants_[old_node.const_index]);
                }
                int new_idx = new_dirty_node(old_node.op, std::move(new_children), new_const);
                old_to_new[old_idx] = new_idx;
            }
            return old_to_new[other.root_];
        }
        else {
            // Clean -> Dirty: клонируем через временный объект
            LazyRational temp = other.clone();
            temp.ensure_dirty();
            return import_tree(temp);
        }
    }
    // ============================================================================
    // clone
    // ============================================================================
    inline LazyRational LazyRational::clone() const {
        if (state_ == State::Dirty) {
            LazyRational copy;
            copy.state_ = State::Dirty;
            copy.root_ = copy.import_tree(*this);
            return copy;
        }
        else {
            LazyRational copy;
            copy.state_ = State::Clean;
            copy.clean_index_ = clean_index_;
            internal::increment_ref(clean_index_);
            return copy;
        }
    }

    // ============================================================================
    // ensure_dirty
    // ============================================================================
    inline void LazyRational::ensure_dirty() {
        if (state_ == State::Clean) {
            invalidate_interval();  // кэш интервала сбрасывается
            LazyRational temp;
            temp.state_ = State::Dirty;
            std::stack<int> st;
            st.push(clean_index_);
            std::vector<int> postorder;
            while (!st.empty()) {
                int idx = st.top(); st.pop();
                postorder.push_back(idx);
                const auto& node = internal::pool.nodes[idx];
                if (node.children) {
                    for (int child : *node.children) st.push(child);
                }
                else {
                    if (node.child0 != -1) st.push(node.child0);
                    if (node.child1 != -1) st.push(node.child1);
                }
            }
            std::vector<int> clean_to_dirty(internal::pool.nodes.size(), -1);
            for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
                int clean_idx = *it;
                const auto& clean_node = internal::pool.nodes[clean_idx];
                int dirty_idx = -1;
                switch (clean_node.op) {
                case internal::LazyOp::CONST: {
                    int const_idx = temp.add_constant(internal::pool.values[clean_node.value_idx]);
                    dirty_idx = temp.new_dirty_node(clean_node.op, {}, const_idx);
                    break;
                }
                case internal::LazyOp::SUM:
                case internal::LazyOp::PRODUCT: {
                    absl::InlinedVector<int, 2> children;
                    if (clean_node.children) {
                        for (int child : *clean_node.children)
                            children.push_back(clean_to_dirty[child]);
                    }
                    dirty_idx = temp.new_dirty_node(clean_node.op, std::move(children));
                    break;
                }
                case internal::LazyOp::NEG:
                case internal::LazyOp::RECIP:
                case internal::LazyOp::SQRT:
                case internal::LazyOp::EXP:
                case internal::LazyOp::LOG:
                case internal::LazyOp::SIN:
                case internal::LazyOp::COS:
                case internal::LazyOp::ACOS: {
                    int child = clean_to_dirty[clean_node.child0];
                    int eps_const_idx = -1;
                    if (clean_node.eps_idx != -1)
                        eps_const_idx = temp.add_constant(internal::pool.values[clean_node.eps_idx]);
                    dirty_idx = temp.new_dirty_node(clean_node.op, { child }, eps_const_idx);
                    break;
                }
                case internal::LazyOp::PI:
                case internal::LazyOp::E: {
                    int eps_const_idx = -1;
                    if (clean_node.eps_idx != -1)
                        eps_const_idx = temp.add_constant(internal::pool.values[clean_node.eps_idx]);
                    dirty_idx = temp.new_dirty_node(clean_node.op, {}, eps_const_idx);
                    break;
                }
                case internal::LazyOp::POW: {
                    int base = clean_to_dirty[clean_node.child0];
                    int exp = clean_to_dirty[clean_node.child1];
                    int eps_const_idx = -1;
                    if (clean_node.eps_idx != -1)
                        eps_const_idx = temp.add_constant(internal::pool.values[clean_node.eps_idx]);
                    dirty_idx = temp.new_dirty_node(clean_node.op, { base, exp }, eps_const_idx);
                    break;
                }
                default:
                    throw std::logic_error("ensure_dirty: unknown LazyOp");
                }
                clean_to_dirty[clean_idx] = dirty_idx;
            }
            temp.root_ = clean_to_dirty[clean_index_];
            *this = std::move(temp);
            internal::decrement_ref(clean_index_);
        }
    }

    // ============================================================================
    // Вспомогательные методы для грязного дерева
    // ============================================================================
    inline int LazyRational::add_constant(const internal::Value& v) {
        assert(state_ == State::Dirty);
        constants_.push_back(v);
        return static_cast<int>(constants_.size() - 1);
    }

    inline int LazyRational::new_dirty_node(internal::LazyOp op, absl::InlinedVector<int, 2> children, int const_index) {
        assert(state_ == State::Dirty);
        nodes_.emplace_back(op, std::move(children), const_index);
        return static_cast<int>(nodes_.size() - 1);
    }

    inline void LazyRational::append_sum_children(int sum_node, const LazyRational& other) {
        assert(state_ == State::Dirty);
        assert(nodes_[sum_node].op == internal::LazyOp::SUM);
        int other_root = import_tree(other);
        const auto& other_root_node = nodes_[other_root];
        if (other_root_node.op == internal::LazyOp::SUM) {
            for (int child : other_root_node.children)
                nodes_[sum_node].children.push_back(child);
        }
        else {
            nodes_[sum_node].children.push_back(other_root);
        }
    }

    inline void LazyRational::append_product_children(int prod_node, const LazyRational& other) {
        assert(state_ == State::Dirty);
        assert(nodes_[prod_node].op == internal::LazyOp::PRODUCT);
        int other_root = import_tree(other);
        const auto& other_root_node = nodes_[other_root];
        if (other_root_node.op == internal::LazyOp::PRODUCT) {
            for (int child : other_root_node.children)
                nodes_[prod_node].children.push_back(child);
        }
        else {
            nodes_[prod_node].children.push_back(other_root);
        }
    }

    // ============================================================================
    // Мутирующие операторы (с инвалидацией интервала)
    // ============================================================================
    inline LazyRational& operator+(LazyRational& a, const LazyRational& b) {
        a.ensure_dirty();
        a.invalidate_interval();
        int root = a.root_;
        if (a.nodes_[root].op != internal::LazyOp::SUM) {
            int b_root = a.import_tree(b);
            int new_root = a.new_dirty_node(internal::LazyOp::SUM, { root, b_root });
            a.root_ = new_root;
        }
        else {
            a.append_sum_children(root, b);
        }
        return a;
    }

    inline LazyRational&& operator+(LazyRational&& a, const LazyRational& b) {
        return std::move(operator+(a, b));
    }

    inline LazyRational& operator+(LazyRational& a, const Rational& b) {
        LazyRational temp(b);
        return a + temp;
    }

    inline LazyRational&& operator+(LazyRational&& a, const Rational& b) {
        return std::move(operator+(a, b));
    }

    inline LazyRational& operator-(LazyRational& a, const LazyRational& b) {
        a.ensure_dirty();
        a.invalidate_interval();
        int b_root = a.import_tree(b);
        int neg_root = a.new_dirty_node(internal::LazyOp::NEG, { b_root });
        int root = a.root_;
        if (a.nodes_[root].op != internal::LazyOp::SUM) {
            int new_root = a.new_dirty_node(internal::LazyOp::SUM, { root, neg_root });
            a.root_ = new_root;
        }
        else {
            a.nodes_[root].children.push_back(neg_root);
        }
        return a;
    }

    inline LazyRational&& operator-(LazyRational&& a, const LazyRational& b) {
        return std::move(operator-(a, b));
    }

    inline LazyRational& operator-(LazyRational& a, const Rational& b) {
        LazyRational temp(b);
        return a - temp;
    }

    inline LazyRational&& operator-(LazyRational&& a, const Rational& b) {
        return std::move(operator-(a, b));
    }

    inline LazyRational operator-(const LazyRational& a) {
        LazyRational result = a.clone();
        result.ensure_dirty();
        result.invalidate_interval();
        int root = result.root_;
        int neg_root = result.new_dirty_node(internal::LazyOp::NEG, { root });
        result.root_ = neg_root;
        return result;
    }

    inline LazyRational& operator*(LazyRational& a, const LazyRational& b) {
        a.ensure_dirty();
        a.invalidate_interval();
        int root = a.root_;
        if (a.nodes_[root].op != internal::LazyOp::PRODUCT) {
            int b_root = a.import_tree(b);
            int new_root = a.new_dirty_node(internal::LazyOp::PRODUCT, { root, b_root });
            a.root_ = new_root;
        }
        else {
            a.append_product_children(root, b);
        }
        return a;
    }

    inline LazyRational&& operator*(LazyRational&& a, const LazyRational& b) {
        return std::move(operator*(a, b));
    }

    inline LazyRational& operator*(LazyRational& a, const Rational& b) {
        LazyRational temp(b);
        return a * temp;
    }

    inline LazyRational&& operator*(LazyRational&& a, const Rational& b) {
        return std::move(operator*(a, b));
    }

    inline LazyRational& operator/(LazyRational& a, const LazyRational& b) {
        a.ensure_dirty();
        a.invalidate_interval();
        int b_root = a.import_tree(b);
        int recip_root = a.new_dirty_node(internal::LazyOp::RECIP, { b_root });
        int root = a.root_;
        if (a.nodes_[root].op != internal::LazyOp::PRODUCT) {
            int new_root = a.new_dirty_node(internal::LazyOp::PRODUCT, { root, recip_root });
            a.root_ = new_root;
        }
        else {
            a.nodes_[root].children.push_back(recip_root);
        }
        return a;
    }

    inline LazyRational&& operator/(LazyRational&& a, const LazyRational& b) {
        return std::move(operator/(a, b));
    }

    inline LazyRational& operator/(LazyRational& a, const Rational& b) {
        LazyRational temp(b);
        return a / temp;
    }

    inline LazyRational&& operator/(LazyRational&& a, const Rational& b) {
        return std::move(operator/(a, b));
    }

    inline LazyRational& operator+=(LazyRational& a, const LazyRational& b) { return a + b; }
    inline LazyRational& operator+=(LazyRational& a, const Rational& b) { return a + b; }
    inline LazyRational& operator-=(LazyRational& a, const LazyRational& b) { return a - b; }
    inline LazyRational& operator-=(LazyRational& a, const Rational& b) { return a - b; }
    inline LazyRational& operator*=(LazyRational& a, const LazyRational& b) { return a * b; }
    inline LazyRational& operator*=(LazyRational& a, const Rational& b) { return a * b; }
    inline LazyRational& operator/=(LazyRational& a, const LazyRational& b) { return a / b; }
    inline LazyRational& operator/=(LazyRational& a, const Rational& b) { return a / b; }

    // ============================================================================
    // Канонизация (Dirty -> Clean) с локальным упрощением
    // ============================================================================
    inline void LazyRational::canonicalize() const {
        if (state_ != State::Dirty) return;

        // 1. Построить TempNode дерево (без упрощений)
        std::vector<internal::TempNode> temp_nodes;
        std::vector<internal::Value> temp_values;
        std::vector<int> dirty_to_temp(nodes_.size(), -1);

        // Пост-порядный обход грязного дерева
        std::stack<int> st;
        st.push(root_);
        std::vector<int> postorder;
        while (!st.empty()) {
            int idx = st.top(); st.pop();
            postorder.push_back(idx);
            for (int child : nodes_[idx].children) {
                st.push(child);
            }
        }

        for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
            int dirty_idx = *it;
            const auto& dn = nodes_[dirty_idx];
            std::vector<int> temp_children;
            for (int child : dn.children) {
                temp_children.push_back(dirty_to_temp[child]);
            }

            int value_idx = -1, eps_idx = -1;
            if (dn.const_index != -1) {
                // 🔧 ИСПРАВЛЕНИЕ: явное разделение семантики const_index
                if (dn.op == internal::LazyOp::CONST) {
                    // Для CONST: const_index указывает на ЗНАЧЕНИЕ константы
                    value_idx = static_cast<int>(temp_values.size());
                    temp_values.push_back(constants_[dn.const_index]);
                }
                else if (dn.op == internal::LazyOp::SQRT || dn.op == internal::LazyOp::EXP ||
                    dn.op == internal::LazyOp::LOG || dn.op == internal::LazyOp::SIN ||
                    dn.op == internal::LazyOp::COS || dn.op == internal::LazyOp::ACOS ||
                    dn.op == internal::LazyOp::PI || dn.op == internal::LazyOp::E ||
                    dn.op == internal::LazyOp::POW) {
                    // Для операций с eps: const_index указывает на ЗНАЧЕНИЕ параметра точности
                    eps_idx = static_cast<int>(temp_values.size());
                    temp_values.push_back(constants_[dn.const_index]);
                    // value_idx остаётся -1 (не используется для этих операций)
                }
            }
            assert((dn.op == internal::LazyOp::CONST) ? (eps_idx == -1 && value_idx != -1) : true);
            assert((dn.op == internal::LazyOp::POW || dn.op == internal::LazyOp::SQRT ||
                dn.op == internal::LazyOp::EXP || dn.op == internal::LazyOp::LOG ||
                dn.op == internal::LazyOp::SIN || dn.op == internal::LazyOp::COS ||
                dn.op == internal::LazyOp::ACOS || dn.op == internal::LazyOp::PI ||
                dn.op == internal::LazyOp::E)
                ? (value_idx == -1 && eps_idx != -1) : true);
            // Вычисляем hash, approx, depth (аналогично глобальному пулу, но без кэша)
            uint64_t hash = static_cast<uint64_t>(dn.op);
            internal::Interval approx;
            int32_t depth = 0;

            if (dn.op == internal::LazyOp::CONST) {
                double d = internal::to_double(temp_values[value_idx]);
                hash = internal::compute_hash_const(temp_values[value_idx]);
                approx = internal::Interval(d);
                depth = 0;
            }
            else if (dn.op == internal::LazyOp::SUM) {
                approx = internal::Interval::zero();
                for (int c : temp_children) {
                    depth = std::max(depth, temp_nodes[c].depth + 1);
                    approx = approx + temp_nodes[c].approx;
                    hash = internal::combine_hash(internal::LazyOp::SUM, hash, temp_nodes[c].hash);
                }
            }
            else if (dn.op == internal::LazyOp::PRODUCT) {
                approx = internal::Interval::one();
                for (int c : temp_children) {
                    depth = std::max(depth, temp_nodes[c].depth + 1);
                    approx = approx * temp_nodes[c].approx;
                    hash = internal::combine_hash(internal::LazyOp::PRODUCT, hash, temp_nodes[c].hash);
                }
            }
            else if (dn.op == internal::LazyOp::NEG || dn.op == internal::LazyOp::RECIP ||
                dn.op == internal::LazyOp::SQRT || dn.op == internal::LazyOp::EXP ||
                dn.op == internal::LazyOp::LOG || dn.op == internal::LazyOp::SIN ||
                dn.op == internal::LazyOp::COS || dn.op == internal::LazyOp::ACOS) {
                int c = temp_children[0];
                depth = 1 + temp_nodes[c].depth;
                approx = internal::compute_interval(dn.op, temp_nodes[c].approx);
                hash = internal::combine_hash(dn.op, temp_nodes[c].hash, 0, eps_idx);
            }
            else if (dn.op == internal::LazyOp::PI || dn.op == internal::LazyOp::E) {
                depth = 0;
                approx = internal::compute_interval(dn.op, internal::Interval());
                hash = internal::combine_hash(dn.op, 0, eps_idx);
            }
            else if (dn.op == internal::LazyOp::POW) {
                int base = temp_children[0];
                int exp = temp_children[1];
                depth = 1 + std::max(temp_nodes[base].depth, temp_nodes[exp].depth);
                approx = internal::compute_interval(internal::LazyOp::POW, temp_nodes[base].approx, temp_nodes[exp].approx);
                hash = internal::combine_hash(internal::LazyOp::POW, temp_nodes[base].hash, temp_nodes[exp].hash, eps_idx);
            }
            else {
                throw std::logic_error("canonicalize: unknown LazyOp");
            }

            int temp_idx = static_cast<int>(temp_nodes.size());
            temp_nodes.emplace_back(dn.op, std::move(temp_children), value_idx, eps_idx, hash, approx, depth);
            dirty_to_temp[dirty_idx] = temp_idx;
        }
        int temp_root = dirty_to_temp[root_];

        // 2. Упрощение (локальное, без глобального пула)
        int simplified_root = internal::simplify_tree(temp_nodes, temp_values, temp_root);

        // 3. Интернирование упрощённого дерева в глобальный пул
        std::vector<int> temp_to_global(temp_nodes.size(), -1);
        for (size_t i = 0; i < temp_nodes.size(); ++i) {
            const auto& tn = temp_nodes[i];
            int global_idx = -1;
            switch (tn.op) {
            case internal::LazyOp::CONST:
                global_idx = internal::add_const(temp_values[tn.value_idx]);
                break;
            case internal::LazyOp::SUM: {
                std::vector<int32_t> global_children;
                for (int c : tn.children) global_children.push_back(temp_to_global[c]);
                global_idx = internal::make_sum_node(std::move(global_children));
                break;
            }
            case internal::LazyOp::PRODUCT: {
                std::vector<int32_t> global_children;
                for (int c : tn.children) global_children.push_back(temp_to_global[c]);
                global_idx = internal::make_product_node(std::move(global_children));
                break;
            }
            case internal::LazyOp::NEG:
            case internal::LazyOp::RECIP:
            case internal::LazyOp::SQRT:
            case internal::LazyOp::EXP:
            case internal::LazyOp::LOG:
            case internal::LazyOp::SIN:
            case internal::LazyOp::COS:
            case internal::LazyOp::ACOS: {
                int child_global = temp_to_global[tn.children[0]];
                int eps_global = (tn.eps_idx != -1) ? internal::pool.add_value(temp_values[tn.eps_idx]) : -1;
                global_idx = internal::get_unary_node(tn.op, child_global, eps_global);
                break;
            }
            case internal::LazyOp::PI:
            case internal::LazyOp::E: {
                int eps_global = (tn.eps_idx != -1) ? internal::pool.add_value(temp_values[tn.eps_idx]) : -1;
                global_idx = internal::get_unary_node(tn.op, -1, eps_global);
                break;
            }
            case internal::LazyOp::POW: {
                int base_global = temp_to_global[tn.children[0]];
                int exp_global = temp_to_global[tn.children[1]];
                int eps_global = (tn.eps_idx != -1) ? internal::pool.add_value(temp_values[tn.eps_idx]) : -1;
                global_idx = internal::get_pow_node(base_global, exp_global, eps_global);
                break;
            }
            default:
                throw std::logic_error("canonicalize: unknown LazyOp");
            }
            temp_to_global[i] = global_idx;
        }
        int new_clean_root = temp_to_global[simplified_root];
        internal::increment_ref(new_clean_root);

        // 4. Переход в Clean (без вызова shrink_to_fit, т.к. это const метод)
        state_ = State::Clean;
        clean_index_ = new_clean_root;
        nodes_.clear();
        constants_.clear();
        root_ = -1;
        cached_interval_.reset();
    }
    // ============================================================================
    // eval, eval_inplace, simplify, simplify_inplace, approx_interval
    // ============================================================================

    // Minor ToDo: перенести в evaluate_impl.h и тоже имплементировать оптимальный батчинг (неоптимальный - не имплементировать).
    inline Rational LazyRational::eval_dirty() const {
        assert(state_ == State::Dirty);
        const auto& nodes = nodes_;
        const auto& constants = constants_;
        const size_t n = nodes.size();
        std::vector<std::optional<internal::Value>> cache(n);
        std::stack<int> st;
        st.push(root_);

        while (!st.empty()) {
            int idx = st.top();
            if (cache[idx].has_value()) {
                st.pop();
                continue;
            }
            const auto& dn = nodes[idx];
            bool children_ready = true;
            for (int child : dn.children) {
                if (!cache[child].has_value()) {
                    st.push(child);
                    children_ready = false;
                }
            }
            if (!children_ready) continue;

            internal::Value result;
            switch (dn.op) {
            case internal::LazyOp::CONST:
                result = constants[dn.const_index];
                break;
            case internal::LazyOp::SUM: {
                if (dn.children.empty()) {
                    result = internal::Value(internal::SmallStorage(0));
                }
                else {
                    result = cache[dn.children[0]].value();
                    for (size_t i = 1; i < dn.children.size(); ++i) {
                        result = internal::eager_add(result, cache[dn.children[i]].value());
                    }
                }
                break;
            }
            case internal::LazyOp::PRODUCT: {
                if (dn.children.empty()) {
                    result = internal::Value(internal::SmallStorage(1));
                }
                else {
                    result = cache[dn.children[0]].value();
                    for (size_t i = 1; i < dn.children.size(); ++i) {
                        result = internal::eager_mul(result, cache[dn.children[i]].value());
                    }
                }
                break;
            }
            case internal::LazyOp::NEG:
                result = internal::eager_neg(cache[dn.children[0]].value());
                break;
            case internal::LazyOp::RECIP:
                result = internal::eager_div(internal::Value(internal::SmallStorage(1)), cache[dn.children[0]].value());
                break;
            case internal::LazyOp::SQRT:
            case internal::LazyOp::EXP:
            case internal::LazyOp::LOG:
            case internal::LazyOp::SIN:
            case internal::LazyOp::COS:
            case internal::LazyOp::ACOS: {
                internal::Value eps = constants[dn.const_index]; // eps хранится в const_index для этих операций
                internal::Value arg = cache[dn.children[0]].value();
                switch (dn.op) {
                case internal::LazyOp::SQRT: result = internal::eager_sqrt(arg, eps); break;
                case internal::LazyOp::EXP:  result = internal::eager_exp(arg, eps); break;
                case internal::LazyOp::LOG:  result = internal::eager_log(arg, eps); break;
                case internal::LazyOp::SIN:  result = internal::eager_sin(arg, eps); break;
                case internal::LazyOp::COS:  result = internal::eager_cos(arg, eps); break;
                case internal::LazyOp::ACOS: result = internal::eager_acos(arg, eps); break;
                default: break;
                }
                break;
            }
            case internal::LazyOp::PI:
            case internal::LazyOp::E: {
                internal::Value eps = constants[dn.const_index];
                if (dn.op == internal::LazyOp::PI) result = internal::eager_pi(eps);
                else result = internal::eager_e(eps);
                break;
            }
            case internal::LazyOp::POW: {
                internal::Value base = cache[dn.children[0]].value();
                internal::Value exp = cache[dn.children[1]].value();
                internal::Value eps = constants[dn.const_index];
                result = internal::eager_pow(base, exp, eps);
                break;
            }
            default:
                throw std::logic_error("eval_dirty: unknown op");
            }
            cache[idx] = result;
            st.pop();
        }
        return Rational(cache[root_].value());
    }

    // Minor-major ToDo: ввести eval_cache внутрь LazyRational чтоб кэшировать результат вычисления для выдачи при повторных требованиях.
    // - Кэш тухнет при изменении дерева. Имплементировать так чтобы все эти дурацкие кэши не жрали лишнего байтового веса объекта когда не используются. 
    inline Rational LazyRational::eval(bool skip_simplify) const {
        if (state_ == State::Dirty) {
            if (skip_simplify) {
                return eval_dirty();//выход: считаем от гряхного дерева потому что попросили. Иначе идём в станлартный путь.
            }
            // Исходное поведение: канонизируем (если Dirty) и вычисляем от индекса в пуле.
            canonicalize();
        }
        return Rational(internal::evaluate(clean_index_));
    }

    inline void LazyRational::eval_inplace(bool skip_simplify) {
        if (state_ == State::Dirty) {
            if (skip_simplify) {
                Rational result = eval_dirty();
                // Заменяем текущее дерево на константу
                int new_clean = internal::add_const(result.value());
                internal::increment_ref(new_clean);
                if (state_ == State::Clean) internal::decrement_ref(clean_index_);
                state_ = State::Clean;
                clean_index_ = new_clean;
                nodes_.clear();
                constants_.clear();
                root_ = -1;
                cached_interval_.reset();// а зачем ресетить интервал если численный результат дерева не изменился? Не знаю. Подумаем над этим.
                return;
            }
            //исходное поведение
            canonicalize();
        }
        // Исходное поведение
        Rational result = Rational(internal::evaluate(clean_index_));
        int new_clean = internal::add_const(result.value());
        internal::increment_ref(new_clean);
        internal::decrement_ref(clean_index_);
        clean_index_ = new_clean;
        cached_interval_.reset();
    }

    inline void LazyRational::simplify_inplace() {
        if (state_ == State::Dirty) canonicalize();
    }

    inline LazyRational LazyRational::simplify() const {
        LazyRational copy = clone();
        copy.simplify_inplace();
        return copy;
    }

    inline internal::Interval LazyRational::approx_interval() const {
        if (cached_interval_.has_value()) return *cached_interval_;
        internal::Interval result;
        if (state_ == State::Clean) {
            result = internal::pool.nodes[clean_index_].approx;
        }
        else {
            result = compute_interval_dirty(*this);   // теперь без detail::
        }
        cached_interval_ = result;
        return result;
    }

    // ------------------------------------------------------------------------
    // Операторы сравнения с многоуровневой логикой (канонизируют оригиналы)
    // ------------------------------------------------------------------------

    inline bool operator==(const LazyRational& a, const LazyRational& b) {
        // 1. Быстрая проверка: одинаковые чистые узлы
        if (a.is_clean() && b.is_clean() && a.clean_index_ == b.clean_index_)
            return true;

        // 2. Интервальная проверка: если интервалы не пересекаются -> точно не равны
        if (!a.approx_interval().overlaps(b.approx_interval()))
            return false;

        // 3. Канонизация (теперь деревья становятся чистыми)
        a.canonicalize();
        b.canonicalize();

        // 4. Повторная проверка индексов
        if (a.clean_index_ == b.clean_index_)
            return true;

        // 5. Вычисление значений и сравнение
        return a.eval() == b.eval();
    }

    inline bool operator!=(const LazyRational& a, const LazyRational& b) {
        return !(a == b);
    }

    inline bool operator<(const LazyRational& a, const LazyRational& b) {
        // Интервальная проверка
        internal::Interval ia = a.approx_interval();
        internal::Interval ib = b.approx_interval();

        // Если интервал a полностью меньше интервала b -> a < b точно
        if (ia.upper() < ib.lower())
            return true;
        // Если интервал a не меньше (т.е. a.lower() >= b.upper()) -> a >= b точно
        if (ia.lower() >= ib.upper())
            return false;

        // Интервалы пересекаются – нужна канонизация и точное сравнение
        a.canonicalize();
        b.canonicalize();
        return a.eval() < b.eval();
    }

    inline bool operator<=(const LazyRational& a, const LazyRational& b) {
        return !(b < a);
    }

    inline bool operator>(const LazyRational& a, const LazyRational& b) {
        return b < a;
    }

    inline bool operator>=(const LazyRational& a, const LazyRational& b) {
        return !(a < b);
    }

    // ============================================================================
    // Вывод в поток
    // ============================================================================
    inline std::ostream& operator<<(std::ostream& os, const LazyRational& lr) {
        os << lr.eval();
        return os;
    }

} // namespace delta