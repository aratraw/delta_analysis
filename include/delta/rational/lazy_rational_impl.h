// lazy_rational_impl.h
// Версия 3.0 – реализация LazyRational с унифицированными узлами и гетерогенным хранением
// ----------------------------------------------------------------------------
// Изменения:
//   - Адаптировано под новую структуру DirtyNode (value_idx, eps_idx, leaf_values, complex_children)
//   - Гетерогенное добавление в SUM/PRODUCT через append_sum_children / append_product_children
//   - Zero-overhead операторы +=, + с Rational (прямая вставка в leaf_values)
//   - Методы массовой вставки append_values, append_nodes
//   - Канонизация через TempNode с поддержкой гетерогенного хранения
//   - Вычисление через evaluate_impl (evaluate_dirty, evaluate_dirty_inplace)
//   - Оптимизация перемещений, избегание копий
//   - Исправления: сигнатура new_dirty_node с явными value_idx и eps_idx, исправлены все вызовы
//   - Удалены упоминания SmallStorage, заменены на Value
// ----------------------------------------------------------------------------

#pragma once

#include "node_pool.h"
#include "evaluate_impl.h"
#include "lazy_nodes.h"
#include "simplify_impl.h"
#include "interval.h"
#include <stack>
#include <cassert>
#include <algorithm>
#include <optional>
#include <vector>
#include <string>

namespace delta {

    // ------------------------------------------------------------------------
    // Вспомогательные функции для грязного дерева
    // ------------------------------------------------------------------------

    // Вычисление интервала для грязного дерева (итеративно)
    inline internal::Interval compute_interval_dirty(const LazyRational& lr) {
        assert(lr.is_dirty());
        const auto& nodes = lr.nodes_;
        const auto& constants = lr.constants_;
        std::vector<internal::Interval> intervals(nodes.size());

        std::stack<int> st;
        st.push(lr.root_);
        std::vector<int> postorder;
        while (!st.empty()) {
            int idx = st.top(); st.pop();
            postorder.push_back(idx);
            const auto& dn = nodes[idx];
            if (dn.op == internal::LazyOp::SUM || dn.op == internal::LazyOp::PRODUCT) {
                for (int child : dn.complex_children) st.push(child);
            }
            else {
                for (int child : dn.children) st.push(child);
            }
        }

        for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
            int idx = *it;
            const auto& dn = nodes[idx];
            switch (dn.op) {
            case internal::LazyOp::CONST: {
                intervals[idx] = internal::Interval(internal::to_double(constants[dn.value_idx]));
                break;
            }
            case internal::LazyOp::SUM: {
                internal::Interval sum = internal::Interval::zero();
                for (const auto& v : dn.leaf_values) {
                    sum = sum + internal::Interval(internal::to_double(v));
                }
                for (int child : dn.complex_children) {
                    sum = sum + intervals[child];
                }
                intervals[idx] = sum;
                break;
            }
            case internal::LazyOp::PRODUCT: {
                internal::Interval prod = internal::Interval::one();
                for (const auto& v : dn.leaf_values) {
                    prod = prod * internal::Interval(internal::to_double(v));
                }
                for (int child : dn.complex_children) {
                    prod = prod * intervals[child];
                }
                intervals[idx] = prod;
                break;
            }
            case internal::LazyOp::NEG: {
                intervals[idx] = -intervals[dn.children[0]];
                break;
            }
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

    // ------------------------------------------------------------------------
    // Конструкторы
    // ------------------------------------------------------------------------
    inline LazyRational::LazyRational() : state_(State::Dirty) {
        int const_idx = add_constant(internal::Value(0));
        nodes_.emplace_back(internal::LazyOp::CONST, const_idx);
        root_ = 0;
    }

    inline LazyRational::LazyRational(const Rational& r) : state_(State::Dirty) {
        int const_idx = add_constant(r.value());
        nodes_.emplace_back(internal::LazyOp::CONST, const_idx);
        root_ = 0;
    }

    inline LazyRational::LazyRational(Rational&& r) : state_(State::Dirty) {
        int const_idx = add_constant(std::move(r.value()));
        nodes_.emplace_back(internal::LazyOp::CONST, const_idx);
        root_ = 0;
    }

    // Move-конструктор/оператор
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
            new (this) LazyRational(std::move(other));
        }
        return *this;
    }

    inline LazyRational::~LazyRational() {
        if (state_ == State::Clean) {
            internal::decrement_ref(clean_index_);
        }
    }

    // ------------------------------------------------------------------------
    // Приватные методы: add_constant, new_dirty_node
    // ------------------------------------------------------------------------
    inline int LazyRational::add_constant(const internal::Value& v) {
        assert(state_ == State::Dirty);
        constants_.push_back(v);
        return static_cast<int>(constants_.size() - 1);
    }

    inline int LazyRational::new_dirty_node(internal::LazyOp op,
        absl::InlinedVector<int32_t, 4> children,
        int value_idx,   // для CONST
        int eps_idx) {   // для операций с eps
        assert(state_ == State::Dirty);
        if (op == internal::LazyOp::CONST) {
            nodes_.emplace_back(op, value_idx);
        }
        else if (op == internal::LazyOp::SUM || op == internal::LazyOp::PRODUCT) {
            nodes_.emplace_back(op, std::vector<internal::Value>{}, absl::InlinedVector<int32_t, 4>{});
        }
        else {
            // Унарные/бинарные/PI/E: eps_idx передаётся явно
            nodes_.emplace_back(op, std::move(children), eps_idx);
        }
        return static_cast<int>(nodes_.size() - 1);
    }

    // ------------------------------------------------------------------------
    // import_tree – копирование поддерева в грязное состояние
    // ------------------------------------------------------------------------
    inline int LazyRational::import_tree(const LazyRational& other) {
        assert(state_ == State::Dirty);

        // Self-import: глубокое клонирование
        if (this == &other) {
            LazyRational temp = other.clone();
            return import_tree(temp);
        }

        // ------------------------------------------------------------------------
        // 1. Источник — грязное дерево (Dirty -> Dirty)
        // ------------------------------------------------------------------------
        if (other.state_ == State::Dirty) {
            std::vector<int> old_to_new(other.nodes_.size(), -1);
            std::vector<int> old_const_to_new(other.constants_.size(), -1);  // ← маппинг констант

            // Пост-порядный обход источника
            std::stack<int> st;
            st.push(other.root_);
            std::vector<int> postorder;
            while (!st.empty()) {
                int idx = st.top(); st.pop();
                postorder.push_back(idx);
                const auto& dn = other.nodes_[idx];
                if (dn.op == internal::LazyOp::SUM || dn.op == internal::LazyOp::PRODUCT) {
                    for (int child : dn.complex_children) st.push(child);
                }
                else {
                    for (int child : dn.children) st.push(child);
                }
            }

            for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
                int old_idx = *it;
                const auto& old_node = other.nodes_[old_idx];
                int new_idx = -1;

                if (old_node.op == internal::LazyOp::CONST) {
                    // Добавляем константу, если ещё не скопирована
                    if (old_const_to_new[old_node.value_idx] == -1) {
                        old_const_to_new[old_node.value_idx] =
                            add_constant(other.constants_[old_node.value_idx]);
                    }
                    int new_const = old_const_to_new[old_node.value_idx];
                    new_idx = new_dirty_node(old_node.op, {}, new_const, -1);
                }
                else if (old_node.op == internal::LazyOp::SUM || old_node.op == internal::LazyOp::PRODUCT) {
                    absl::InlinedVector<int32_t, 4> new_complex;
                    for (int child : old_node.complex_children) {
                        new_complex.push_back(old_to_new[child]);
                    }
                    std::vector<internal::Value> new_leaf = old_node.leaf_values;  // значения копируются
                    int new_node_idx = static_cast<int>(nodes_.size());
                    nodes_.emplace_back(old_node.op, std::move(new_leaf), std::move(new_complex));
                    new_idx = new_node_idx;
                }
                else {
                    // Унарные / бинарные / PI / E
                    absl::InlinedVector<int32_t, 4> new_children;
                    for (int child : old_node.children) {
                        new_children.push_back(old_to_new[child]);
                    }

                    // Обработка eps_idx
                    int new_eps = -1;
                    if (old_node.eps_idx != -1) {
                        if (old_const_to_new[old_node.eps_idx] == -1) {
                            old_const_to_new[old_node.eps_idx] =
                                add_constant(other.constants_[old_node.eps_idx]);
                        }
                        new_eps = old_const_to_new[old_node.eps_idx];
                    }

                    new_idx = new_dirty_node(old_node.op, std::move(new_children), -1, new_eps);
                }

                old_to_new[old_idx] = new_idx;
            }

            return old_to_new[other.root_];
        }

        // ------------------------------------------------------------------------
        // 2. Источник — чистое дерево (Clean -> Dirty)
        // ------------------------------------------------------------------------
        else {
            // Строим временное грязное дерево из чистого с правильным маппингом констант
            LazyRational temp;
            temp.state_ = State::Dirty;

            std::stack<int> st;
            st.push(other.clean_index_);
            std::vector<int> postorder;
            while (!st.empty()) {
                int idx = st.top(); st.pop();
                postorder.push_back(idx);
                const auto& node = internal::pool.nodes[idx];
                if (node.op == internal::LazyOp::SUM || node.op == internal::LazyOp::PRODUCT) {
                    for (int child : node.complex_children) st.push(child);
                }
                else {
                    for (int child : node.children) st.push(child);
                }
            }

            std::vector<int> clean_to_dirty(internal::pool.nodes.size(), -1);
            std::vector<int> value_idx_map(internal::pool.values.size(), -1);  // маппинг констант чистого пула

            for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
                int clean_idx = *it;
                const auto& clean_node = internal::pool.nodes[clean_idx];
                int dirty_idx = -1;

                if (clean_node.op == internal::LazyOp::CONST) {
                    int const_idx = clean_node.value_idx;
                    if (value_idx_map[const_idx] == -1) {
                        value_idx_map[const_idx] = temp.add_constant(internal::pool.values[const_idx]);
                    }
                    int new_const = value_idx_map[const_idx];
                    dirty_idx = temp.new_dirty_node(clean_node.op, {}, new_const, -1);
                }
                else if (clean_node.op == internal::LazyOp::SUM || clean_node.op == internal::LazyOp::PRODUCT) {
                    absl::InlinedVector<int32_t, 4> new_complex;
                    for (int child : clean_node.complex_children) {
                        new_complex.push_back(clean_to_dirty[child]);
                    }
                    std::vector<internal::Value> new_leaf = clean_node.leaf_values;
                    int new_node_idx = static_cast<int>(temp.nodes_.size());
                    temp.nodes_.emplace_back(clean_node.op, std::move(new_leaf), std::move(new_complex));
                    dirty_idx = new_node_idx;
                }
                else {
                    // Унарные / бинарные / PI / E
                    absl::InlinedVector<int32_t, 4> new_children;
                    for (int child : clean_node.children) {
                        new_children.push_back(clean_to_dirty[child]);
                    }

                    int new_eps = -1;
                    if (clean_node.eps_idx != -1) {
                        if (value_idx_map[clean_node.eps_idx] == -1) {
                            value_idx_map[clean_node.eps_idx] = temp.add_constant(internal::pool.values[clean_node.eps_idx]);
                        }
                        new_eps = value_idx_map[clean_node.eps_idx];
                    }

                    dirty_idx = temp.new_dirty_node(clean_node.op, std::move(new_children), -1, new_eps);
                }

                clean_to_dirty[clean_idx] = dirty_idx;
            }

            temp.root_ = clean_to_dirty[other.clean_index_];

            // Теперь у нас есть корректное грязное представление чистого дерева,
            // импортируем его в текущий объект рекурсивным вызовом.
            return import_tree(temp);
        }
    }
    // ------------------------------------------------------------------------
    // clone
    // ------------------------------------------------------------------------
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

    // ------------------------------------------------------------------------
    // ensure_dirty
    // ------------------------------------------------------------------------
    inline void LazyRational::ensure_dirty() {
        if (state_ == State::Clean) {
            invalidate_interval();
            LazyRational temp;
            temp.state_ = State::Dirty;
            std::stack<int> st;
            st.push(clean_index_);
            std::vector<int> postorder;
            while (!st.empty()) {
                int idx = st.top(); st.pop();
                postorder.push_back(idx);
                const auto& node = internal::pool.nodes[idx];
                if (node.op == internal::LazyOp::SUM || node.op == internal::LazyOp::PRODUCT) {
                    for (int child : node.complex_children) st.push(child);
                }
                else {
                    for (int child : node.children) st.push(child);
                }
            }

            std::vector<int> clean_to_dirty(internal::pool.nodes.size(), -1);
            // Маппинг для индексов констант в чистом пуле -> индексы в temp.constants_
            std::vector<int> value_idx_map(internal::pool.values.size(), -1);

            for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
                int clean_idx = *it;
                const auto& clean_node = internal::pool.nodes[clean_idx];
                int dirty_idx = -1;
                if (clean_node.op == internal::LazyOp::CONST) {
                    // Добавляем константу, если ещё не добавлена
                    int const_idx = clean_node.value_idx;
                    if (value_idx_map[const_idx] == -1) {
                        value_idx_map[const_idx] = temp.add_constant(internal::pool.values[const_idx]);
                    }
                    int new_const = value_idx_map[const_idx];
                    dirty_idx = temp.new_dirty_node(clean_node.op, {}, new_const, -1);
                }
                else if (clean_node.op == internal::LazyOp::SUM || clean_node.op == internal::LazyOp::PRODUCT) {
                    absl::InlinedVector<int32_t, 4> new_complex;
                    for (int child : clean_node.complex_children) {
                        new_complex.push_back(clean_to_dirty[child]);
                    }
                    std::vector<internal::Value> new_leaf = clean_node.leaf_values;
                    int new_node_idx = static_cast<int>(temp.nodes_.size());
                    temp.nodes_.emplace_back(clean_node.op, std::move(new_leaf), std::move(new_complex));
                    dirty_idx = new_node_idx;
                }
                else {
                    absl::InlinedVector<int32_t, 4> new_children;
                    for (int child : clean_node.children) {
                        new_children.push_back(clean_to_dirty[child]);
                    }
                    // Обрабатываем eps_idx
                    int new_eps = -1;
                    if (clean_node.eps_idx != -1) {
                        if (value_idx_map[clean_node.eps_idx] == -1) {
                            value_idx_map[clean_node.eps_idx] = temp.add_constant(internal::pool.values[clean_node.eps_idx]);
                        }
                        new_eps = value_idx_map[clean_node.eps_idx];
                    }
                    dirty_idx = temp.new_dirty_node(clean_node.op, std::move(new_children), -1, new_eps);
                }
                clean_to_dirty[clean_idx] = dirty_idx;
            }
            temp.root_ = clean_to_dirty[clean_index_];
            *this = std::move(temp);
            internal::decrement_ref(clean_index_);
        }
    }
    // ------------------------------------------------------------------------
    // append_sum_children / append_product_children (гетерогенные)
    // ------------------------------------------------------------------------
    inline void LazyRational::append_sum_children(int sum_node, const LazyRational& other) {
        assert(state_ == State::Dirty);
        assert(nodes_[sum_node].op == internal::LazyOp::SUM);
        int other_root = import_tree(other);
        auto& target = nodes_[sum_node];
        const auto& other_node = nodes_[other_root];

        if (other_node.op == internal::LazyOp::SUM) {
            target.leaf_values.insert(target.leaf_values.end(),
                std::make_move_iterator(other_node.leaf_values.begin()),
                std::make_move_iterator(other_node.leaf_values.end()));
            for (int child : other_node.complex_children) {
                target.complex_children.push_back(child);
            }
        }
        else if (other_node.op == internal::LazyOp::CONST) {
            target.leaf_values.push_back(constants_[other_node.value_idx]);
        }
        else {
            target.complex_children.push_back(other_root);
        }
    }

    inline void LazyRational::append_product_children(int prod_node, const LazyRational& other) {
        assert(state_ == State::Dirty);
        assert(nodes_[prod_node].op == internal::LazyOp::PRODUCT);
        int other_root = import_tree(other);
        auto& target = nodes_[prod_node];
        const auto& other_node = nodes_[other_root];

        if (other_node.op == internal::LazyOp::PRODUCT) {
            target.leaf_values.insert(target.leaf_values.end(),
                std::make_move_iterator(other_node.leaf_values.begin()),
                std::make_move_iterator(other_node.leaf_values.end()));
            for (int child : other_node.complex_children) {
                target.complex_children.push_back(child);
            }
        }
        else if (other_node.op == internal::LazyOp::CONST) {
            target.leaf_values.push_back(constants_[other_node.value_idx]);
        }
        else {
            target.complex_children.push_back(other_root);
        }
    }

    // ------------------------------------------------------------------------
    // Мутирующие операторы (с гетерогенным добавлением)
    // ------------------------------------------------------------------------
    inline LazyRational& operator+(LazyRational& a, const LazyRational& b) {
        a.ensure_dirty();
        a.invalidate_interval();
        int root = a.root_;
        if (a.nodes_[root].op != internal::LazyOp::SUM) {
            int b_root = a.import_tree(b);
            std::vector<internal::Value> leaf_vals;
            absl::InlinedVector<int32_t, 4> complex_children;
            if (a.nodes_[root].op == internal::LazyOp::CONST) {
                leaf_vals.push_back(a.constants_[a.nodes_[root].value_idx]);
            }
            else {
                complex_children.push_back(root);
            }
            if (b_root != -1) {
                const auto& b_node = a.nodes_[b_root];
                if (b_node.op == internal::LazyOp::CONST) {
                    leaf_vals.push_back(a.constants_[b_node.value_idx]);
                }
                else {
                    complex_children.push_back(b_root);
                }
            }
            int new_root = a.new_dirty_node(internal::LazyOp::SUM, {}, -1, -1);
            a.nodes_[new_root].leaf_values = std::move(leaf_vals);
            a.nodes_[new_root].complex_children = std::move(complex_children);
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
        a.ensure_dirty();
        a.invalidate_interval();
        int root = a.root_;
        if (a.nodes_[root].op == internal::LazyOp::SUM) {
            a.nodes_[root].leaf_values.push_back(b.value());
        }
        else {
            std::vector<internal::Value> leaf_vals;
            absl::InlinedVector<int32_t, 4> complex_children;
            if (a.nodes_[root].op == internal::LazyOp::CONST) {
                leaf_vals.push_back(a.constants_[a.nodes_[root].value_idx]);
            }
            else {
                complex_children.push_back(root);
            }
            leaf_vals.push_back(b.value());
            int new_root = a.new_dirty_node(internal::LazyOp::SUM, {}, -1, -1);
            a.nodes_[new_root].leaf_values = std::move(leaf_vals);
            a.nodes_[new_root].complex_children = std::move(complex_children);
            a.root_ = new_root;
        }
        return a;
    }

    inline LazyRational&& operator+(LazyRational&& a, const Rational& b) {
        return std::move(operator+(a, b));
    }

    // Унарный минус
    inline LazyRational operator-(const LazyRational& a) {
        LazyRational result = a.clone();
        result.ensure_dirty();
        result.invalidate_interval();
        int root = result.root_;
        int neg_root = result.new_dirty_node(internal::LazyOp::NEG, { root }, -1, -1);
        result.root_ = neg_root;
        return result;
    }

    // Бинарное вычитание через сложение с унарным минусом
    inline LazyRational& operator-(LazyRational& a, const LazyRational& b) {
        LazyRational neg_b = -b;
        return a + neg_b;
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

    // Умножение
    inline LazyRational& operator*(LazyRational& a, const LazyRational& b) {
        a.ensure_dirty();
        a.invalidate_interval();
        int root = a.root_;
        if (a.nodes_[root].op != internal::LazyOp::PRODUCT) {
            int b_root = a.import_tree(b);
            std::vector<internal::Value> leaf_vals;
            absl::InlinedVector<int32_t, 4> complex_children;
            if (a.nodes_[root].op == internal::LazyOp::CONST) {
                leaf_vals.push_back(a.constants_[a.nodes_[root].value_idx]);
            }
            else {
                complex_children.push_back(root);
            }
            if (b_root != -1) {
                const auto& b_node = a.nodes_[b_root];
                if (b_node.op == internal::LazyOp::CONST) {
                    leaf_vals.push_back(a.constants_[b_node.value_idx]);
                }
                else {
                    complex_children.push_back(b_root);
                }
            }
            int new_root = a.new_dirty_node(internal::LazyOp::PRODUCT, {}, -1, -1);
            a.nodes_[new_root].leaf_values = std::move(leaf_vals);
            a.nodes_[new_root].complex_children = std::move(complex_children);
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
        a.ensure_dirty();
        a.invalidate_interval();
        int root = a.root_;
        if (a.nodes_[root].op == internal::LazyOp::PRODUCT) {
            a.nodes_[root].leaf_values.push_back(b.value());
        }
        else {
            std::vector<internal::Value> leaf_vals;
            absl::InlinedVector<int32_t, 4> complex_children;
            if (a.nodes_[root].op == internal::LazyOp::CONST) {
                leaf_vals.push_back(a.constants_[a.nodes_[root].value_idx]);
            }
            else {
                complex_children.push_back(root);
            }
            leaf_vals.push_back(b.value());
            int new_root = a.new_dirty_node(internal::LazyOp::PRODUCT, {}, -1, -1);
            a.nodes_[new_root].leaf_values = std::move(leaf_vals);
            a.nodes_[new_root].complex_children = std::move(complex_children);
            a.root_ = new_root;
        }
        return a;
    }

    inline LazyRational&& operator*(LazyRational&& a, const Rational& b) {
        return std::move(operator*(a, b));
    }

    // Деление = * RECIP
    inline LazyRational& operator/(LazyRational& a, const LazyRational& b) {
        LazyRational recip_b = b.clone();
        recip_b.ensure_dirty();
        recip_b.invalidate_interval();
        int b_root = recip_b.root_;
        int recip_root = recip_b.new_dirty_node(internal::LazyOp::RECIP, { b_root }, -1, -1);
        recip_b.root_ = recip_root;
        return a * recip_b;
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

    // Составные операторы
    inline LazyRational& operator+=(LazyRational& a, const LazyRational& b) { return a + b; }
    inline LazyRational& operator+=(LazyRational& a, const Rational& b) { return a + b; }
    inline LazyRational& operator-=(LazyRational& a, const LazyRational& b) { return a - b; }
    inline LazyRational& operator-=(LazyRational& a, const Rational& b) { return a - b; }
    inline LazyRational& operator*=(LazyRational& a, const LazyRational& b) { return a * b; }
    inline LazyRational& operator*=(LazyRational& a, const Rational& b) { return a * b; }
    inline LazyRational& operator/=(LazyRational& a, const LazyRational& b) { return a / b; }
    inline LazyRational& operator/=(LazyRational& a, const Rational& b) { return a / b; }

    // ------------------------------------------------------------------------
    // Методы массовой вставки (bulk append)
    // ------------------------------------------------------------------------
    inline void LazyRational::append_values(std::vector<internal::Value>&& values) {
        ensure_dirty();
        invalidate_interval();
        if (nodes_[root_].op == internal::LazyOp::SUM) {
            auto& leaf = nodes_[root_].leaf_values;
            leaf.insert(leaf.end(),
                std::make_move_iterator(values.begin()),
                std::make_move_iterator(values.end()));
        }
        else {
            std::vector<internal::Value> leaf_vals = std::move(values);
            absl::InlinedVector<int32_t, 4> complex_children;
            if (nodes_[root_].op == internal::LazyOp::CONST) {
                leaf_vals.insert(leaf_vals.begin(), constants_[nodes_[root_].value_idx]);
            }
            else {
                complex_children.push_back(root_);
            }
            int new_root = new_dirty_node(internal::LazyOp::SUM, {}, -1, -1);
            nodes_[new_root].leaf_values = std::move(leaf_vals);
            nodes_[new_root].complex_children = std::move(complex_children);
            root_ = new_root;
        }
    }

    inline void LazyRational::append_nodes(std::vector<int>&& node_indices) {
        ensure_dirty();
        invalidate_interval();
        absl::InlinedVector<int32_t, 4> complex_children(node_indices.begin(), node_indices.end());
        if (nodes_[root_].op == internal::LazyOp::SUM) {
            auto& complex = nodes_[root_].complex_children;
            for (int idx : complex_children) {
                complex.push_back(idx);
            }
        }
        else {
            if (nodes_[root_].op != internal::LazyOp::SUM) {
                complex_children.insert(complex_children.begin(), root_);
            }
            int new_root = new_dirty_node(internal::LazyOp::SUM, {}, -1, -1);
            nodes_[new_root].complex_children = std::move(complex_children);
            root_ = new_root;
        }
    }

    // ------------------------------------------------------------------------
    // Канонизация (Dirty -> Clean) с использованием TempNode и simplify_tree
    // ------------------------------------------------------------------------
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
            const auto& dn = nodes_[idx];
            if (dn.op == internal::LazyOp::SUM || dn.op == internal::LazyOp::PRODUCT) {
                for (int child : dn.complex_children) st.push(child);
            }
            else {
                for (int child : dn.children) st.push(child);
            }
        }

        for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
            int dirty_idx = *it;
            const auto& dn = nodes_[dirty_idx];

            std::vector<int> temp_children;
            for (int child : dn.children) {
                temp_children.push_back(dirty_to_temp[child]);
            }

            std::vector<int> temp_complex;
            for (int child : dn.complex_children) {
                temp_complex.push_back(dirty_to_temp[child]);
            }

            int value_idx = -1, eps_idx = -1;
            if (dn.op == internal::LazyOp::CONST) {
                value_idx = static_cast<int>(temp_values.size());
                temp_values.push_back(constants_[dn.value_idx]);
            }
            else if (dn.eps_idx != -1) {
                eps_idx = static_cast<int>(temp_values.size());
                temp_values.push_back(constants_[dn.eps_idx]);
            }

            // Копируем leaf_values (без изменений)
            std::vector<internal::Value> leaf_vals = dn.leaf_values;

            // Вычисляем метаданные
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
                for (const auto& v : leaf_vals) {
                    approx = approx + internal::Interval(internal::to_double(v));
                    hash = absl::HashOf(hash, v);
                }
                for (int c : temp_complex) {
                    depth = std::max(depth, temp_nodes[c].depth + 1);
                    approx = approx + temp_nodes[c].approx;
                    hash = internal::combine_hash(internal::LazyOp::SUM, hash, temp_nodes[c].hash);
                }
            }
            else if (dn.op == internal::LazyOp::PRODUCT) {
                approx = internal::Interval::one();
                for (const auto& v : leaf_vals) {
                    approx = approx * internal::Interval(internal::to_double(v));
                    hash = absl::HashOf(hash, v);
                }
                for (int c : temp_complex) {
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
            if (dn.op == internal::LazyOp::SUM || dn.op == internal::LazyOp::PRODUCT) {
                temp_nodes.emplace_back(dn.op, std::move(leaf_vals), std::move(temp_complex),
                    value_idx, eps_idx, hash, approx, depth);
            }
            else {
                temp_nodes.emplace_back(dn.op, std::move(temp_children), value_idx, eps_idx, hash, approx, depth);
            }
            dirty_to_temp[dirty_idx] = temp_idx;
        }

        int temp_root = dirty_to_temp[root_];

        // 2. Упрощение (локальное, без глобального пула)
        int simplified_root = internal::simplify_tree(temp_nodes, temp_values, temp_root);

        // 3. Интернирование упрощённого дерева в глобальный пул (ПОСТ-ПОРЯДНЫЙ ОБХОД)
        std::vector<int> temp_to_global(temp_nodes.size(), -1);

        std::stack<int> st_glob;
        st_glob.push(simplified_root);
        std::vector<int> postorder_glob;
        while (!st_glob.empty()) {
            int idx = st_glob.top(); st_glob.pop();
            postorder_glob.push_back(idx);
            const auto& tn = temp_nodes[idx];
            if (tn.op == internal::LazyOp::SUM || tn.op == internal::LazyOp::PRODUCT) {
                for (int c : tn.complex_children) st_glob.push(c);
            }
            else {
                for (int c : tn.children) st_glob.push(c);
            }
        }

        for (auto it = postorder_glob.rbegin(); it != postorder_glob.rend(); ++it) {
            int idx = *it;
            const auto& tn = temp_nodes[idx];
            int global_idx = -1;

            if (tn.op == internal::LazyOp::CONST) {
                global_idx = internal::add_const(temp_values[tn.value_idx]);
            }
            else if (tn.op == internal::LazyOp::SUM) {
                absl::InlinedVector<int32_t, 4> complex_children;
                for (int c : tn.complex_children) complex_children.push_back(temp_to_global[c]);
                global_idx = internal::make_sum_node(tn.leaf_values, std::move(complex_children));
            }
            else if (tn.op == internal::LazyOp::PRODUCT) {
                absl::InlinedVector<int32_t, 4> complex_children;
                for (int c : tn.complex_children) complex_children.push_back(temp_to_global[c]);
                global_idx = internal::make_product_node(tn.leaf_values, std::move(complex_children));
            }
            else {
                absl::InlinedVector<int32_t, 4> children;
                for (int c : tn.children) children.push_back(temp_to_global[c]);
                int eps_global = (tn.eps_idx != -1) ? internal::pool.add_value(temp_values[tn.eps_idx]) : -1;
                global_idx = internal::get_unary_node(tn.op, std::move(children), eps_global);
            }

            temp_to_global[idx] = global_idx;
        }

        int new_clean_root = temp_to_global[simplified_root];
        internal::increment_ref(new_clean_root);

        // 4. Переход в Clean
        const_cast<LazyRational*>(this)->state_ = State::Clean;
        const_cast<LazyRational*>(this)->clean_index_ = new_clean_root;
        const_cast<LazyRational*>(this)->nodes_.clear();
        const_cast<LazyRational*>(this)->constants_.clear();
        const_cast<LazyRational*>(this)->root_ = -1;
        const_cast<LazyRational*>(this)->cached_interval_.reset();
    }
    // ------------------------------------------------------------------------
    // eval, eval_inplace, simplify, approx_interval
    // ------------------------------------------------------------------------
    inline Rational LazyRational::eval(bool skip_simplify) const {
        if (state_ == State::Clean) {
            const auto& node = internal::pool.nodes[clean_index_];
            if (node.op == internal::LazyOp::CONST) {
                return Rational(internal::pool.values[node.value_idx]);
            }
        }
        else {
            if (nodes_.size() == 1 && nodes_[0].op == internal::LazyOp::CONST) {
                return Rational(constants_[nodes_[0].value_idx]);
            }
            if (skip_simplify) {
                return Rational(internal::evaluate_dirty(nodes_, constants_, root_));
            }
            canonicalize();
        }
        return Rational(internal::evaluate(clean_index_));
    }

    inline void LazyRational::eval_inplace(bool skip_simplify) {
        Rational result;
        if (state_ == State::Dirty) {
            if (skip_simplify) {
                result = Rational(internal::evaluate_dirty_inplace(nodes_, constants_, root_));
            }
            else {
                canonicalize();
                result = Rational(internal::evaluate(clean_index_));
            }
        }
        else {
            result = Rational(internal::evaluate(clean_index_));
        }

        int new_clean = internal::add_const(result.value());
        internal::increment_ref(new_clean);
        if (state_ == State::Clean) {
            internal::decrement_ref(clean_index_);
        }
        state_ = State::Clean;
        clean_index_ = new_clean;
        nodes_.clear();
        constants_.clear();
        root_ = -1;
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
            result = compute_interval_dirty(*this);
        }
        cached_interval_ = result;
        return result;
    }

    // ------------------------------------------------------------------------
   // Сравнения (многоуровневые)
   // ------------------------------------------------------------------------
    inline bool operator==(const LazyRational& a, const LazyRational& b) {
        // 1. Быстрая проверка: одинаковые чистые узлы (структурное равенство)
        if (a.is_clean() && b.is_clean() && a.clean_index_ == b.clean_index_)
            return true;

        // 2. Получаем кэшированные интервалы или вычисляем с кэшированием если кэша нет
        // Интервальная проверка: если интервалы не пересекаются – точно не равны.
        if (!a.approx_interval().overlaps(b.approx_interval()))
            return false;

        // 3. Если быстрый выход не сработал
        // Канонизация (объекты становятся чистыми, грязные данные очищаются)
        a.canonicalize();
        b.canonicalize();

        // 4. Повторная проверка индексов (после канонизации могло совпасть)
        if (a.clean_index_ == b.clean_index_)
            return true;

        // 5. Полное вычисление значений и сравнение (дорогой путь)
        return a.eval() == b.eval();
    }

    inline bool operator!=(const LazyRational& a, const LazyRational& b) {
        return !(a == b);
    }

    inline bool operator<(const LazyRational& a, const LazyRational& b) {
        // 1. Получаем кэшированные интервалы или вычисляем с кэшированием если кэша нет
        internal::Interval ia = a.approx_interval();
        internal::Interval ib = b.approx_interval();

        // 2. Интервальная проверка: если a строго меньше b, ответ известен
        if (ia.upper() < ib.lower())
            return true;

        // 3. Если a точно не меньше b (т.е. a >= b), ответ тоже известен
        if (ia.lower() >= ib.upper())
            return false;

        // 4. Интервалы пересекаются – канонизируем и сравниваем точные значения
        a.canonicalize();
        b.canonicalize();
        return a.eval() < b.eval();
    }

    inline bool operator<=(const LazyRational& a, const LazyRational& b) { return !(b < a); }
    inline bool operator>(const LazyRational& a, const LazyRational& b) { return b < a; }
    inline bool operator>=(const LazyRational& a, const LazyRational& b) { return !(a < b); }

    // ------------------------------------------------------------------------
    // Вывод в поток
    // ------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const LazyRational& lr) {
        os << lr.eval();
        return os;
    }

} // namespace delta