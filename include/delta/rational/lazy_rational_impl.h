// lazy_rational_impl.h
// Версия 3.2 – удалены approx и depth из Node/TempNode, интервалы вычисляются лениво.
// ----------------------------------------------------------------------------
// Модификации для реестра чистых объектов (global_state.h)

    // -------------------------------------------------------------------------
    // TODO: Перспектива компактификации пула с использованием реестра.
    // -------------------------------------------------------------------------
    // Теперь, когда у нас есть реестр чистых объектов (g_clean_rationals),
    // появляется возможность не просто заменять живые поддеревья на константы,
    // но и перестраивать новый пул так, чтобы все корневые узлы (CONST) шли
    // подряд с индекса 0. Это позволит:
    //   - Уменьшить фрагментацию и повысить локальность данных.
    //   - Сократить размер пула до реально необходимого (после GC).
    //   - Гарантировать, что clean_index_ каждого чистого LazyRational
    //     можно обновить, пройдя по реестру и переписав индексы.
    // 
    // Алгоритм (потенциальный):
    //   1. В collect_garbage() после определения живых корней строим
    //      отображение old_index -> new_compact_index (только для живых узлов).
    //   2. Создаём новый пул, куда последовательно помещаем узлы-константы
    //      для каждого живого корня, начиная с индекса 0.
    //   3. Проходим по реестру g_clean_rationals, для каждого объекта
    //      обновляем clean_index_ = new_compact_index[старый индекс].
    //   4. Увеличиваем счётчики ссылок новых узлов соответственно.
    //   5. Заменяем старый пул новым.
    //
    // Это не является критичным для текущей версии, но при необходимости
    // может быть реализовано без изменения интерфейсов благодаря реестру.
    // -------------------------------------------------------------------------
#pragma once

#include "node_pool.h"
#include "evaluate_impl.h"
#include "lazy_nodes.h"
#include "simplify_impl.h"
#include "interval.h"
#include "global_state.h"
#include <stack>
#include <cassert>
#include <algorithm>
#include <optional>
#include <vector>
#include <string>
#include <cstdint> 

    namespace delta {

    // ------------------------------------------------------------------------
    // Вспомогательные функции для грязного дерева (интервал)
    // ------------------------------------------------------------------------
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
            for (int child : dn.children) st.push(child);
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
                for (int child : dn.children) {
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
                for (int child : dn.children) {
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
    // Вычисление интервала для чистого дерева (по требованию)
    // ------------------------------------------------------------------------
    inline internal::Interval compute_interval_clean(int root) {
        const auto& nodes = internal::pool.nodes;
        const auto& values = internal::pool.values;
        std::vector<internal::Interval> intervals(nodes.size());
        std::stack<int> st;
        st.push(root);
        std::vector<int> postorder;
        while (!st.empty()) {
            int idx = st.top(); st.pop();
            postorder.push_back(idx);
            for (int child : nodes[idx].children) st.push(child);
        }
        for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
            int idx = *it;
            const auto& node = nodes[idx];
            switch (node.op) {
            case internal::LazyOp::CONST:
                intervals[idx] = internal::Interval(internal::to_double(values[node.value_idx]));
                break;
            case internal::LazyOp::SUM: {
                internal::Interval sum = internal::Interval::zero();
                for (const auto& v : node.leaf_values)
                    sum = sum + internal::Interval(internal::to_double(v));
                for (int child : node.children)
                    sum = sum + intervals[child];
                intervals[idx] = sum;
                break;
            }
            case internal::LazyOp::PRODUCT: {
                internal::Interval prod = internal::Interval::one();
                for (const auto& v : node.leaf_values)
                    prod = prod * internal::Interval(internal::to_double(v));
                for (int child : node.children)
                    prod = prod * intervals[child];
                intervals[idx] = prod;
                break;
            }
            case internal::LazyOp::NEG:
                intervals[idx] = -intervals[node.children[0]];
                break;
            case internal::LazyOp::RECIP: {
                const auto& child_int = intervals[node.children[0]];
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
                const auto& child_int = intervals[node.children[0]];
                if (child_int.upper() < 0) intervals[idx] = internal::Interval();
                else {
                    double lo = child_int.lower() < 0 ? 0.0 : std::sqrt(child_int.lower());
                    double hi = std::sqrt(child_int.upper());
                    intervals[idx] = internal::Interval(lo, hi);
                }
                break;
            }
            case internal::LazyOp::EXP: {
                const auto& child_int = intervals[node.children[0]];
                intervals[idx] = internal::Interval(std::exp(child_int.lower()), std::exp(child_int.upper()));
                break;
            }
            case internal::LazyOp::LOG: {
                const auto& child_int = intervals[node.children[0]];
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
                const auto& child_int = intervals[node.children[0]];
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
                const auto& base_int = intervals[node.children[0]];
                const auto& exp_int = intervals[node.children[1]];
                double lo = std::pow(base_int.lower(), exp_int.lower());
                double hi = std::pow(base_int.upper(), exp_int.upper());
                intervals[idx] = internal::Interval(lo, hi);
                break;
            }
            default: throw std::logic_error("compute_interval_clean: unknown op");
            }
        }
        return intervals[root];
    }
    // RAII-структура для отключения GC и временного снятия лимита размера пула
    // RAII-структура для отключения GC и временного снятия лимита размера пула
    struct CanonicalizeGuard {
        bool old_gc_disabled;
        size_t old_max_size;
        size_t old_gc_threshold;
        bool expanded;

        CanonicalizeGuard(size_t needed_nodes)
            : old_gc_disabled(internal::gc_disabled)
            , old_max_size(internal::pool.max_size)
            , old_gc_threshold(internal::pool.gc_threshold)
            , expanded(false)
        {
            internal::gc_disabled = true;

            if (internal::pool.next_free_index + needed_nodes > internal::pool.max_size) {
                internal::pool.max_size = internal::DEFAULT_POOL_MAX_SIZE;
                internal::pool.update_gc_threshold();
                expanded = true;
            }
        }

        ~CanonicalizeGuard() noexcept(false) {
            // Восстанавливаем состояние пула и флагов СРАЗУ
            internal::gc_disabled = old_gc_disabled;
            internal::pool.max_size = old_max_size;
            internal::pool.gc_threshold = old_gc_threshold;

            // Если расширяли, но после канонизации не влезли в исходный лимит — бросаем исключение
            if (expanded && internal::pool.next_free_index > old_max_size) {
                throw std::runtime_error(
                    "Canonicalization requires more nodes than max_size allows. "
                    "Increase max_size before building expression. "
                    "To silently expand pool, comment out this throw."
                );
            }
        }
    };
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

    // Move-конструктор с поддержкой реестра чистых объектов
    inline LazyRational::LazyRational(LazyRational&& other) noexcept
        : state_(other.state_),
        nodes_(std::move(other.nodes_)),
        constants_(std::move(other.constants_)),
        root_(other.root_),
        clean_index_(other.clean_index_),
        cached_interval_(std::move(other.cached_interval_))
    {
        if (other.state_ == State::Clean) {
            other.unregister_clean();   // other теряет чистоту
        }
        other.state_ = State::Dirty;
        other.root_ = -1;
        other.clean_index_ = -1;
        other.cached_interval_.reset();

        if (state_ == State::Clean) {
            register_clean();           // новый объект чистый
        }
    }

    inline LazyRational& LazyRational::operator=(LazyRational&& other) noexcept {
        if (this != &other) {
            this->~LazyRational();
            new (this) LazyRational(std::move(other));
        }
        return *this;
    }

    // Деструктор с дерегистрацией, если объект был чистым
    inline LazyRational::~LazyRational() {
        if (state_ == State::Clean) {
            unregister_clean();
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
        absl::InlinedVector<int32_t, 2> children,
        int value_idx,
        int eps_idx) {
        assert(state_ == State::Dirty);
        if (op == internal::LazyOp::CONST) {
            nodes_.emplace_back(op, value_idx);
        }
        else if (op == internal::LazyOp::SUM || op == internal::LazyOp::PRODUCT) {
            nodes_.emplace_back(op, std::vector<internal::Value>{}, absl::InlinedVector<int32_t, 2>{});
        }
        else {
            nodes_.emplace_back(op, std::move(children), eps_idx);
        }
        return static_cast<int>(nodes_.size() - 1);
    }

    // ------------------------------------------------------------------------
    // import_tree – копирование поддерева в грязное состояние
    // ------------------------------------------------------------------------
    inline int LazyRational::import_tree(const LazyRational& other) {
        assert(state_ == State::Dirty);

        if (this == &other) {
            LazyRational temp = other.clone();
            return import_tree(temp);
        }

        if (other.state_ == State::Dirty) {
            std::vector<int> old_to_new(other.nodes_.size(), -1);
            std::vector<int> old_const_to_new(other.constants_.size(), -1);

            std::stack<int> st;
            st.push(other.root_);
            std::vector<int> postorder;
            while (!st.empty()) {
                int idx = st.top(); st.pop();
                postorder.push_back(idx);
                const auto& dn = other.nodes_[idx];
                for (int child : dn.children) st.push(child);
            }

            for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
                int old_idx = *it;
                const auto& old_node = other.nodes_[old_idx];
                int new_idx = -1;

                if (old_node.op == internal::LazyOp::CONST) {
                    if (old_const_to_new[old_node.value_idx] == -1) {
                        old_const_to_new[old_node.value_idx] =
                            add_constant(other.constants_[old_node.value_idx]);
                    }
                    int new_const = old_const_to_new[old_node.value_idx];
                    new_idx = new_dirty_node(old_node.op, {}, new_const, -1);
                }
                else if (old_node.op == internal::LazyOp::SUM || old_node.op == internal::LazyOp::PRODUCT) {
                    absl::InlinedVector<int32_t, 2> new_complex;
                    for (int child : old_node.children) {
                        new_complex.push_back(old_to_new[child]);
                    }
                    std::vector<internal::Value> new_leaf = old_node.leaf_values;
                    int new_node_idx = static_cast<int>(nodes_.size());
                    nodes_.emplace_back(old_node.op, std::move(new_leaf), std::move(new_complex));
                    new_idx = new_node_idx;
                }
                else {
                    absl::InlinedVector<int32_t, 2> new_children;
                    for (int child : old_node.children) {
                        new_children.push_back(old_to_new[child]);
                    }
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
        else {
            LazyRational temp;
            temp.state_ = State::Dirty;

            std::stack<int> st;
            st.push(other.clean_index_);
            std::vector<int> postorder;
            while (!st.empty()) {
                int idx = st.top(); st.pop();
                postorder.push_back(idx);
                const auto& node = internal::pool.nodes[idx];
                for (int child : node.children) st.push(child);
            }

            std::vector<int> clean_to_dirty(internal::pool.nodes.size(), -1);
            std::vector<int> value_idx_map(internal::pool.values.size(), -1);

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
                    absl::InlinedVector<int32_t, 2> new_complex;
                    for (int child : clean_node.children) {
                        new_complex.push_back(clean_to_dirty[child]);
                    }
                    std::vector<internal::Value> new_leaf = clean_node.leaf_values;
                    int new_node_idx = static_cast<int>(temp.nodes_.size());
                    temp.nodes_.emplace_back(clean_node.op, std::move(new_leaf), std::move(new_complex));
                    dirty_idx = new_node_idx;
                }
                else {
                    absl::InlinedVector<int32_t, 2> new_children;
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
            copy.register_clean();   // <-- ДОБАВИТЬ
            return copy;
        }
    }

    // ------------------------------------------------------------------------
    // ensure_dirty – переход в грязное состояние с дерегистрацией
    // ------------------------------------------------------------------------
    inline void LazyRational::ensure_dirty() {
        if (state_ == State::Clean) {
            unregister_clean();   // удаляем из реестра чистых объектов

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
                for (int child : node.children) st.push(child);
            }

            std::vector<int> clean_to_dirty(internal::pool.nodes.size(), -1);
            std::vector<int> value_idx_map(internal::pool.values.size(), -1);

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
                    absl::InlinedVector<int32_t, 2> new_complex;
                    for (int child : clean_node.children) {
                        new_complex.push_back(clean_to_dirty[child]);
                    }
                    std::vector<internal::Value> new_leaf = clean_node.leaf_values;
                    int new_node_idx = static_cast<int>(temp.nodes_.size());
                    temp.nodes_.emplace_back(clean_node.op, std::move(new_leaf), std::move(new_complex));
                    dirty_idx = new_node_idx;
                }
                else {
                    absl::InlinedVector<int32_t, 2> new_children;
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
            for (int child : other_node.children) {
                target.children.push_back(child);
            }
        }
        else if (other_node.op == internal::LazyOp::CONST) {
            target.leaf_values.push_back(constants_[other_node.value_idx]);
        }
        else {
            target.children.push_back(other_root);
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
            for (int child : other_node.children) {
                target.children.push_back(child);
            }
        }
        else if (other_node.op == internal::LazyOp::CONST) {
            target.leaf_values.push_back(constants_[other_node.value_idx]);
        }
        else {
            target.children.push_back(other_root);
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
            absl::InlinedVector<int32_t, 2> children;
            if (a.nodes_[root].op == internal::LazyOp::CONST) {
                leaf_vals.push_back(a.constants_[a.nodes_[root].value_idx]);
            }
            else {
                children.push_back(root);
            }
            if (b_root != -1) {
                const auto& b_node = a.nodes_[b_root];
                if (b_node.op == internal::LazyOp::CONST) {
                    leaf_vals.push_back(a.constants_[b_node.value_idx]);
                }
                else {
                    children.push_back(b_root);
                }
            }
            int new_root = a.new_dirty_node(internal::LazyOp::SUM, {}, -1, -1);
            a.nodes_[new_root].leaf_values = std::move(leaf_vals);
            a.nodes_[new_root].children = std::move(children);
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
            absl::InlinedVector<int32_t, 2> children;
            if (a.nodes_[root].op == internal::LazyOp::CONST) {
                leaf_vals.push_back(a.constants_[a.nodes_[root].value_idx]);
            }
            else {
                children.push_back(root);
            }
            leaf_vals.push_back(b.value());
            int new_root = a.new_dirty_node(internal::LazyOp::SUM, {}, -1, -1);
            a.nodes_[new_root].leaf_values = std::move(leaf_vals);
            a.nodes_[new_root].children = std::move(children);
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
            absl::InlinedVector<int32_t, 2> children;
            if (a.nodes_[root].op == internal::LazyOp::CONST) {
                leaf_vals.push_back(a.constants_[a.nodes_[root].value_idx]);
            }
            else {
                children.push_back(root);
            }
            if (b_root != -1) {
                const auto& b_node = a.nodes_[b_root];
                if (b_node.op == internal::LazyOp::CONST) {
                    leaf_vals.push_back(a.constants_[b_node.value_idx]);
                }
                else {
                    children.push_back(b_root);
                }
            }
            int new_root = a.new_dirty_node(internal::LazyOp::PRODUCT, {}, -1, -1);
            a.nodes_[new_root].leaf_values = std::move(leaf_vals);
            a.nodes_[new_root].children = std::move(children);
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
            absl::InlinedVector<int32_t, 2> children;
            if (a.nodes_[root].op == internal::LazyOp::CONST) {
                leaf_vals.push_back(a.constants_[a.nodes_[root].value_idx]);
            }
            else {
                children.push_back(root);
            }
            leaf_vals.push_back(b.value());
            int new_root = a.new_dirty_node(internal::LazyOp::PRODUCT, {}, -1, -1);
            a.nodes_[new_root].leaf_values = std::move(leaf_vals);
            a.nodes_[new_root].children = std::move(children);
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

    // Операторы для случая typeof(LValue)==Rational && typeof(Rvalue)==LazyRational.
    // ------------------------------------------------------------------------
    inline LazyRational mutating_unary_minus(LazyRational&& a) {
        a.ensure_dirty();
        a.invalidate_interval();
        int root = a.root_;
        int neg_root = a.new_dirty_node(internal::LazyOp::NEG, { root }, -1, -1);
        a.root_ = neg_root;
        return std::move(a);
    }

    // ------------------------------------------------------------------------
    // Операторы с Rational слева (Rational +/*/-/ / LazyRational)
    // ------------------------------------------------------------------------
    inline LazyRational& operator+(const Rational& a, LazyRational& b) {
        return b += a;
    }

    inline LazyRational&& operator+(const Rational& a, LazyRational&& b) {
        return std::move(b += a);
    }

    inline LazyRational operator-(const Rational& a, const LazyRational& b) {
        LazyRational result = -b;
        result += a;
        return result;
    }

    inline LazyRational operator-(const Rational& a, LazyRational&& b) {
        LazyRational result = mutating_unary_minus(std::move(b));
        result += a;
        return result;
    }

    inline LazyRational& operator*(const Rational& a, LazyRational& b) {
        return b *= a;
    }

    inline LazyRational&& operator*(const Rational& a, LazyRational&& b) {
        return std::move(b *= a);
    }

    inline LazyRational operator/(const Rational& a, const LazyRational& b) {
        LazyRational recip = b.clone();
        recip.ensure_dirty();
        recip.invalidate_interval();
        int b_root = recip.root_;
        int recip_root = recip.new_dirty_node(internal::LazyOp::RECIP, { b_root }, -1, -1);
        recip.root_ = recip_root;
        recip *= a;
        return recip;
    }

    inline LazyRational operator/(const Rational& a, LazyRational&& b) {
        b.ensure_dirty();
        b.invalidate_interval();
        int b_root = b.root_;
        int recip_root = b.new_dirty_node(internal::LazyOp::RECIP, { b_root }, -1, -1);
        b.root_ = recip_root;
        b *= a;
        return std::move(b);
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
            absl::InlinedVector<int32_t, 2> children;
            if (nodes_[root_].op == internal::LazyOp::CONST) {
                leaf_vals.insert(leaf_vals.begin(), constants_[nodes_[root_].value_idx]);
            }
            else {
                children.push_back(root_);
            }
            int new_root = new_dirty_node(internal::LazyOp::SUM, {}, -1, -1);
            nodes_[new_root].leaf_values = std::move(leaf_vals);
            nodes_[new_root].children = std::move(children);
            root_ = new_root;
        }
    }

    inline void LazyRational::append_nodes(std::vector<int>&& node_indices) {
        ensure_dirty();
        invalidate_interval();
        absl::InlinedVector<int32_t, 2> children(node_indices.begin(), node_indices.end());
        if (nodes_[root_].op == internal::LazyOp::SUM) {
            auto& complex = nodes_[root_].children;
            for (int idx : children) {
                complex.push_back(idx);
            }
        }
        else {
            if (nodes_[root_].op != internal::LazyOp::SUM) {
                children.insert(children.begin(), root_);
            }
            int new_root = new_dirty_node(internal::LazyOp::SUM, {}, -1, -1);
            nodes_[new_root].children = std::move(children);
            root_ = new_root;
        }
    }

    // ------------------------------------------------------------------------
    // Канонизация (Dirty -> Clean) с регистрацией в реестре
    // ------------------------------------------------------------------------
    inline void LazyRational::canonicalize() const {
        if (state_ != State::Dirty) return;

        // ------------------------------------------------------------
        // 1. Построение временных узлов TempNode из грязного дерева
        // ------------------------------------------------------------
        std::vector<internal::TempNode> temp_nodes;
        std::vector<internal::Value> temp_values;
        std::vector<int> dirty_to_temp(nodes_.size(), -1);

        // Обход в пост-порядке для грязных узлов
        std::stack<int> st;
        st.push(root_);
        std::vector<int> postorder;
        while (!st.empty()) {
            int idx = st.top(); st.pop();
            postorder.push_back(idx);
            const auto& dn = nodes_[idx];
            for (int child : dn.children) st.push(child);
        }

        for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
            int dirty_idx = *it;
            const auto& dn = nodes_[dirty_idx];

            // Дети текущего узла (уже преобразованные в TempNode)
            std::vector<int> temp_complex;
            for (int child : dn.children) {
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

            std::vector<internal::Value> leaf_vals = dn.leaf_values;

            // Вычисление хеша (без approx/depth)
            uint64_t hash = static_cast<uint64_t>(dn.op);
            if (dn.op == internal::LazyOp::CONST) {
                hash = internal::compute_hash_const(temp_values[value_idx]);
            }
            else if (dn.op == internal::LazyOp::SUM) {
                for (const auto& v : leaf_vals) hash = absl::HashOf(hash, v);
                for (int c : temp_complex) {
                    hash = internal::combine_hash(internal::LazyOp::SUM, hash, temp_nodes[c].hash);
                }
            }
            else if (dn.op == internal::LazyOp::PRODUCT) {
                for (const auto& v : leaf_vals) hash = absl::HashOf(hash, v);
                for (int c : temp_complex) {
                    hash = internal::combine_hash(internal::LazyOp::PRODUCT, hash, temp_nodes[c].hash);
                }
            }
            else if (dn.op == internal::LazyOp::NEG || dn.op == internal::LazyOp::RECIP ||
                dn.op == internal::LazyOp::SQRT || dn.op == internal::LazyOp::EXP ||
                dn.op == internal::LazyOp::LOG || dn.op == internal::LazyOp::SIN ||
                dn.op == internal::LazyOp::COS || dn.op == internal::LazyOp::ACOS) {
                int c = temp_complex[0];
                hash = internal::combine_hash(dn.op, temp_nodes[c].hash, 0, eps_idx);
            }
            else if (dn.op == internal::LazyOp::PI || dn.op == internal::LazyOp::E) {
                hash = internal::combine_hash(dn.op, 0, eps_idx);
            }
            else if (dn.op == internal::LazyOp::POW) {
                int base = temp_complex[0];
                int exp = temp_complex[1];
                hash = internal::combine_hash(internal::LazyOp::POW, temp_nodes[base].hash, temp_nodes[exp].hash, eps_idx);
            }
            else {
                throw std::logic_error("canonicalize: unknown LazyOp");
            }

            int temp_idx = static_cast<int>(temp_nodes.size());
            if (dn.op == internal::LazyOp::SUM || dn.op == internal::LazyOp::PRODUCT) {
                temp_nodes.emplace_back(dn.op, std::move(leaf_vals), std::move(temp_complex),
                    value_idx, eps_idx, hash);
            }
            else {
                temp_nodes.emplace_back(dn.op, std::move(temp_complex), value_idx, eps_idx, hash);
            }
            dirty_to_temp[dirty_idx] = temp_idx;
        }

        int temp_root = dirty_to_temp[root_];
        int simplified_root = internal::simplify_tree(temp_nodes, temp_values, temp_root);

        // ------------------------------------------------------------
        // 2. Оценка количества узлов, которые будут созданы в пуле
        // ------------------------------------------------------------
        size_t needed_nodes = temp_nodes.size();   // каждый TempNode → один чистый узел

        // ------------------------------------------------------------
        // 3. Попытка уложиться в лимит max_size с помощью GC
        // ------------------------------------------------------------
        bool use_guard = false;
        if (internal::pool.next_free_index + needed_nodes > internal::pool.max_size) {
            // Не хватает места → пробуем собрать мусор
            internal::collect_garbage();
            // После GC проверяем снова
            if (internal::pool.next_free_index + needed_nodes > internal::pool.max_size) {
                // Даже после GC не влезаем → придётся отключать GC и снимать лимит (старый путь)
                use_guard = true;
            }
        }

        // ------------------------------------------------------------
        // 4. Лямбда, выполняющая реальное выделение узлов в пул
        // ------------------------------------------------------------
        auto allocate_global_nodes = [&]() -> int {
            std::vector<int> temp_to_global(temp_nodes.size(), -1);
            std::stack<int> st_glob;
            st_glob.push(simplified_root);
            std::vector<int> postorder_glob;
            while (!st_glob.empty()) {
                int idx = st_glob.top(); st_glob.pop();
                postorder_glob.push_back(idx);
                const auto& tn = temp_nodes[idx];
                for (int c : tn.children) st_glob.push(c);
            }

            for (auto it = postorder_glob.rbegin(); it != postorder_glob.rend(); ++it) {
                int idx = *it;
                const auto& tn = temp_nodes[idx];
                int global_idx = -1;

                if (tn.op == internal::LazyOp::CONST) {
                    global_idx = internal::add_const(temp_values[tn.value_idx]);
                }
                else if (tn.op == internal::LazyOp::SUM) {
                    absl::InlinedVector<int32_t, 2> children;
                    for (int c : tn.children) children.push_back(temp_to_global[c]);
                    global_idx = internal::make_sum_node(tn.leaf_values, std::move(children));
                }
                else if (tn.op == internal::LazyOp::PRODUCT) {
                    absl::InlinedVector<int32_t, 2> children;
                    for (int c : tn.children) children.push_back(temp_to_global[c]);
                    global_idx = internal::make_product_node(tn.leaf_values, std::move(children));
                }
                else {
                    absl::InlinedVector<int32_t, 2> children;
                    for (int c : tn.children) children.push_back(temp_to_global[c]);
                    int eps_global = (tn.eps_idx != -1) ? internal::pool.add_value(temp_values[tn.eps_idx]) : -1;
                    global_idx = internal::get_unary_node(tn.op, std::move(children), eps_global);
                }
                temp_to_global[idx] = global_idx;
            }
            return temp_to_global[simplified_root];
            };

        // ------------------------------------------------------------
        // 5. Выделение узлов (с guard или без)
        // ------------------------------------------------------------
        int new_clean_root;
        if (use_guard) {
            // Старый режим: отключаем GC и при необходимости временно расширяем пул
            CanonicalizeGuard guard(needed_nodes);
            new_clean_root = allocate_global_nodes();
        }
        else {
            // Новый режим: GC разрешён, лимит max_size соблюдается
            new_clean_root = allocate_global_nodes();
        }

        // ------------------------------------------------------------
        // 6. Переключение объекта в чистое состояние
        // ------------------------------------------------------------
        internal::increment_ref(new_clean_root);

        const_cast<LazyRational*>(this)->state_ = State::Clean;
        const_cast<LazyRational*>(this)->clean_index_ = new_clean_root;
        const_cast<LazyRational*>(this)->nodes_.clear();
        const_cast<LazyRational*>(this)->constants_.clear();
        const_cast<LazyRational*>(this)->root_ = -1;
        const_cast<LazyRational*>(this)->cached_interval_.reset();

        const_cast<LazyRational*>(this)->register_clean();
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
            result = compute_interval_clean(clean_index_);
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
        if (a.is_clean() && b.is_clean() && a.clean_index_ == b.clean_index_)
            return true;

        if (!a.approx_interval().overlaps(b.approx_interval()))
            return false;

        a.canonicalize();
        b.canonicalize();

        if (a.clean_index_ == b.clean_index_)
            return true;

        return a.eval() == b.eval();
    }

    inline bool operator!=(const LazyRational& a, const LazyRational& b) {
        return !(a == b);
    }

    inline bool operator<(const LazyRational& a, const LazyRational& b) {
        internal::Interval ia = a.approx_interval();
        internal::Interval ib = b.approx_interval();

        if (ia.upper() < ib.lower()) return true;
        if (ia.lower() >= ib.upper()) return false;

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