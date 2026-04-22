// evaluate_impl.h
// Версия 3.0 – единая шаблонная функция evaluate_tree с PCR-стратегиями
// ----------------------------------------------------------------------------
// Изменения:
//   - Удалены старые evaluate, eval_dirty, eval_dirty_inplace
//   - Добавлены PCR-функции (pyramidal_compact_reduce_inplace/copy)
//   - Добавлены стратегии суммирования (SumStrategy_Standard, SumStrategy_Inplace)
//   - Реализована шаблонная evaluate_tree, работающая с любыми NodeType
//   - Три публичных API: evaluate(чистое), evaluate_dirty, evaluate_dirty_inplace
//   - Исправлены сигнатуры функций evaluate_dirty* (используют DirtyNode)
//   - Устранена циклическая зависимость: теперь включаются только node_types.h и lazy_nodes.h
//   - Шаблонные параметры стратегий передаются по значению (исправлена ошибка компиляции)
//   - Адаптировано под новый Value (без SmallStorage)
// ----------------------------------------------------------------------------

#pragma once

#include "node_types.h"
#include "lazy_nodes.h"
#include "evaluation_core.h"
#include "reduce.h"        // <-- добавлено: функции PCR вынесены в отдельный заголовок
#include "utils.h"

#include <stack>
#include <vector>
#include <optional>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

namespace delta::internal {

    // ------------------------------------------------------------------------
    // Стратегии суммирования (политика обработки узла SUM)
    // ------------------------------------------------------------------------
    struct SumStrategy_Standard {
        static constexpr bool allows_inplace = false;
        Value operator()(const std::vector<Value>& values) const {
            return pyramidal_compact_reduce_copy(values);
        }
    };

    struct SumStrategy_Inplace {
        static constexpr bool allows_inplace = true;
        Value operator()(std::vector<Value>& values) const {
            pyramidal_compact_reduce_inplace(values);
            return std::move(values[0]);
        }
    };

    // ------------------------------------------------------------------------
    // Стратегия умножения (последовательная, без батчинга)
    // ------------------------------------------------------------------------
    struct ProdStrategy_Sequential {
        Value operator()(std::vector<Value> leaf_values, const std::vector<Value>& child_values) const {
            if (leaf_values.empty() && child_values.empty()) {
                return Value(1);
            }
            Value result = !leaf_values.empty() ? leaf_values[0] : child_values[0];
            size_t start_leaf = !leaf_values.empty() ? 1 : 0;
            size_t start_child = !leaf_values.empty() ? 0 : 1;

            for (size_t i = start_leaf; i < leaf_values.size(); ++i) {
                result *= leaf_values[i];
            }
            for (size_t i = start_child; i < child_values.size(); ++i) {
                result *= child_values[i];
            }
            return result;
        }
    };

    // ------------------------------------------------------------------------
    // Единая шаблонная функция вычисления дерева
    // ------------------------------------------------------------------------
    template<typename NodeType, typename ValueAccessor, typename SumStrategy, typename ProdStrategy>
    Value evaluate_tree(int root,
        const std::vector<NodeType>& nodes,
        ValueAccessor&& value_accessor,
        SumStrategy sum_strategy,       // передача по значению
        ProdStrategy prod_strategy)     // передача по значению
    {
        const size_t n = nodes.size();
        std::vector<std::optional<Value>> cache(n);
        std::stack<int> st;
        st.push(root);

        while (!st.empty()) {
            int idx = st.top();
            if (cache[idx].has_value()) {
                st.pop();
                continue;
            }

            const NodeType& node = nodes[idx];
            bool children_ready = true;

            // Проверяем готовность детей
            if (node.op == LazyOp::SUM || node.op == LazyOp::PRODUCT) {
                for (int child : node.complex_children) {
                    if (!cache[child].has_value()) {
                        st.push(child);
                        children_ready = false;
                    }
                }
            }
            else {
                for (int child : node.children) {
                    if (child != -1 && !cache[child].has_value()) {
                        st.push(child);
                        children_ready = false;
                    }
                }
            }

            if (!children_ready) continue;

            Value result;
            switch (node.op) {
            case LazyOp::CONST: {
                result = value_accessor.const_value(node);
                break;
            }

            case LazyOp::SUM: {
                std::vector<Value> to_reduce;
                if constexpr (SumStrategy::allows_inplace) {
                    to_reduce = std::move(const_cast<NodeType&>(node).leaf_values);
                }
                else {
                    to_reduce = node.leaf_values;
                }
                to_reduce.reserve(to_reduce.size() + node.complex_children.size());
                for (int child : node.complex_children) {
                    to_reduce.push_back(cache[child].value());
                }
                result = sum_strategy(to_reduce);
                break;
            }

            case LazyOp::PRODUCT: {
                std::vector<Value> leaf_vals;
                if constexpr (SumStrategy::allows_inplace) {
                    leaf_vals = std::move(const_cast<NodeType&>(node).leaf_values);
                }
                else {
                    leaf_vals = node.leaf_values;
                }
                std::vector<Value> child_vals;
                child_vals.reserve(node.complex_children.size());
                for (int child : node.complex_children) {
                    child_vals.push_back(cache[child].value());
                }
                result = prod_strategy(std::move(leaf_vals), child_vals);
                break;
            }

            case LazyOp::NEG:
                result = -cache[node.children[0]].value();
                break;
            case LazyOp::RECIP:
                result = Value(1)/cache[node.children[0]].value();
                break;
            case LazyOp::SQRT: {
                Value eps = value_accessor.eps_value(node);
                result = eager_sqrt(cache[node.children[0]].value(), eps);
                break;
            }
            case LazyOp::EXP: {
                Value eps = value_accessor.eps_value(node);
                result = eager_exp(cache[node.children[0]].value(), eps);
                break;
            }
            case LazyOp::LOG: {
                Value eps = value_accessor.eps_value(node);
                result = eager_log(cache[node.children[0]].value(), eps);
                break;
            }
            case LazyOp::SIN: {
                Value eps = value_accessor.eps_value(node);
                result = eager_sin(cache[node.children[0]].value(), eps);
                break;
            }
            case LazyOp::COS: {
                Value eps = value_accessor.eps_value(node);
                result = eager_cos(cache[node.children[0]].value(), eps);
                break;
            }
            case LazyOp::ACOS: {
                Value eps = value_accessor.eps_value(node);
                result = eager_acos(cache[node.children[0]].value(), eps);
                break;
            }
            case LazyOp::PI: {
                Value eps = value_accessor.eps_value(node);
                result = eager_pi(eps);
                break;
            }
            case LazyOp::E: {
                Value eps = value_accessor.eps_value(node);
                result = eager_e(eps);
                break;
            }
            case LazyOp::POW: {
                Value eps = value_accessor.eps_value(node);
                result = eager_pow(cache[node.children[0]].value(),
                    cache[node.children[1]].value(),
                    eps);
                break;
            }
            default:
                throw std::logic_error("evaluate_tree: unknown LazyOp");
            }

            cache[idx] = std::move(result);
            st.pop();
        }

        return std::move(cache[root].value());
    }

    // ------------------------------------------------------------------------
    // Публичные API для чистого дерева (NodePool)
    // ------------------------------------------------------------------------
    // Определение pool находится в node_pool.h, но здесь мы используем только
    // pool.nodes и pool.values, поэтому достаточно объявления (линковка разрешится).
    // Чтобы избежать циклической зависимости, мы не включаем node_pool.h.
    // Вместо этого используем forward-декларацию и предполагаем, что pool доступен.
    // В реальности код ниже будет скомпилирован после включения node_pool.h,
    // где pool уже полностью определён.

    // Вспомогательная функция доступа для чистого дерева
    inline Value evaluate(int root_idx);

    // ------------------------------------------------------------------------
    // Публичные API для грязного дерева (DirtyNode)
    // ------------------------------------------------------------------------

    // evaluate_dirty – вычисление без разрушения дерева
    inline Value evaluate_dirty(const std::vector<DirtyNode>& nodes,
        const std::vector<Value>& constants,
        int root) {
        struct Accessor {
            const std::vector<Value>& constants;
            Value const_value(const DirtyNode& node) const {
                return constants[node.value_idx];
            }
            Value eps_value(const DirtyNode& node) const {
                return (node.eps_idx != -1) ? constants[node.eps_idx] : Value{};
            }
        };
        SumStrategy_Standard sum_strategy;
        ProdStrategy_Sequential prod_strategy;
        return evaluate_tree<DirtyNode>(root, nodes, Accessor{ constants }, sum_strategy, prod_strategy);
    }

    // evaluate_dirty_inplace – вычисление с разрушением leaf_values (оптимизация)
    inline Value evaluate_dirty_inplace(std::vector<DirtyNode>& nodes,
        std::vector<Value>& constants,
        int root) {
        struct Accessor {
            std::vector<Value>& constants;
            Value const_value(const DirtyNode& node) const {   // <-- const&
                return constants[node.value_idx];
            }
            Value eps_value(const DirtyNode& node) const {     // <-- const&
                return (node.eps_idx != -1) ? constants[node.eps_idx] : Value{};
            }
        };
        SumStrategy_Inplace sum_strategy;
        ProdStrategy_Sequential prod_strategy;
        return evaluate_tree<DirtyNode>(root, nodes, Accessor{ constants }, sum_strategy, prod_strategy);
    }

} // namespace delta::internal