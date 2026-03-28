// evaluate_impl.h
#pragma once

#include "node_pool.h"
#include "evaluation_core.h"
#include <stack>
#include <optional>
#include <vector>

namespace delta::internal {

    // ----------------------------------------------------------------------------
    // compute_node – вычисление узла по операции и уже готовым детям
    // ----------------------------------------------------------------------------
    inline Value compute_node(LazyOp op, const Value& left, const Value& right, const Value& eps) {
        switch (op) {
        case LazyOp::ADD:   return eager_add(left, right);
        case LazyOp::MUL:   return eager_mul(left, right);
        case LazyOp::NEG:   return eager_neg(left);
        case LazyOp::RECIP: return eager_div(Value(SmallStorage(1)), left);
        case LazyOp::SQRT:  return eager_sqrt(left, eps);
        case LazyOp::EXP:   return eager_exp(left, eps);
        case LazyOp::LOG:   return eager_log(left, eps);
        case LazyOp::SIN:   return eager_sin(left, eps);
        case LazyOp::COS:   return eager_cos(left, eps);
        case LazyOp::ACOS:  return eager_acos(left, eps);
        case LazyOp::PI:    return eager_pi(eps);
        case LazyOp::E:     return eager_e(eps);
        default:
            throw std::logic_error("compute_node: unknown LazyOp");
        }
    }

    // ----------------------------------------------------------------------------
    // evaluate – итеративный пост-порядок с локальным кэшем
    // ----------------------------------------------------------------------------
    inline Value evaluate(int root_idx) {
        const auto& nodes = pool.nodes;
        const auto& values = pool.values;
        const size_t n = nodes.size();

        std::vector<std::optional<Value>> cache(n);
        std::stack<int> st;
        st.push(root_idx);

        while (!st.empty()) {
            int idx = st.top();
            if (cache[idx].has_value()) {
                st.pop();
                continue;
            }

            const Node& node = nodes[idx];

            if (node.op == LazyOp::CONST) {
                cache[idx] = values[node.value_idx];
                st.pop();
                continue;
            }

            // Узел без детей (например, PI, E или любые другие будущие константы)
            if (node.child0 == -1 && node.child1 == -1) {
                Value left{}, right{};
                Value eps = node.value_idx != -1 ? values[node.value_idx] : Value{};
                Value result = compute_node(node.op, left, right, eps);
                cache[idx] = result;
                st.pop();
                continue;
            }

            bool need_children = false;
            if (node.child0 != -1 && !cache[node.child0].has_value()) {
                st.push(node.child0);
                need_children = true;
            }
            if (node.child1 != -1 && !cache[node.child1].has_value()) {
                st.push(node.child1);
                need_children = true;
            }
            if (need_children) {
                continue;
            }

            Value left = node.child0 != -1 ? cache[node.child0].value() : Value{};
            Value right = node.child1 != -1 ? cache[node.child1].value() : Value{};
            Value eps = node.value_idx != -1 ? values[node.value_idx] : Value{};
            Value result = compute_node(node.op, left, right, eps);
            cache[idx] = result;
            st.pop();
        }

        return cache[root_idx].value();
    }

} // namespace delta::internal