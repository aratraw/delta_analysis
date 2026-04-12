// simplify.h
#pragma once

#include "expression_root.h"          // для delta::ExpressionRoot
#include "node_pool.h"                // для pool
#include "simplify_impl.h"            // для simplify_impl
#include <stack>

namespace delta::internal {

    // Обёртка над simplify_impl
    inline delta::ExpressionRoot simplify(const delta::ExpressionRoot& root) {
        return delta::ExpressionRoot(simplify_impl(root.root_index()));
    }

    // Структурное сравнение двух выражений
    inline bool structurally_equal(const delta::ExpressionRoot& a, const delta::ExpressionRoot& b) {
        if (a.root_index() == b.root_index()) return true;
        if (a.hash() != b.hash()) return false;

        std::stack<std::pair<int, int>> st;
        st.emplace(a.root_index(), b.root_index());

        while (!st.empty()) {
            auto [idx_a, idx_b] = st.top();
            st.pop();

            const Node& na = pool.nodes[idx_a];
            const Node& nb = pool.nodes[idx_b];

            if (na.op != nb.op) return false;

            // Сравнение SUM узлов
            if (na.op == LazyOp::SUM && nb.op == LazyOp::SUM) {
                if (!na.sum_children || !nb.sum_children) return false;
                if (na.sum_children->size() != nb.sum_children->size()) return false;
                for (size_t i = 0; i < na.sum_children->size(); ++i) {
                    if ((*na.sum_children)[i] != (*nb.sum_children)[i]) return false;
                }
                continue;
            }

            if (na.op == LazyOp::CONST) {
                const Value& va = pool.values[na.value_idx];
                const Value& vb = pool.values[nb.value_idx];
                if (va != vb) return false;
                continue;
            }

            if ((na.child0 == -1) != (nb.child0 == -1)) return false;
            if (na.child0 != -1) st.emplace(na.child0, nb.child0);

            if ((na.child1 == -1) != (nb.child1 == -1)) return false;
            if (na.child1 != -1) st.emplace(na.child1, nb.child1);
        }

        return true;
    }

} // namespace delta::internal