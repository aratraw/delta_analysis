#pragma once

#include "expression_root.h"
#include "evaluation_core.h"  // for to_double
#include <functional>
#include <vector>
#include <stack>   // <-- добавлено

//ОБЕ СВОБОДНЫЕ ФУНКЦИИ В ЭТОМ ФАЙЛЕ ДОЛЖНЫ БЫТЬ inline ИНАЧЕ ДВИЖОК НЕ ПОЕДЕТ.
namespace delta::internal {

    inline bool structurally_equal(const ExpressionRoot& a, const ExpressionRoot& b) {
        if (&a == &b) return true;
        if (a.hash() != b.hash()) return false;

        const auto& nodes_a = a.nodes();
        const auto& nodes_b = b.nodes();
        const auto& values_a = a.values();
        const auto& values_b = b.values();

        std::function<bool(int, int)> compare = [&](int idx_a, int idx_b) -> bool {
            if (idx_a == idx_b) return true;
            const Node& na = nodes_a[idx_a];
            const Node& nb = nodes_b[idx_b];
            if (na.op != nb.op) return false;
            if (na.op == LazyOp::CONST) {
                const Value& va = values_a[na.value_idx];
                const Value& vb = values_b[nb.value_idx];
                return va == vb;
            }
            bool eq = true;
            if (na.child0 != -1 && nb.child0 != -1)
                eq = eq && compare(na.child0, nb.child0);
            else if (na.child0 != -1 || nb.child0 != -1)
                return false;
            if (na.child1 != -1 && nb.child1 != -1)
                eq = eq && compare(na.child1, nb.child1);
            else if (na.child1 != -1 || nb.child1 != -1)
                return false;
            return eq;
            };
        return compare(a.root_index(), b.root_index());
    }

   inline ExpressionRoot simplify(const ExpressionRoot& root) {
        const auto& nodes = root.nodes();
        const auto& values = root.values();
        if (nodes.empty()) return root;

        // Маппинг старых индексов на новые
        std::vector<int> simplified_idx(nodes.size(), -1);
        std::vector<Node> new_nodes;
        std::vector<Value> new_values;

        auto add_const = [&](const Value& val) -> int {
            int val_idx = static_cast<int>(new_values.size());
            new_values.push_back(val);
            Node const_node(LazyOp::CONST, -1, -1, val_idx,
                Interval(to_double(val)), 0);
            new_nodes.push_back(const_node);
            return static_cast<int>(new_nodes.size()) - 1;
            };

        // Итеративный DFS с явным стеком
        struct Frame {
            int old_idx;
            int state; // 0 = children not processed, 1 = ready to create node
        };
        std::stack<Frame> st;
        // Запускаем обход с корня
        st.push({ root.root_index(), 0 });

        while (!st.empty()) {
            Frame& f = st.top();
            int old_idx = f.old_idx;

            // Если уже обработан, пропускаем
            if (simplified_idx[old_idx] != -1) {
                st.pop();
                continue;
            }

            const Node& node = nodes[old_idx];
            if (f.state == 0) {
                // Проверяем детей, которые ещё не обработаны
                bool need_children = false;
                if (node.child0 != -1 && simplified_idx[node.child0] == -1) {
                    st.push({ node.child0, 0 });
                    need_children = true;
                }
                if (node.child1 != -1 && simplified_idx[node.child1] == -1) {
                    st.push({ node.child1, 0 });
                    need_children = true;
                }
                if (need_children) {
                    f.state = 1; // после возврата из детей создадим узел
                    continue;
                }
                else {
                    // Нет детей или все уже обработаны – можно создавать узел
                    f.state = 1;
                }
            }

            if (f.state == 1) {
                // Получаем индексы уже упрощённых детей
                int new_child0 = (node.child0 != -1) ? simplified_idx[node.child0] : -1;
                int new_child1 = (node.child1 != -1) ? simplified_idx[node.child1] : -1;

                // Функция для получения значения константы из нового узла (если это CONST)
                auto get_const_value = [&](int idx) -> const Value* {
                    if (idx < 0) return nullptr;
                    const Node& n = new_nodes[idx];
                    if (n.op == LazyOp::CONST)
                        return &new_values[n.value_idx];
                    return nullptr;
                    };

                Value zero_val = SmallStorage(absl::int128(0));
                Value one_val = SmallStorage(absl::int128(1));

                // Лямбда для создания узла (применяет правила упрощения)
                auto simplify_node = [&]() -> int {
                    switch (node.op) {
                    case LazyOp::CONST:
                        return add_const(values[node.value_idx]);

                    case LazyOp::ADD: {
                        const Value* left = get_const_value(new_child0);
                        const Value* right = get_const_value(new_child1);
                        if (left && *left == zero_val) return new_child1;
                        if (right && *right == zero_val) return new_child0;
                        break;
                    }
                    case LazyOp::SUB: {
                        const Value* right = get_const_value(new_child1);
                        if (right && *right == zero_val) return new_child0;
                        if (new_child0 == new_child1) {
                            return add_const(zero_val);
                        }
                        break;
                    }
                    case LazyOp::MUL: {
                        const Value* left = get_const_value(new_child0);
                        const Value* right = get_const_value(new_child1);
                        if (left && *left == one_val) return new_child1;
                        if (right && *right == one_val) return new_child0;
                        if (left && *left == zero_val) return add_const(zero_val);
                        if (right && *right == zero_val) return add_const(zero_val);
                        break;
                    }
                    case LazyOp::DIV: {
                        const Value* right = get_const_value(new_child1);
                        if (right && *right == one_val) return new_child0;
                        if (new_child0 == new_child1) {
                            const Value* x_val = get_const_value(new_child0);
                            if (!x_val || *x_val != zero_val)
                                return add_const(one_val);
                        }
                        break;
                    }
                    case LazyOp::NEG: {
                        const Node& child_node = new_nodes[new_child0];
                        if (child_node.op == LazyOp::NEG && child_node.child0 != -1)
                            return child_node.child0;
                        break;
                    }
                    case LazyOp::SQRT: {
                        const Value* arg = get_const_value(new_child0);
                        if (arg && *arg == zero_val) return add_const(zero_val);
                        if (arg && *arg == one_val) return add_const(one_val);
                        break;
                    }
                    case LazyOp::EXP: {
                        const Value* arg = get_const_value(new_child0);
                        if (arg && *arg == zero_val) return add_const(one_val);
                        if (new_child0 >= 0) {
                            const Node& child_node = new_nodes[new_child0];
                            if (child_node.op == LazyOp::LOG && child_node.child0 != -1)
                                return child_node.child0;
                        }
                        break;
                    }
                    case LazyOp::LOG: {
                        const Value* arg = get_const_value(new_child0);
                        if (arg && *arg == one_val) return add_const(zero_val);
                        if (new_child0 >= 0) {
                            const Node& child_node = new_nodes[new_child0];
                            if (child_node.op == LazyOp::EXP && child_node.child0 != -1)
                                return child_node.child0;
                        }
                        break;
                    }
                    case LazyOp::SIN: {
                        const Value* arg = get_const_value(new_child0);
                        if (arg && *arg == zero_val) return add_const(zero_val);
                        break;
                    }
                    case LazyOp::COS: {
                        const Value* arg = get_const_value(new_child0);
                        if (arg && *arg == zero_val) return add_const(one_val);
                        break;
                    }
                    default:
                        break;
                    }

                    // Нет упрощения: создаём новый узел
                    Interval approx;
                    int depth = 0;
                    if (node.op != LazyOp::CONST) {
                        Interval left_approx, right_approx;
                        if (new_child0 >= 0) left_approx = new_nodes[new_child0].approx;
                        if (new_child1 >= 0) right_approx = new_nodes[new_child1].approx;
                        approx = ExpressionRoot::compute_interval(node.op, left_approx, right_approx);
                        depth = 1 + std::max(
                            (new_child0 >= 0 ? new_nodes[new_child0].depth : 0),
                            (new_child1 >= 0 ? new_nodes[new_child1].depth : 0)
                        );
                    }

                    int val_idx = -1;
                    if (node.op == LazyOp::CONST) {
                        val_idx = node.value_idx;
                    }
                    else if (node.op == LazyOp::SQRT || node.op == LazyOp::EXP ||
                        node.op == LazyOp::LOG || node.op == LazyOp::SIN ||
                        node.op == LazyOp::COS || node.op == LazyOp::ACOS ||
                        node.op == LazyOp::PI || node.op == LazyOp::E) {
                        val_idx = static_cast<int>(new_values.size());
                        new_values.push_back(values[node.value_idx]);
                    }

                    new_nodes.emplace_back(node.op, new_child0, new_child1, val_idx, approx, depth);
                    return static_cast<int>(new_nodes.size()) - 1;
                    };

                int new_idx = simplify_node();
                simplified_idx[old_idx] = new_idx;
                st.pop();
            }
        }

        int new_root_idx = simplified_idx[root.root_index()];
        return ExpressionRoot(std::move(new_nodes), std::move(new_values), new_root_idx);
    }

} // namespace delta::internal