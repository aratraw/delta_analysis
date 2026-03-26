// simplify_impl.h
#pragma once

#include "node_pool.h"
#include "evaluation_core.h"   // для is_zero, is_one
#include "evaluate_impl.h"     // для compute_node
#include <stack>
#include <vector>
#include <optional>

namespace delta::internal {

    // ----------------------------------------------------------------------------
    // simplify – итеративное упрощение дерева
    // ----------------------------------------------------------------------------
    inline int simplify_impl(int root_idx) {
        const auto& nodes = pool.nodes;
        const auto& values = pool.values;
        const size_t n = nodes.size();

        // кэш: для каждого исходного индекса храним индекс упрощённого узла
        std::vector<int> simplified_idx(n, -1);
        // стек для обхода в пост-порядке
        std::stack<int> st;
        st.push(root_idx);

        while (!st.empty()) {
            int idx = st.top();
            if (simplified_idx[idx] != -1) {
                st.pop();
                continue;
            }

            const Node& node = nodes[idx];

            // Константа – сразу сохраняем себя
            if (node.op == LazyOp::CONST) {
                simplified_idx[idx] = idx;   // константа остаётся собой
                st.pop();
                continue;
            }


            // Для бинарных и унарных операций проверяем, упрощены ли дети
            bool children_ready = true;
            if (node.child0 != -1 && simplified_idx[node.child0] == -1) {
                st.push(node.child0);
                children_ready = false;
            }
            if (node.child1 != -1 && simplified_idx[node.child1] == -1) {
                st.push(node.child1);
                children_ready = false;
            }
            if (!children_ready) {
                // ждём детей
                continue;
            }

            // Получаем упрощённые индексы детей (если были)
            int child0_simp = node.child0 != -1 ? simplified_idx[node.child0] : -1;
            int child1_simp = node.child1 != -1 ? simplified_idx[node.child1] : -1;

            // Базовое упрощение: если оба ребёнка – константы, вычисляем результат
            bool is_const0 = (child0_simp != -1 && nodes[child0_simp].op == LazyOp::CONST);
            bool is_const1 = (child1_simp != -1 && nodes[child1_simp].op == LazyOp::CONST);

            if (is_const0 && (node.child1 == -1 || is_const1)) {
                // Оба операнда константы – вычисляем результат через eager
                Value left = child0_simp != -1 ? values[nodes[child0_simp].value_idx] : Value{};
                Value right = child1_simp != -1 ? values[nodes[child1_simp].value_idx] : Value{};
                Value eps = node.value_idx != -1 ? values[node.value_idx] : Value{};
                Value const_result = compute_node(node.op, left, right, eps);
                // Создаём новый константный узел (уникальный)
                int const_node = add_const(const_result);
                simplified_idx[idx] = const_node;
                st.pop();
                continue;
            }

            // Правила упрощения для унарных операций
            if (node.child1 == -1) {
                // Если у узла нет детей (PI, E), он уже упрощён
                if (node.child0 == -1 && node.child1 == -1) {
                    simplified_idx[idx] = idx;
                    st.pop();
                    continue;
                }

                int new_idx = idx; // по умолчанию оставляем старый

                switch (node.op) {
                case LazyOp::NEG: {
                    // NEG(NEG(x)) -> x
                    if (child0_simp != -1 && nodes[child0_simp].op == LazyOp::NEG) {
                        int grand = nodes[child0_simp].child0;
                        if (grand != -1) {
                            new_idx = grand;
                            break;
                        }
                    }
                    break;
                }
                case LazyOp::RECIP: {
                    // RECIP(RECIP(x)) -> x
                    if (child0_simp != -1 && nodes[child0_simp].op == LazyOp::RECIP) {
                        int grand = nodes[child0_simp].child0;
                        if (grand != -1) {
                            new_idx = grand;
                            break;
                        }
                    }
                    break;
                }
                case LazyOp::EXP: {
                    // EXP(LOG(x)) -> x
                    if (child0_simp != -1 && nodes[child0_simp].op == LazyOp::LOG) {
                        int grand = nodes[child0_simp].child0;
                        if (grand != -1) {
                            new_idx = grand;
                            break;
                        }
                    }
                    break;
                }
                case LazyOp::LOG: {
                    // LOG(EXP(x)) -> x
                    if (child0_simp != -1 && nodes[child0_simp].op == LazyOp::EXP) {
                        int grand = nodes[child0_simp].child0;
                        if (grand != -1) {
                            new_idx = grand;
                            break;
                        }
                    }
                    break;
                }
                default: break;
                }

                if (new_idx != idx) {
                    simplified_idx[idx] = new_idx;
                    st.pop();
                    continue;
                }

                // Если упрощение не сработало, создаём новый унарный узел (с интернированием)
                int new_unary = get_unary_node(node.op, child0_simp, node.value_idx);
                simplified_idx[idx] = new_unary;
                st.pop();
                continue;
            }

            // Бинарные операции
            const Node* left_node = child0_simp != -1 ? &nodes[child0_simp] : nullptr;
            const Node* right_node = child1_simp != -1 ? &nodes[child1_simp] : nullptr;
            bool left_const = left_node && left_node->op == LazyOp::CONST;
            bool right_const = right_node && right_node->op == LazyOp::CONST;
            Value left_val = left_const ? values[left_node->value_idx] : Value{};
            Value right_val = right_const ? values[right_node->value_idx] : Value{};

            int new_idx = idx; // по умолчанию

            if (node.op == LazyOp::ADD) {
                // x + 0 -> x
                if (right_const && is_zero(right_val)) {
                    new_idx = child0_simp;
                }
                else if (left_const && is_zero(left_val)) {
                    new_idx = child1_simp;
                }
                // x + (-x) -> 0
                else if (child0_simp != -1 && child1_simp != -1) {
                    const Node& right_n = nodes[child1_simp];
                    if (right_n.op == LazyOp::NEG && right_n.child0 == child0_simp) {
                        new_idx = add_const(Value(SmallStorage(0)));
                    }
                    const Node& left_n = nodes[child0_simp];
                    if (left_n.op == LazyOp::NEG && left_n.child0 == child1_simp) {
                        new_idx = add_const(Value(SmallStorage(0)));
                    }
                }
            }
            else if (node.op == LazyOp::MUL) {
                // x * 1 -> x
                if (right_const && is_one(right_val)) {
                    new_idx = child0_simp;
                }
                else if (left_const && is_one(left_val)) {
                    new_idx = child1_simp;
                }
                // x * 0 -> 0
                else if (right_const && is_zero(right_val)) {
                    new_idx = add_const(Value(SmallStorage(0)));
                }
                else if (left_const && is_zero(left_val)) {
                    new_idx = add_const(Value(SmallStorage(0)));
                }
                // x * (1/x) -> 1 (если x не ноль)
                else if (child0_simp != -1 && child1_simp != -1) {
                    const Node& right_n = nodes[child1_simp];
                    if (right_n.op == LazyOp::RECIP && right_n.child0 == child0_simp) {
                        new_idx = add_const(Value(SmallStorage(1)));
                    }
                    const Node& left_n = nodes[child0_simp];
                    if (left_n.op == LazyOp::RECIP && left_n.child0 == child1_simp) {
                        new_idx = add_const(Value(SmallStorage(1)));
                    }
                }
            }

            if (new_idx != idx) {
                simplified_idx[idx] = new_idx;
                st.pop();
                continue;
            }

            // Нет упрощения – создаём новый бинарный узел через интернирование
            int new_binary = get_binary_node(node.op, child0_simp, child1_simp);
            simplified_idx[idx] = new_binary;
            st.pop();
        }

        return simplified_idx[root_idx];
    }

} // namespace delta::internal