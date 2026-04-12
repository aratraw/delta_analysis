// simplify_impl.h (адаптирован под новый tagged union Value, флаг small_reduced в Value)
#pragma once

#include "node_pool.h"
#include "evaluation_core.h"
#include "evaluate_impl.h"
#include "context.h"
#include <stack>
#include <vector>
#include <optional>

namespace delta::internal {

    inline bool is_algebraic(LazyOp op) {
        return op == LazyOp::ADD || op == LazyOp::MUL ||
            op == LazyOp::NEG || op == LazyOp::RECIP;
    }

    // ----------------------------------------------------------------------------
    // collect_add_operands – собирает все листья (не ADD) из бинарного дерева ADD
    // ----------------------------------------------------------------------------
    inline void collect_add_operands(int idx, std::vector<int>& out) {
        std::vector<int> stack;
        stack.reserve(256);
        stack.push_back(idx);
        while (!stack.empty()) {
            int cur = stack.back();
            stack.pop_back();
            const Node& node = pool.nodes[cur];
            if (node.op == LazyOp::ADD) {
                if (node.child1 != -1) stack.push_back(node.child1);
                if (node.child0 != -1) stack.push_back(node.child0);
            }
            else {
                out.push_back(cur);
            }
        }
    }

    inline int simplify_impl(int root_idx) {
        const auto& nodes = pool.nodes;
        const auto& values = pool.values;
        const size_t n = nodes.size();

        std::vector<int> simplified_idx(n, -1);
        std::stack<int> st;
        st.push(root_idx);

        while (!st.empty()) {
            int idx = st.top();
            if (simplified_idx[idx] != -1) {
                st.pop();
                continue;
            }

            const Node& node = nodes[idx];

            if (node.op == LazyOp::CONST) {
                simplified_idx[idx] = idx;
                st.pop();
                continue;
            }

            // --------------------------------------------------------------------
            // НОВАЯ ВЕТКА ДЛЯ LazyOp::SUM
            // --------------------------------------------------------------------
            if (node.op == LazyOp::SUM) {
                // Сначала убеждаемся, что все дети упрощены
                bool children_ready = true;
                if (node.sum_children) {
                    for (int child : *node.sum_children) {
                        if (simplified_idx[child] == -1) {
                            st.push(child);
                            children_ready = false;
                        }
                    }
                }
                if (!children_ready) continue;

                // Собираем упрощённых детей, отфильтровывая нулевые константы
                std::vector<int> simplified_children;
                if (node.sum_children) {
                    simplified_children.reserve(node.sum_children->size());
                    for (int child : *node.sum_children) {
                        int simp_child = simplified_idx[child];
                        const Node& cnode = nodes[simp_child];
                        if (cnode.op == LazyOp::CONST && is_zero(values[cnode.value_idx]))
                            continue;
                        simplified_children.push_back(simp_child);
                    }
                }

                int new_idx;
                if (simplified_children.empty()) {
                    new_idx = add_const(Value(SmallStorage(0)));
                }
                else if (simplified_children.size() == 1) {
                    new_idx = simplified_children[0];
                }
                else if (simplified_children.size() == 2) {
                    new_idx = get_binary_node(LazyOp::ADD, simplified_children[0], simplified_children[1]);
                }
                else {
                    // Проверка: все ли дети константы?
                    bool all_const = true;
                    for (int c : simplified_children) {
                        if (nodes[c].op != LazyOp::CONST) {
                            all_const = false;
                            break;
                        }
                    }
                    if (all_const) {
                        Value sum = values[nodes[simplified_children[0]].value_idx];
                        for (size_t i = 1; i < simplified_children.size(); ++i) {
                            sum = eager_add(sum, values[nodes[simplified_children[i]].value_idx]);
                        }
                        new_idx = add_const(sum);
                    }
                    else {
                        new_idx = make_sum_node(std::move(simplified_children));
                    }
                }
                simplified_idx[idx] = new_idx;
                st.pop();
                continue;
            }

            // --------------------------------------------------------------------
            // Унарные операции (child1 == -1)
            // --------------------------------------------------------------------
            if (node.child1 == -1) {
                if (node.child0 == -1) {
                    // без детей (константа уже обработана выше)
                    simplified_idx[idx] = idx;
                    st.pop();
                    continue;
                }

                // Убеждаемся, что ребёнок упрощён
                if (node.child0 != -1 && simplified_idx[node.child0] == -1) {
                    st.push(node.child0);
                    continue;
                }
                int child0_simp = node.child0 != -1 ? simplified_idx[node.child0] : -1;
                bool is_const0 = (child0_simp != -1 && nodes[child0_simp].op == LazyOp::CONST);

                int new_idx = idx;

                switch (node.op) {
                case LazyOp::NEG:
                    if (child0_simp != -1 && nodes[child0_simp].op == LazyOp::NEG) {
                        int grand = nodes[child0_simp].child0;
                        if (grand != -1) new_idx = grand;
                    }
                    break;
                case LazyOp::RECIP:
                    if (child0_simp != -1 && nodes[child0_simp].op == LazyOp::RECIP) {
                        int grand = nodes[child0_simp].child0;
                        if (grand != -1) new_idx = grand;
                    }
                    break;
                case LazyOp::EXP:
                    if (child0_simp != -1 && nodes[child0_simp].op == LazyOp::LOG) {
                        int grand = nodes[child0_simp].child0;
                        if (grand != -1) new_idx = grand;
                    }
                    else if (is_const0 && is_zero(values[nodes[child0_simp].value_idx])) {
                        new_idx = add_const(Value(SmallStorage(1)));
                    }
                    break;
                case LazyOp::LOG:
                    if (child0_simp != -1 && nodes[child0_simp].op == LazyOp::EXP) {
                        int grand = nodes[child0_simp].child0;
                        if (grand != -1) new_idx = grand;
                    }
                    else if (is_const0 && is_one(values[nodes[child0_simp].value_idx])) {
                        new_idx = add_const(Value(SmallStorage(0)));
                    }
                    break;
                case LazyOp::SQRT:
                    if (is_const0) {
                        const Value& arg = values[nodes[child0_simp].value_idx];
                        if (is_one(arg)) {
                            new_idx = add_const(Value(SmallStorage(1)));
                        }
                        else if (is_zero(arg)) {
                            new_idx = add_const(Value(SmallStorage(0)));
                        }
                    }
                    else if (child0_simp != -1 && nodes[child0_simp].op == LazyOp::EXP) {
                        int inner = nodes[child0_simp].child0;
                        if (inner != -1) {
                            Value eps = node.value_idx != -1 ? values[node.value_idx] : internal::default_eps_value;
                            int eps_idx = internal::pool.add_value(eps);
                            int two = add_const(Value(SmallStorage(2)));
                            int half = get_unary_node(LazyOp::RECIP, two);
                            int x_half = get_binary_node(LazyOp::MUL, inner, half);
                            new_idx = get_unary_node(LazyOp::EXP, x_half, eps_idx);
                        }
                    }
                    break;
                case LazyOp::SIN:
                    if (is_const0 && is_zero(values[nodes[child0_simp].value_idx])) {
                        new_idx = add_const(Value(SmallStorage(0)));
                    }
                    break;
                case LazyOp::COS:
                    if (is_const0 && is_zero(values[nodes[child0_simp].value_idx])) {
                        new_idx = add_const(Value(SmallStorage(1)));
                    }
                    break;
                case LazyOp::ACOS:
                    if (is_const0) {
                        const Value& arg = values[nodes[child0_simp].value_idx];
                        if (is_one(arg)) {
                            new_idx = add_const(Value(SmallStorage(0)));
                        }
                        else if (is_zero(arg)) {
                            Value eps = node.value_idx != -1 ? values[node.value_idx] : internal::default_eps_value;
                            int eps_idx = internal::pool.add_value(eps);
                            int pi_node = get_unary_node(LazyOp::PI, -1, eps_idx);
                            int two = add_const(Value(SmallStorage(2)));
                            int half = get_unary_node(LazyOp::RECIP, two);
                            new_idx = get_binary_node(LazyOp::MUL, pi_node, half);
                        }
                    }
                    break;
                default: break;
                }

                if (new_idx != idx) {
                    simplified_idx[idx] = new_idx;
                    st.pop();
                    continue;
                }

                if (is_const0 && is_algebraic(node.op)) {
                    Value arg = values[nodes[child0_simp].value_idx];
                    Value eps = node.value_idx != -1 ? values[node.value_idx] : Value{};
                    Value const_result = compute_node(node.op, arg, Value{}, eps);
                    int const_node = add_const(const_result);
                    simplified_idx[idx] = const_node;
                    st.pop();
                    continue;
                }

                int new_unary = get_unary_node(node.op, child0_simp, node.value_idx);
                simplified_idx[idx] = new_unary;
                st.pop();
                continue;
            }

            // --------------------------------------------------------------------
            // Бинарные операции (child0 != -1 && child1 != -1)
            // --------------------------------------------------------------------
            if (node.child0 != -1 && simplified_idx[node.child0] == -1) {
                st.push(node.child0);
                continue;
            }
            if (node.child1 != -1 && simplified_idx[node.child1] == -1) {
                st.push(node.child1);
                continue;
            }

            int child0_simp = node.child0 != -1 ? simplified_idx[node.child0] : -1;
            int child1_simp = node.child1 != -1 ? simplified_idx[node.child1] : -1;

            bool is_const0 = (child0_simp != -1 && nodes[child0_simp].op == LazyOp::CONST);
            bool is_const1 = (child1_simp != -1 && nodes[child1_simp].op == LazyOp::CONST);

            int new_idx = idx;

            // --------------------------------------------------------------------
            // Специальная обработка для ADD: превращаем глубокие деревья в SUM
            // --------------------------------------------------------------------
            if (node.op == LazyOp::ADD) {
                std::vector<int> operands;
                collect_add_operands(idx, operands);
                if (operands.size() > 2) {
                    int new_sum = make_sum_node(operands);
                    simplified_idx[idx] = new_sum;
                    st.pop();
                    continue;
                }
            }

            // Остальные упрощения для бинарных операций
            if (node.op == LazyOp::ADD) {
                if (is_const0 && is_zero(values[nodes[child0_simp].value_idx]))
                    new_idx = child1_simp;
                else if (is_const1 && is_zero(values[nodes[child1_simp].value_idx]))
                    new_idx = child0_simp;
                else if (child0_simp != -1 && child1_simp != -1) {
                    const Node& left_n = nodes[child0_simp];
                    const Node& right_n = nodes[child1_simp];
                    if (right_n.op == LazyOp::NEG && right_n.child0 == child0_simp)
                        new_idx = add_const(Value(SmallStorage(0)));
                    if (left_n.op == LazyOp::NEG && left_n.child0 == child1_simp)
                        new_idx = add_const(Value(SmallStorage(0)));
                }
            }
            else if (node.op == LazyOp::MUL) {
                if (is_const1 && is_one(values[nodes[child1_simp].value_idx]))
                    new_idx = child0_simp;
                else if (is_const0 && is_one(values[nodes[child0_simp].value_idx]))
                    new_idx = child1_simp;
                else if (is_const1 && is_zero(values[nodes[child1_simp].value_idx]))
                    new_idx = add_const(Value(SmallStorage(0)));
                else if (is_const0 && is_zero(values[nodes[child0_simp].value_idx]))
                    new_idx = add_const(Value(SmallStorage(0)));
                else if (child0_simp != -1 && child1_simp != -1) {
                    const Node& right_n = nodes[child1_simp];
                    if (right_n.op == LazyOp::RECIP && right_n.child0 == child0_simp)
                        new_idx = add_const(Value(SmallStorage(1)));
                    const Node& left_n = nodes[child0_simp];
                    if (left_n.op == LazyOp::RECIP && left_n.child0 == child1_simp)
                        new_idx = add_const(Value(SmallStorage(1)));
                }
            }
            else if (node.op == LazyOp::POW) {
                if (is_const1 && is_one(values[nodes[child1_simp].value_idx]))
                    new_idx = child0_simp;
                else if (is_const1 && is_zero(values[nodes[child1_simp].value_idx]))
                    new_idx = add_const(Value(SmallStorage(1)));
                else if (is_const0 && is_one(values[nodes[child0_simp].value_idx]))
                    new_idx = add_const(Value(SmallStorage(1)));
                else if (is_const0 && is_zero(values[nodes[child0_simp].value_idx]) && is_positive(values[nodes[child1_simp].value_idx]))
                    new_idx = add_const(Value(SmallStorage(0)));
                else if (is_const1) {
                    const Value& exp_val = values[nodes[child1_simp].value_idx];
                    Value one = SmallStorage(1);
                    Value two = SmallStorage(2);
                    Value half = eager_div(one, two);
                    if (exp_val == half) {
                        int sqrt_node = get_unary_node(LazyOp::SQRT, child0_simp, node.value_idx);
                        simplified_idx[idx] = sqrt_node;
                        st.pop();
                        continue;
                    }
                }
                // (a^b)^c -> a^(b*c) для целых показателей
                else if (child0_simp != -1 && nodes[child0_simp].op == LazyOp::POW &&
                    nodes[child0_simp].value_idx == -1) {
                    int a = nodes[child0_simp].child0;
                    int b = nodes[child0_simp].child1;
                    int c = child1_simp;
                    if (b != -1 && c != -1 && nodes[b].op == LazyOp::CONST && nodes[c].op == LazyOp::CONST) {
                        const Value& vb = values[nodes[b].value_idx];
                        const Value& vc = values[nodes[c].value_idx];
                        bool b_int = false, c_int = false;

                        // Проверка целочисленности vb
                        if (vb.tag == ValueType::Small) {
                            SmallStorage sb_norm = vb.storage.small;
                            bool red_b = false;
                            if (!vb.small_reduced) sb_norm.normalize(red_b);
                            if (sb_norm.den == 1) b_int = true;
                        }
                        else if (vb.tag == ValueType::Big) {
                            if (vb.storage.big.denominator() == 1) b_int = true;
                        }

                        // Проверка целочисленности vc
                        if (vc.tag == ValueType::Small) {
                            SmallStorage sc_norm = vc.storage.small;
                            bool red_c = false;
                            if (!vc.small_reduced) sc_norm.normalize(red_c);
                            if (sc_norm.den == 1) c_int = true;
                        }
                        else if (vc.tag == ValueType::Big) {
                            if (vc.storage.big.denominator() == 1) c_int = true;
                        }

                        if (b_int && c_int) {
                            Value vprod = eager_mul(vb, vc);
                            int prod_idx = add_const(vprod);
                            int new_pow = get_pow_node(a, prod_idx, node.value_idx);
                            simplified_idx[idx] = new_pow;
                            st.pop();
                            continue;
                        }
                    }
                }
            }

            if (new_idx != idx) {
                simplified_idx[idx] = new_idx;
                st.pop();
                continue;
            }

            if (is_const0 && is_const1) {
                if (node.op == LazyOp::ADD || node.op == LazyOp::MUL || node.op == LazyOp::POW) {
                    Value left = values[nodes[child0_simp].value_idx];
                    Value right = values[nodes[child1_simp].value_idx];
                    Value eps = node.value_idx != -1 ? values[node.value_idx] : Value{};
                    Value const_result = compute_node(node.op, left, right, eps);
                    int const_node = add_const(const_result);
                    simplified_idx[idx] = const_node;
                    st.pop();
                    continue;
                }
            }

            int new_binary;
            if (node.op == LazyOp::POW) {
                new_binary = get_pow_node(child0_simp, child1_simp, node.value_idx);
            }
            else {
                new_binary = get_binary_node(node.op, child0_simp, child1_simp);
            }
            simplified_idx[idx] = new_binary;
            st.pop();
        }

        return simplified_idx[root_idx];
    }

} // namespace delta::internal