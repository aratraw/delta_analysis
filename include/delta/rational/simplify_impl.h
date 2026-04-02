// simplify_impl.h
#pragma once

#include "node_pool.h"
#include "evaluation_core.h"
#include "evaluate_impl.h"
#include "context.h"          // для default_eps()
#include <stack>
#include <vector>
#include <optional>

namespace delta::internal {

    // Вспомогательная функция: является ли операция алгебраической (вычисляемой точно)
    inline bool is_algebraic(LazyOp op) {
        return op == LazyOp::ADD || op == LazyOp::MUL ||
            op == LazyOp::NEG || op == LazyOp::RECIP;
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

            // Ждём детей
            bool children_ready = true;
            if (node.child0 != -1 && simplified_idx[node.child0] == -1) {
                st.push(node.child0);
                children_ready = false;
            }
            if (node.child1 != -1 && simplified_idx[node.child1] == -1) {
                st.push(node.child1);
                children_ready = false;
            }
            if (!children_ready) continue;

            int child0_simp = node.child0 != -1 ? simplified_idx[node.child0] : -1;
            int child1_simp = node.child1 != -1 ? simplified_idx[node.child1] : -1;

            bool is_const0 = (child0_simp != -1 && nodes[child0_simp].op == LazyOp::CONST);
            bool is_const1 = (child1_simp != -1 && nodes[child1_simp].op == LazyOp::CONST);

            int new_idx = idx; // по умолчанию

            // --- Унарные операции (node.child1 == -1) ---
            if (node.child1 == -1) {
                // Узлы без детей (PI, E) уже упрощены
                if (node.child0 == -1) {
                    simplified_idx[idx] = idx;
                    st.pop();
                    continue;
                }

                // 1. Структурные правила
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
                        new_idx = add_const(Value(SmallStorage(1))); // exp(0) → 1
                    }
                    break;
                case LazyOp::LOG:
                    if (child0_simp != -1 && nodes[child0_simp].op == LazyOp::EXP) {
                        int grand = nodes[child0_simp].child0;
                        if (grand != -1) new_idx = grand;
                    }
                    else if (is_const0 && is_one(values[nodes[child0_simp].value_idx])) {
                        new_idx = add_const(Value(SmallStorage(0))); // log(1) → 0
                    }
                    break;
                case LazyOp::SQRT:
                    if (is_const0) {
                        const Value& arg = values[nodes[child0_simp].value_idx];
                        if (is_one(arg)) {
                            new_idx = add_const(Value(SmallStorage(1))); // sqrt(1) → 1
                        }
                        else if (is_zero(arg)) {
                            new_idx = add_const(Value(SmallStorage(0))); // sqrt(0) → 0
                        }
                    }
                    else if (child0_simp != -1 && nodes[child0_simp].op == LazyOp::EXP) {
                        // sqrt(exp(x)) → exp(x/2)
                        int inner = nodes[child0_simp].child0;
                        if (inner != -1) {
                            Value eps = node.value_idx != -1 ? values[node.value_idx]
                                : internal::default_eps_value;
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
                        new_idx = add_const(Value(SmallStorage(0))); // sin(0) → 0
                    }
                    break;
                case LazyOp::COS:
                    if (is_const0 && is_zero(values[nodes[child0_simp].value_idx])) {
                        new_idx = add_const(Value(SmallStorage(1))); // cos(0) → 1
                    }
                    break;
                case LazyOp::ACOS:
                    if (is_const0) {
                        const Value& arg = values[nodes[child0_simp].value_idx];
                        if (is_one(arg)) {
                            new_idx = add_const(Value(SmallStorage(0))); // acos(1) → 0
                        }
                        else if (is_zero(arg)) {
                            // acos(0) → π/2
                            Value eps = node.value_idx != -1 ? values[node.value_idx]
                                : internal::default_eps_value;
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

                // Если структурное правило сработало
                if (new_idx != idx) {
                    simplified_idx[idx] = new_idx;
                    st.pop();
                    continue;
                }

                // 2. Constant folding только для алгебраических операций
                if (is_const0 && is_algebraic(node.op)) {
                    Value arg = values[nodes[child0_simp].value_idx];
                    Value eps = node.value_idx != -1 ? values[node.value_idx] : Value{};
                    Value const_result = compute_node(node.op, arg, Value{}, eps);
                    int const_node = add_const(const_result);
                    simplified_idx[idx] = const_node;
                    st.pop();
                    continue;
                }

                // 3. Ничего не упростилось – создаём новый узел
                int new_unary = get_unary_node(node.op, child0_simp, node.value_idx);
                simplified_idx[idx] = new_unary;
                st.pop();
                continue;
            }

            // --- Бинарные операции (node.child1 != -1) ---

            // 1. Алгебраические правила
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
                // Структурные правила
                if (is_const1 && is_one(values[nodes[child1_simp].value_idx]))
                    new_idx = child0_simp;                          // x^1 → x
                else if (is_const1 && is_zero(values[nodes[child1_simp].value_idx]))
                    new_idx = add_const(Value(SmallStorage(1)));   // x^0 → 1
                else if (is_const0 && is_one(values[nodes[child0_simp].value_idx]))
                    new_idx = add_const(Value(SmallStorage(1)));   // 1^y → 1
                else if (is_const0 && is_zero(values[nodes[child0_simp].value_idx]) && is_positive(values[nodes[child1_simp].value_idx]))
                    new_idx = add_const(Value(SmallStorage(0)));   // 0^positive → 0
                // НОВОЕ ПРАВИЛО: x^(1/2) → sqrt(x)
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
                // Дополнительные упрощения (можно расширять):
                // (a^b)^c → a^(b*c) если a > 0 или показатели целые
                else if (child0_simp != -1 && nodes[child0_simp].op == LazyOp::POW &&
                    nodes[child0_simp].value_idx == -1) {
                    // Внутренний POW не имеет своего eps (т.е. это не POW с eps, а просто степень)
                    int a = nodes[child0_simp].child0;
                    int b = nodes[child0_simp].child1;
                    int c = child1_simp;
                    // Проверяем, что b и c константы (целые)
                    if (b != -1 && c != -1 && nodes[b].op == LazyOp::CONST && nodes[c].op == LazyOp::CONST) {
                        Value vb = values[nodes[b].value_idx];
                        Value vc = values[nodes[c].value_idx];
                        // Пока упрощаем только для целых показателей (чтобы избежать сложностей)
                        bool b_int = false, c_int = false;
                        if (const auto* sb = std::get_if<SmallStorage>(&vb)) {
                            SmallStorage sb_norm = *sb;
                            sb_norm.normalize();
                            if (sb_norm.den == 1) b_int = true;
                        }
                        else if (const auto* bb = std::get_if<BigStorage>(&vb)) {
                            if (bb->den() == 1) b_int = true;
                        }
                        if (const auto* sc = std::get_if<SmallStorage>(&vc)) {
                            SmallStorage sc_norm = *sc;
                            sc_norm.normalize();
                            if (sc_norm.den == 1) c_int = true;
                        }
                        else if (const auto* bc = std::get_if<BigStorage>(&vc)) {
                            if (bc->den() == 1) c_int = true;
                        }
                        if (b_int && c_int) {
                            Value vprod = eager_mul(vb, vc);
                            int prod_idx = add_const(vprod);
                            // Создаём новый узел a^(b*c) с тем же eps, что и у внешнего POW
                            int new_pow = get_pow_node(a, prod_idx, node.value_idx);
                            simplified_idx[idx] = new_pow;
                            st.pop();
                            continue;
                        }
                    }
                }
                // a^(b+c) → a^b * a^c, если b и c константы (можно добавить позже)
            }

            if (new_idx != idx) {
                simplified_idx[idx] = new_idx;
                st.pop();
                continue;
            }

            // 2. Constant folding для бинарных операций
            if (is_const0 && is_const1) {
                // Для ADD и MUL используем compute_node, для POW тоже, но с осторожностью
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

            // 3. Ничего не упростилось – создаём новый узел
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