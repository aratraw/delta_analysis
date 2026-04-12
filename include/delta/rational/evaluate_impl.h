// evaluate_impl.h
#pragma once

#include "node_pool.h"
#include "evaluation_core.h"
#include <boost/multiprecision/integer.hpp>   // для lcm, gcd
#include <stack>
#include <optional>
#include <vector>
#include <utility>

namespace delta::internal {

    // ----------------------------------------------------------------------------
    // compute_node – без изменений (использует eager_*, которые уже работают с новым Value)
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
        case LazyOp::POW:   return eager_pow(left, right, eps);
        default:
            throw std::logic_error("compute_node: unknown LazyOp");
        }
    }

    // ----------------------------------------------------------------------------
    // batch_add_values – адаптирован под новый tagged union Value (флаг small_reduced)
    // ----------------------------------------------------------------------------
    inline Value batch_add_values(const std::vector<int>& indices,
        const std::vector<std::optional<Value>>& cache) {
        if (indices.empty()) {
            return Value(SmallStorage(0));
        }
        if (indices.size() == 1) {
            return cache[indices[0]].value();
        }

        // --- Быстрый путь: все операнды SmallStorage и количество ≤ 64 ---
        bool all_small = true;
        for (int idx : indices) {
            const Value& v = cache[idx].value();
            if (v.tag != ValueType::Small) {
                all_small = false;
                break;
            }
        }

        if (all_small && indices.size() <= 64) {
            absl::uint128 common_denom = 1;
            std::vector<absl::int128> nums;
            std::vector<absl::uint128> dens;
            nums.reserve(indices.size());
            dens.reserve(indices.size());

            for (int idx : indices) {
                const SmallStorage& s = cache[idx].value().storage.small;
                nums.push_back(s.num);
                // Используем s.den (без битового флага)
                dens.push_back(s.den);

                absl::uint128 g = binary_gcd(common_denom, s.den);
                absl::uint128 lcm_candidate = common_denom / g;
                if (lcm_candidate > (std::numeric_limits<absl::uint128>::max)() / s.den) {
                    all_small = false;
                    break;
                }
                common_denom = lcm_candidate * s.den;
            }

            if (all_small) {
                absl::int128 sum_num = 0;
                bool overflow = false;
                for (size_t i = 0; i < indices.size(); ++i) {
                    absl::uint128 factor = common_denom / dens[i];
                    if (nums[i] != 0 && factor > 0) {
                        absl::uint128 abs_num = nums[i] < 0 ? static_cast<absl::uint128>(-nums[i]) : static_cast<absl::uint128>(nums[i]);
                        if (abs_num > (std::numeric_limits<absl::uint128>::max)() / factor) {
                            overflow = true;
                            break;
                        }
                    }
                    absl::int128 term = nums[i] * static_cast<absl::int128>(factor);
                    if (would_overflow_add(sum_num, term)) {
                        overflow = true;
                        break;
                    }
                    sum_num += term;
                }
                if (!overflow) {
                    bool negative = (sum_num < 0);
                    absl::uint128 abs_sum = negative ? static_cast<absl::uint128>(-sum_num) : static_cast<absl::uint128>(sum_num);
                    absl::uint128 g = binary_gcd(abs_sum, common_denom);
                    if (g > 1) {
                        abs_sum /= g;
                        common_denom /= g;
                    }
                    if (abs_sum <= static_cast<absl::uint128>((std::numeric_limits<absl::int128>::max)()) &&
                        common_denom <= (std::numeric_limits<absl::uint128>::max)()) {
                        absl::int128 final_num = negative ? -static_cast<absl::int128>(abs_sum) : static_cast<absl::int128>(abs_sum);
                        SmallStorage result_small(final_num, common_denom);
                        // Создаём Value с флагом false и нормализуем
                        Value result_val(result_small, false);
                        result_val.normalize();
                        return result_val;
                    }
                }
            }
        }

        // --- Общий путь с dumb_int – кэшируем нормализованные пары (num, den) ---
        using delta::internal::dumb_int;
        std::vector<std::pair<dumb_int, dumb_int>> norms;
        norms.reserve(indices.size());
        dumb_int common_denom(1);

        for (int idx : indices) {
            auto nd = internal::normalize_to_dumb_int(cache[idx].value());
            norms.emplace_back(nd.first, nd.second);
            common_denom = boost::multiprecision::lcm(common_denom, nd.second);
        }

        dumb_int sum_num(0);
        for (size_t i = 0; i < norms.size(); ++i) {
            dumb_int factor = common_denom / norms[i].second;
            sum_num += norms[i].first * factor;
        }

        dumb_int g = boost::multiprecision::gcd(sum_num, common_denom);
        if (g != 0) {
            sum_num /= g;
            common_denom /= g;
        }

        if (fits_in_int128(sum_num) && fits_in_uint128(common_denom)) {
            SmallStorage result_small(dumb_int_to_int128(sum_num), dumb_int_to_uint128(common_denom));
            Value result_val(result_small, false);
            result_val.normalize();
            return result_val;
        }
        return Value(BigStorage(sum_num, common_denom));
    }

    // ----------------------------------------------------------------------------
    // process_batch_list – без изменений (использует eager_add и batch_add_values)
    // ----------------------------------------------------------------------------
    inline Value process_batch_list(const std::vector<int>& indices,
        size_t batch_size,
        const std::vector<std::optional<Value>>& cache) {
        if (indices.empty()) return Value(SmallStorage(0));
        if (indices.size() <= batch_size) {
            return batch_add_values(indices, cache);
        }
        std::vector<Value> partial_sums;
        partial_sums.reserve((indices.size() + batch_size - 1) / batch_size);
        for (size_t start = 0; start < indices.size(); start += batch_size) {
            size_t end = std::min(start + batch_size, indices.size());
            std::vector<int> chunk(indices.begin() + start, indices.begin() + end);
            partial_sums.push_back(batch_add_values(chunk, cache));
        }
        Value sum = partial_sums[0];
        for (size_t i = 1; i < partial_sums.size(); ++i) {
            sum = eager_add(sum, partial_sums[i]);
        }
        return sum;
    }

    // ----------------------------------------------------------------------------
    // evaluate – итеративный пост-порядок – адаптирован под новый Value
    // ----------------------------------------------------------------------------
    inline Value evaluate(int root_idx) {
        const auto& nodes = pool.nodes;
        const auto& values = pool.values;
        const size_t n = nodes.size();

        std::vector<std::optional<Value>> cache(n);
        std::vector<std::optional<std::vector<int>>> add_cache(n);

        std::stack<int> st;
        st.push(root_idx);

        constexpr size_t DIRECT_ADD_LIMIT = 3;
        constexpr size_t MAX_BATCH_SIZE = 64;
        constexpr size_t MAX_TRANS_BATCH = 8;

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

            // НОВАЯ ВЕТКА ДЛЯ SUM
            if (node.op == LazyOp::SUM) {
                bool all_computed = true;
                if (node.sum_children) {
                    for (int child : *node.sum_children) {
                        if (!cache[child].has_value()) {
                            st.push(child);
                            all_computed = false;
                        }
                    }
                }
                if (!all_computed) continue;

                // Линейная свёртка через eager_add
                Value acc = cache[(*node.sum_children)[0]].value();
                for (size_t i = 1; i < node.sum_children->size(); ++i) {
                    acc = eager_add(acc, cache[(*node.sum_children)[i]].value());
                }
                cache[idx] = std::move(acc);
                st.pop();
                continue;
            }

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

            // --- Специальная обработка для ADD ---
            if (node.op == LazyOp::ADD) {
                std::vector<int> operands;
                operands.reserve(16);

                if (node.child0 != -1) {
                    const Node& left_node = nodes[node.child0];
                    if (left_node.op == LazyOp::ADD && add_cache[node.child0].has_value()) {
                        auto& child_vec = add_cache[node.child0].value();
                        if (pool.refcount[node.child0] == 1) {
                            operands.insert(operands.end(),
                                std::make_move_iterator(child_vec.begin()),
                                std::make_move_iterator(child_vec.end()));
                            child_vec.clear();
                            add_cache[node.child0].reset();
                        }
                        else {
                            operands.insert(operands.end(), child_vec.begin(), child_vec.end());
                        }
                    }
                    else {
                        operands.push_back(node.child0);
                    }
                }

                if (node.child1 != -1) {
                    const Node& right_node = nodes[node.child1];
                    if (right_node.op == LazyOp::ADD && add_cache[node.child1].has_value()) {
                        auto& child_vec = add_cache[node.child1].value();
                        if (pool.refcount[node.child1] == 1) {
                            operands.insert(operands.end(),
                                std::make_move_iterator(child_vec.begin()),
                                std::make_move_iterator(child_vec.end()));
                            child_vec.clear();
                            add_cache[node.child1].reset();
                        }
                        else {
                            operands.insert(operands.end(), child_vec.begin(), child_vec.end());
                        }
                    }
                    else {
                        operands.push_back(node.child1);
                    }
                }

                add_cache[idx] = std::move(operands);
                const auto& indices = add_cache[idx].value();

                if (indices.size() <= DIRECT_ADD_LIMIT) {
                    Value sum = cache[indices[0]].value();
                    for (size_t i = 1; i < indices.size(); ++i) {
                        sum = eager_add(sum, cache[indices[i]].value());
                    }
                    cache[idx] = sum;
                    st.pop();
                    continue;
                }

                std::vector<int> rational_idxs;
                std::vector<int> trans_idxs;
                rational_idxs.reserve(indices.size());
                trans_idxs.reserve(indices.size() / 2);

                for (int operand_idx : indices) {
                    LazyOp op = nodes[operand_idx].op;
                    if (op == LazyOp::CONST || op == LazyOp::MUL || op == LazyOp::NEG || op == LazyOp::RECIP) {
                        rational_idxs.push_back(operand_idx);
                    }
                    else {
                        trans_idxs.push_back(operand_idx);
                    }
                }

                Value final_sum;
                if (trans_idxs.empty()) {
                    final_sum = process_batch_list(rational_idxs, MAX_BATCH_SIZE, cache);
                }
                else {
                    Value trans_sum = process_batch_list(trans_idxs, MAX_TRANS_BATCH, cache);
                    if (rational_idxs.empty()) {
                        final_sum = trans_sum;
                    }
                    else {
                        Value rational_sum = process_batch_list(rational_idxs, MAX_BATCH_SIZE, cache);
                        final_sum = eager_add(trans_sum, rational_sum);
                    }
                }

                cache[idx] = final_sum;
                st.pop();
                continue;
            }

            // --- Все остальные операции ---
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