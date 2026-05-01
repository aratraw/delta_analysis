// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// reduce.h
#pragma once

#include "storage.h"   // для Value
#include <vector>
#include <algorithm>
#include <cstddef>

namespace delta::internal {

    // Размер батча для пирамидальной редукции
    inline constexpr size_t BATCH_SIZE = 32;

    // Последовательное суммирование батча (без выделения памяти)
    inline Value reduce_batch(const Value* batch, size_t count) {
        if (count == 0) return Value(0);
        Value result = batch[0];
        for (size_t i = 1; i < count; ++i) {
            result += batch[i];
        }
        return result;
    }

    // Pyramidal Compact Reduction (PCR) – in-place, минимальные аллокации
    inline void pyramidal_compact_reduce_inplace(std::vector<Value>& v_work) {
        size_t current_n = v_work.size();
        if (current_n == 0) {
            v_work = { Value(0) };
            return;
        }
        if (current_n == 1) return;

        while (current_n > 1) {
            size_t next_n = (current_n + BATCH_SIZE - 1) / BATCH_SIZE;
            for (size_t i = 0; i < next_n; ++i) {
                size_t start = i * BATCH_SIZE;
                size_t end = std::min(start + BATCH_SIZE, current_n);
                v_work[i] = reduce_batch(&v_work[start], end - start);
            }
            current_n = next_n;
        }
        v_work.resize(1);
    }

    // PCR с копированием входного вектора (для случаев, когда нельзя разрушать исходные данные)
    inline Value pyramidal_compact_reduce_copy(const std::vector<Value>& values) {
        if (values.empty()) return Value(0);
        std::vector<Value> v_work = values;
        pyramidal_compact_reduce_inplace(v_work);
        return std::move(v_work[0]);
    }

} // namespace delta::internal