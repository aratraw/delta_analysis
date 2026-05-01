// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// reduce.h
// -----------------------------------------------------------------------------
// PYRAMIDAL COMPACT REDUCTION (PCR) FOR RATIONAL SUMMATION
// -----------------------------------------------------------------------------
//
// This header provides efficient summation of a vector of rational Values
// using a hierarchical batching algorithm. It is used by the evaluation
// engine (evaluate_impl.h) to sum many terms in SUM nodes.
//
// -----------------------------------------------------------------------------
// WHY NOT SIMPLE SEQUENTIAL SUMMATION?
// -----------------------------------------------------------------------------
//
// For rational numbers, the naive loop:
//   Value sum = 0;
//   for (const Value& v : values) sum += v;
//
// results in intermediate fractions whose numerators and denominators can
// grow dramatically. The sum is mathematically correct, but the intermediate
// numbers may become astronomically large (hundreds of thousands of bits)
// even if the final result is moderate. This slows down every addition.
//
// PCR does not change the mathematical result, but it reduces the growth
// of intermediate fractions by summing terms in a balanced binary tree
// rather than a linear chain. The difference in practice is substantial.
//
// -----------------------------------------------------------------------------
// HOW PCR WORKS – STEP BY STEP
// -----------------------------------------------------------------------------
//
// PCR is an in‑place, iterative algorithm that repeatedly reduces the vector
// length by grouping elements into batches of size BATCH_SIZE (default 32).
//
// Algorithm:
//   1. Let current_n = v_work.size().
//   2. If current_n == 0 → replace with [0] and stop.
//   3. If current_n == 1 → nothing to do.
//   4. Compute next_n = ceil(current_n / BATCH_SIZE).
//   5. For i = 0 .. next_n-1:
//        start = i * BATCH_SIZE
//        end   = min(start + BATCH_SIZE, current_n)
//        v_work[i] = reduce_batch(&v_work[start], end - start)
//      (reduce_batch sequentially sums the subrange)
//   6. Set current_n = next_n and repeat from step 2.
//   7. When current_n == 1, the result is in v_work[0].
//
// -----------------------------------------------------------------------------
// EXAMPLE (BATCH_SIZE = 4 for illustration)
// -----------------------------------------------------------------------------
//
// Input vector of 7 numbers: [a0, a1, a2, a3, a4, a5, a6]
//
// Level 1 (batch size 4):
//   batch0: indices 0-3 → sum0 = a0+a1+a2+a3
//   batch1: indices 4-6 → sum1 = a4+a5+a6
//   New vector: [sum0, sum1]   (next_n = ceil(7/4) = 2)
//
// Level 2 (batch size 4):
//   only one batch covering both sum0 and sum1 → total = sum0+sum1
//   New vector: [total]   (next_n = ceil(2/4) = 1)
//
// Done. The final sum is the same as a0+a1+...+a6, but intermediate
// additions were performed as (a0+a1+a2+a3) and (a4+a5+a6) before adding
// the two partials. This balanced tree reduces growth of denominators.
//
// 
// -----------------------------------------------------------------------------
// EXAMPLE – IN‑PLACE VERSION (BATCH_SIZE = 4 for illustration)
// -----------------------------------------------------------------------------
//
// Input working vector: [a0, a1, a2, a3, a4, a5, a6]
//
// Level 1 (batch size 4):
//   i=0: start=0, end=4 → reduce_batch(a0..a3) → sum0 → v_work[0] = sum0
//   i=1: start=4, end=7 → reduce_batch(a4..a6) → sum1 → v_work[1] = sum1
//   Vector becomes: [sum0, sum1, a2, a3, a4, a5, a6]
//   current_n = 7 → next_n = ceil(7/4) = 2
//
// Level 2 (batch size 4):
//   current_n = 2, which is ≤ BATCH_SIZE (4), so only one batch:
//   i=0: start=0, end=2 → reduce_batch(sum0, sum1) → total → v_work[0] = total
//   Vector becomes: [total, sum1, a2, a3, a4, a5, a6]
//   current_n = 2 → next_n = ceil(2/4) = 1
//
// Level 3 (current_n = 1):
//   stop. Result is in index 0 (total).
//
// Resize vector to 1: [total]
//
// Done. The algorithm reuses the same vector storage, writing batch results
// into the first next_n slots at each level, ignoring the rest.
// 
// -----------------------------------------------------------------------------
// WHY BATCH_SIZE = 32?
// -----------------------------------------------------------------------------
//
// The batch size 32 was chosen empirically, with the following rationale:
//
//   - Boost.Multiprecision is configured with MinBits = 128 (see storage.h).
//     This means that small integers (up to 128 bits) are stored directly
//     inside the object on the stack, avoiding heap allocation. Operations on
//     such "small" numbers are significantly faster than on heap‑allocated
//     "large" numbers.
//
//   - When summing a batch of rational numbers, the numerators and denominators
//     grow roughly proportionally to the batch size. With BATCH_SIZE = 32,
//     the typical result of summing 32 random fractions stays within the
//     128‑bit limit for many practical inputs. This keeps the numbers
//     "small" (stack‑allocated) and avoids expensive heap allocations.
//
//   - Larger batches (e.g., 64) often cause the sum to exceed 128 bits,
//     triggering heap allocation and slower big‑integer arithmetic.
//     Smaller batches (e.g., 16) increase the number of reduction levels,
//     which adds overhead from more loop iterations and intermediate vectors.
//
//   - The value 32 also balances the number of reduction levels:
//     * log₂(32) = 5 levels for a binary tree
//     * log₃₂(N) = ceil(log₂(N)/5) levels for PCR (much fewer)
//
//   - This constant is not sacred. If benchmarks on specific workloads show
//     that a different batch size yields better performance, it can be
//     adjusted. It could even be made a compile‑time or run‑time parameter.
//     However, 32 is the preliminary statistically default average.
//
// -----------------------------------------------------------------------------
// IN‑PLACE VS COPY VERSIONS
// -----------------------------------------------------------------------------
//
// - pyramidal_compact_reduce_inplace(std::vector<Value>& v_work):
//     Modifies the input vector in‑place and reduces it to a single element.
//     Minimal memory allocations (only one vector reused for all levels).
//
// - pyramidal_compact_reduce_copy(const std::vector<Value>& values):
//     Makes a copy of the input vector, then reduces the copy in‑place.
//     Used when the original vector must be preserved (e.g., during
//     non‑destructive evaluation).
//
// -----------------------------------------------------------------------------
// COMPLEXITY
// -----------------------------------------------------------------------------
//
// Time:  O(N) additions (same as sequential), but with much smaller
//        intermediate rationals → faster in practice.
// Space: O(N) for the working copy (in‑place version uses the same vector).
//
// -----------------------------------------------------------------------------
// P.S. You know you've written an elegant code when the explanatory comments take more space 
// than the code itself, while the code outruns naive solutions x2-6 times.
// 
// Performance note:
//   - in‑place version: modifies the input vector, no extra copies.
//   - copy version: currently makes a full copy of the input vector first.
//     For large N, this can be improved by writing batch sums directly
//     into a new vector (avoiding copying the entire input).
//     See ToDo marker below.

// ToDo: [FIXME] Current implementation for non-inplace evaluation - copies the entire input vector before reduction,
// which is O(N) copies of Values. For large N, a better approach is to read-only values from initial vector,
// with writing first‑level batch sums directly into a new pre-reserved vector of size ceil(N/BATCH_SIZE),
// then reduce that vector in‑place. This reduces copying to O(N/BATCH_SIZE).
// Priority: medium.

#pragma once

#include "storage.h"   // for Value
#include <vector>
#include <algorithm>
#include <cstddef>

namespace delta::internal {

    // Batch size for pyramidal reduction
    inline constexpr size_t BATCH_SIZE = 32;

    // Sequential summation of a batch (no memory allocation)
    inline Value reduce_batch(const Value* batch, size_t count) {
        if (count == 0) return Value(0);
        Value result = batch[0];
        for (size_t i = 1; i < count; ++i) {
            result += batch[i];
        }
        return result;
    }

    // Pyramidal Compact Reduction (PCR) – in‑place, minimal allocations
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

    // PCR with copy of the input vector (when original data must not be destroyed)
    inline Value pyramidal_compact_reduce_copy(const std::vector<Value>& values) {
        if (values.empty()) return Value(0);
        std::vector<Value> v_work = values;
        pyramidal_compact_reduce_inplace(v_work);
        return std::move(v_work[0]);
    }

} // namespace delta::internal