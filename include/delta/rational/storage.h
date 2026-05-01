// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// storage.h
// -----------------------------------------------------------------------------
// THE BACKBONE – ARBITRARY‑PRECISION RATIONAL NUMBERS
// -----------------------------------------------------------------------------
// This file defines the internal `Value` type, which is the actual rational
// number representation used throughout the library. It is based on
// Boost.Multiprecision with specific backend parameters tuned for performance.
//
// -----------------------------------------------------------------------------
// ARCHITECTURAL DECISION: BOOST INSTEAD OF CUSTOM SMALL‑BIG STORAGE
// -----------------------------------------------------------------------------
// Earlier versions of the library implemented a custom `SmallStorage` class
// using Abseil’s inlined vectors and a separate heap‑allocated path for large
// integers. The goal was to avoid heap allocations for small numbers (≤ 64 bits)
// and provide stack‑only storage.
//
// However, benchmarking showed that even the most optimised custom
// implementation was **12% slower** than a naive `boost::multiprecision::cpp_int`
// for typical rational arithmetic. Reasons:
//   - Boost's backend uses highly optimised limb operations, often in assembly.
//   - The `cpp_int_backend` already implements a small‑object optimisation
//     when `MinBits` is set (e.g., 128). It stores numbers that fit into that
//     many bits directly inside the object, avoiding heap allocation.
//   - Custom allocators and runtime branching between small/big paths introduced
//     overhead that outweighed the benefits.
//
// Therefore, we abandoned the custom storage and now rely entirely on Boost.
// The `rational_adaptor` + `cpp_int_backend` gives us:
//   - Small‑object optimisation with `MinBits = 128` (numbers up to 128 bits
//     are stack‑allocated, no `malloc`).
//   - Transparent fallback to heap allocation for larger numbers.
//   - Polynomial‑time GCD, multiplication, etc., polished over decades.
//
// The result: the library is not faster than raw Boost in raw eager arithmetic;
// but it is SMARTER in how it uses Boost (lazy evaluation, algebraic
// simplification, batched sums) – achieving **2–6× speedups** over naive
// eager code for even the basic comparative operations, IF THE LIBRARY POTENTIAL IS UTILIZED WISELY.
// 
// -----------------------------------------------------------------------------
// P.S. GMP BACKEND – TECHNICALLY POSSIBLE, BUT NOT ENDORSED
// -----------------------------------------------------------------------------
// The library author has NO RELATIONSHIP with GPL-licensed software and does
// NOT want to be associated with GPL in any way. The author neither recommends
// nor encourages the use of GMP with this library. This is entirely your local
// choice and your own responsibility.
//
// For completeness – if you choose to do so, you can replace the default
// `cpp_int_backend` with `boost::multiprecision::gmp_int`:
//
//   using Value = boost::multiprecision::number<
//       boost::multiprecision::rational_adaptor<
//           boost::multiprecision::gmp_int
//       >,
//       boost::multiprecision::et_off
//   >;
//
// With GMP, many arithmetic operations (multiplication, GCD, division) could
// become significantly faster – often 2–5× depending on integer size.
// The rest of the library (lazy evaluation, simplification, pool, GC, etc.)
// would require **no changes** – it would just call GMP under the hood.
//
// However:
//   - GMP is licensed under the **GNU Lesser General Public License (LGPL)**
//     or **GNU General Public License (GPL)** depending on version.
//   - If you distribute a binary that links against GMP, you may need to
//     comply with the terms of those licences (e.g., provide source code,
//     allow reverse engineering, state modifications, etc.).
//   - The default `cpp_int_backend` uses the **Boost Software License 1.0**
//     (BSL‑1.0), which is permissive and imposes no such obligations.
//
// The delta_analysis library as a whole is distributed under the
// **PolyForm Small Business License 1.0.0** and does NOT require GMP in any
// form to operate. The default backend is fully sufficient for all intended
// use cases.
//
// Therefore: use the default backend. If you decide to experiment with GMP,
// you are on your own – the author provides no support for GMP‑linked builds.
// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------
// IMPORTANT: USE `assign`, NOT CONSTRUCTORS
// -----------------------------------------------------------------------------
// Throughout the library, when constructing a `Value` (or a `Rational` from
// basic types like `double`, `int`, `std::string`), **you MUST use the
// `assign` method**:
//
//   Value v;
//   v.assign(3.14);           // instead of Value v(3.14)
//   v.assign("123/456");      // instead of Value v("123/456")
//   v.assign(42);             // instead of Value v(42)
//
// Why?
//   1. `assign` writes directly into the backend without creating a temporary
//      `cpp_int`. For large numbers, avoiding the temporary copy is measurable.
//   2. `assign` handles edge cases (NaN, Inf, denormals) more gracefully.
//      The constructor `Value(double)` may throw "Cannot convert a non‑finite
//      number", while `assign` either handles it or gives a clear error.
//   3. `assign` is a documented backend method and is more stable across
//      Boost versions. The constructor is a wrapper whose behaviour might
//      change.
//
// TODO: Scan the entire library for places where `Value(x)` or `Rational(x)`
//       is used (especially in `evaluation_core.h` and float‑path functions)
//       and replace them with `assign`.
//
// -----------------------------------------------------------------------------
// THE SACRED COW – DO NOT CHANGE BACKEND PARAMETERS
// -----------------------------------------------------------------------------
// The backend parameters below were found after extensive debugging:
//   - `MinBits = 128` – numbers up to 128 bits fit on stack (no heap).
//   - `MaxBits = 0`   – unlimited precision.
//   - `signed_magnitude` – standard representation.
//   - `unchecked` – no runtime checks (speed).
//   - `Allocator = std::allocator<limb_type>` – **DO NOT REPLACE WITH `void`**.
//
// If you change the allocator to `void`, the code will still compile and pass
// almost all tests, but will produce bizarre Heisenbugs in some corner cases.
// We discovered this the hard way (with divine help). The moral: configure
// Boost once and never touch it again.
//
// The only possible future optimization is to replace the allocator with a
// custom one that allocates large chunks to reduce fragmentation. However,
// this is low priority; the current allocator is already fast enough.
//
// -----------------------------------------------------------------------------
// PERFORMANCE NOTES
// -----------------------------------------------------------------------------
// - `is_zero`, `is_one`, `is_positive`, `is_negative` are O(1) – they inspect
//   the backend's limb array size and sign directly.
// - `numerator` and `denominator` return `dumb_int` (cpp_int with et_off) –
//   these are copies, but cheap for small numbers.
// - `to_double` and `to_string` are provided for debugging and interval
//   arithmetic; they are not meant for high‑frequency use.
// - Hashing is done via `AbslHashValue`, which delegates to Boost's own
//   `hash_value` (compatible with Abseil's framework).
// -----------------------------------------------------------------------------

#pragma once

#include "utils.h"   // for dumb_int
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/rational_adaptor.hpp>
#include <absl/hash/hash.h>
#include <cassert>
#include <string>

namespace delta::internal {

    // ------------------------------------------------------------------------
    // Unified rational number type with arbitrary precision, no expression templates.
    // Backend parameters are fixed – DO NOT CHANGE.
    // ------------------------------------------------------------------------
    using Value = boost::multiprecision::number<
        boost::multiprecision::rational_adaptor<
        boost::multiprecision::cpp_int_backend<
        128,                                          // MinBits – small numbers stay on stack
        0,                                            // MaxBits – unlimited
        boost::multiprecision::signed_magnitude,     // SignType
        boost::multiprecision::unchecked,            // No excessive checks – no overhead
        std::allocator<boost::multiprecision::limb_type>         // Allocator – do not change to void!
        >
        >,
        boost::multiprecision::et_off                 // Expression templates off – predictable. Do not change.
    >;

    // ------------------------------------------------------------------------
    // Fast predicates (direct backend access, O(1) for small numbers)
    // ------------------------------------------------------------------------
    inline bool is_zero(const Value& v) noexcept {
        const auto& n = v.backend().num();
        // In Boost.MP, zero is represented either by an empty limb array (size == 0)
        // or by a single limb with value 0. Check both.
        return n.size() == 0 || (n.size() == 1 && n.limbs()[0] == 0);
    }

    inline bool is_one(const Value& v) noexcept {
        const auto& n = v.backend().num();
        const auto& d = v.backend().denom();
        // rational_adaptor always normalises fractions. One == 1/1.
        // Check: numerator == 1 (positive), denominator == 1.
        return n.size() == 1 && n.limbs()[0] == 1 && !n.sign() &&
            d.size() == 1 && d.limbs()[0] == 1;
    }

    inline bool is_positive(const Value& v) noexcept {
        const auto& n = v.backend().num();
        // sign() == false (non‑negative) AND it is not zero.
        return !n.sign() && !(n.size() == 0 || (n.size() == 1 && n.limbs()[0] == 0));
    }

    inline bool is_negative(const Value& v) noexcept {
        // In signed_magnitude, sign is stored only in the numerator.
        // Zero has sign() == false, so no extra check needed.
        return v.backend().num().sign();
    }

    // ------------------------------------------------------------------------
    // Access to numerator and denominator as dumb_int (cpp_int with et_off)
    // ------------------------------------------------------------------------
    inline dumb_int numerator(const Value& v) {
        return boost::multiprecision::numerator(v);
    }

    inline dumb_int denominator(const Value& v) {
        return boost::multiprecision::denominator(v);
    }

    // ------------------------------------------------------------------------
    // Conversion to double (for interval arithmetic and debugging)
    // ------------------------------------------------------------------------
    inline double to_double(const Value& v) {
        return v.convert_to<double>();
    }

    // ------------------------------------------------------------------------
    // String representation (for debugging only)
    // ------------------------------------------------------------------------
    inline std::string to_string(const Value& v) {
        return v.str();
    }

    // ------------------------------------------------------------------------
    // Hashing for Value (compatible with Abseil)
    // ------------------------------------------------------------------------
    template <typename H>
    H AbslHashValue(H h, const Value& v) {
        return H::combine(std::move(h), boost::multiprecision::hash_value(v));
    }

} // namespace delta::internal