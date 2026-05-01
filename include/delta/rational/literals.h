// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// literals.h
// -----------------------------------------------------------------------------
// USER-DEFINED LITERALS FOR delta::Rational
// -----------------------------------------------------------------------------
//
// This header provides convenient literal syntax for creating Rational objects:
//
//   auto a = 0_r;           // Rational(0)
//   auto b = 42_r;          // Rational(42)
//   auto c = "0.5"_r;       // Rational(1/2)
//   auto d = "1/3"_r;       // Rational(1/3)
//
// -----------------------------------------------------------------------------
// WHAT WORKS AND WHAT DOES NOT
// -----------------------------------------------------------------------------
//
// ✅ 0_r                         - integer literal, works perfectly.
// ✅ "0.5"_r                     - string literal, parsed exactly as decimal.
// ✅ "1/3"_r                     - string literal, parsed as fraction.
// ❌ 0.5_r                       - DOES NOT WORK (and never will, by design).
//
// -----------------------------------------------------------------------------
// WHY 0.5_r IS NOT SUPPORTED
// -----------------------------------------------------------------------------
//
// The syntax 0.5_r would involve a floating‑point literal of type double.
// Double numbers in IEEE 754 are stored as binary fractions (sums of powers
// of two: 1/2, 1/4, 1/8, ...).
//
// The decimal number 0.1 (1/10) is finite in decimal, but in binary it becomes
// an infinite repeating fraction: 0.00011001100110011... (repeating "0011").
//
// Since double precision has only 53 bits of mantissa, the binary representation
// is truncated. What actually gets stored is not 0.1, but:
//
//   0.1000000000000000055511151231257827021181583404541015625
//
// This is called "binary floating‑point rounding error". Every decimal fraction
// that is not exactly representable in binary (which includes most numbers with
// a finite decimal expansion, like 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, etc.)
// suffers from this.
//
// If we allowed 0.1_r, the literal would first be parsed as a double (acquiring
// this jitter), and only then converted to Rational. The resulting Rational
// would not be the mathematically exact 1/10, but rather:
//
//   1000000000000000055511151231257827021181583404541015625 / 10^58
//   (some huge fraction that approximates 0.1 but is not exact)
//
// This would be a silent source of subtle errors in exact rational computations.
// Users would expect 0.1_r to be 1/10, but would get something very close yet
// not equal – breaking rational arithmetic invariants.
//
// -----------------------------------------------------------------------------
// DESIGN DECISION
// -----------------------------------------------------------------------------
//
// Therefore, we deliberately DO NOT provide a floating‑point literal overload.
// Instead, users must use string literals ("0.1"_r or "1/10"_r). This guarantees that the decimal
// is parsed exactly from its decimal representation, preserving the intended
// rational value.
//
// The sacrifice: slightly less convenient syntax. The gain: exact, predictable,
// and mathematically correct rational numbers.
//
// -----------------------------------------------------------------------------

#pragma once

#include "rational_class.h"
#include <string_view>

namespace delta {

    // Integer literal: 123_r. No loss of precision
    inline Rational operator""_r(unsigned long long num) {
        return Rational(num);
    }

    // String literal: "0.5"_r, "1/3"_r, "12345678901234567890"_r. 
    // No loss of precision due to input data being a string, 
    // passed directly to Boost::multiprecision backend
    inline Rational operator""_r(const char* str, std::size_t len) {
        return Rational(std::string(str, len));
    }

    // NOTE: Floating‑point literal (0.5_r) is intentionally NOT provided.
    // See explanation above.

#ifdef __cpp_user_defined_literals_floating_point
    // This template would be invoked for floating‑point literals (0.5_r).
    // We deliberately leave it undefined (no implementation) to prevent its use.
    // If a user attempts to write 0.5_r, compilation will fail with a link‑time
    // error (undefined reference to operator""_r...), which is better than
    // silently accepting a corrupted value.
    template<char... digits>
    inline Rational operator""_r() = delete;   // explicitly disallowed
#endif

} // namespace delta