// include/delta/rational/utils.h
#pragma once

#include <absl/numeric/int128.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <type_traits>

namespace delta::internal {

    // -------------------------------------------------------------------------
    // Greatest Common Divisor for unsigned 128‑bit integers
    // -------------------------------------------------------------------------
    inline absl::uint128 gcd(absl::uint128 a, absl::uint128 b) {
        while (b != 0) {
            absl::uint128 t = b;
            b = a % b;
            a = t;
        }
        return a;
    }

    // -------------------------------------------------------------------------
    // Greatest Common Divisor for signed 128‑bit integers (returns positive)
    // -------------------------------------------------------------------------
    inline absl::uint128 gcd(absl::int128 a, absl::int128 b) {
        absl::uint128 ua = (a < 0) ? -static_cast<absl::uint128>(a) : static_cast<absl::uint128>(a);
        absl::uint128 ub = (b < 0) ? -static_cast<absl::uint128>(b) : static_cast<absl::uint128>(b);
        return gcd(ua, ub);
    }

    // -------------------------------------------------------------------------
    // Greatest Common Divisor for arbitrary‑precision integers
    // -------------------------------------------------------------------------
    inline boost::multiprecision::cpp_int gcd(const boost::multiprecision::cpp_int& a,
        const boost::multiprecision::cpp_int& b) {
        return boost::multiprecision::gcd(a, b);
    }

    // -------------------------------------------------------------------------
    // Check if multiplication of two 128‑bit integers would overflow the 128‑bit range
    // (used to decide when to promote to BigStorage).
    // Returns true if the product cannot be represented in absl::int128 / absl::uint128.
    // -------------------------------------------------------------------------
    inline bool would_overflow_mul(absl::int128 a, absl::int128 b) {
        if (a == 0 || b == 0) return false;
        // Use __builtin_mul_overflow if available (GCC/Clang)
#ifdef __SIZEOF_INT128__
        __int128 result;
        return __builtin_mul_overflow(static_cast<__int128>(a), static_cast<__int128>(b), &result);
#else
        // Fallback: check against max/min.
        absl::int128 max_abs = (std::numeric_limits<absl::int128>::max)();
        absl::int128 min_abs = (std::numeric_limits<absl::int128>::min)();
        if (a > 0 && b > 0) return a > max_abs / b;
        if (a > 0 && b < 0) return b < min_abs / a;
        if (a < 0 && b > 0) return a < min_abs / b;
        if (a < 0 && b < 0) return (-a) > max_abs / (-b);
        return false;
#endif
    }

    // -------------------------------------------------------------------------
    // Check if addition of two 128‑bit integers would overflow.
    // -------------------------------------------------------------------------
    inline bool would_overflow_add(absl::int128 a, absl::int128 b) {
        if ((b > 0) && (a > (std::numeric_limits<absl::int128>::max)() - b)) return true;
        if ((b < 0) && (a < (std::numeric_limits<absl::int128>::min)() - b)) return true;
        return false;
    }

} // namespace delta::internal