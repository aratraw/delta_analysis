// include/delta/rational/utils.h
#pragma once

#include <absl/numeric/int128.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <limits>
#include <type_traits>

namespace delta::internal {

    inline boost::multiprecision::cpp_int to_cpp_int(absl::uint128 val) {
        uint64_t lo = static_cast<uint64_t>(val);
        uint64_t hi = static_cast<uint64_t>(val >> 64);
        boost::multiprecision::cpp_int result(hi);
        result <<= 64;
        result += lo;
        return result;
    }

    inline boost::multiprecision::cpp_int to_cpp_int(absl::int128 val) {
        if (val < 0) {
            return -to_cpp_int(-static_cast<absl::uint128>(val));
        }
        else {
            return to_cpp_int(static_cast<absl::uint128>(val));
        }
    }

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
    // -------------------------------------------------------------------------
    inline bool would_overflow_mul(absl::int128 a, absl::int128 b) {
        if (a == 0 || b == 0) return false;
#ifdef __SIZEOF_INT128__
        __int128 result;
        return __builtin_mul_overflow(static_cast<__int128>(a), static_cast<__int128>(b), &result);
#else
        absl::int128 max_abs = (std::numeric_limits<absl::int128>::max)();
        absl::int128 min_abs = (std::numeric_limits<absl::int128>::min)();
        if (a > 0 && b > 0) return a > max_abs / b;
        if (a > 0 && b < 0) return b < min_abs / a;
        if (a < 0 && b > 0) return a < min_abs / b;
        if (a < 0 && b < 0) return (-a) > max_abs / (-b);
        return false;
#endif
    }

    inline bool would_overflow_add(absl::int128 a, absl::int128 b) {
        if ((b > 0) && (a > (std::numeric_limits<absl::int128>::max)() - b)) return true;
        if ((b < 0) && (a < (std::numeric_limits<absl::int128>::min)() - b)) return true;
        return false;
    }

    inline boost::multiprecision::cpp_int lcm(absl::uint128 a, absl::uint128 b) {
        absl::uint128 g = gcd(a, b);
        if (g == 0) return boost::multiprecision::cpp_int(0);
        boost::multiprecision::cpp_int ca = to_cpp_int(a);
        boost::multiprecision::cpp_int cb = to_cpp_int(b);
        boost::multiprecision::cpp_int g_cpp = to_cpp_int(g);
        return (ca / g_cpp) * cb;
    }

    inline boost::multiprecision::cpp_int lcm(const boost::multiprecision::cpp_int& a,
        const boost::multiprecision::cpp_int& b) {
        boost::multiprecision::cpp_int g = boost::multiprecision::gcd(a, b);
        if (g == 0) return 0;
        return (a / g) * b;
    }

} // namespace delta::internal