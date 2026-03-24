#pragma once

#include <absl/numeric/int128.h>
#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>

namespace delta::internal {

    // Convert absl::uint128 to boost::multiprecision::cpp_int
    inline boost::multiprecision::cpp_int to_cpp_int(absl::uint128 val) {
        boost::multiprecision::cpp_int result;
        // Split into two 64-bit halves
        uint64_t low = static_cast<uint64_t>(val);
        uint64_t high = static_cast<uint64_t>(val >> 64);
        result = high;
        result <<= 64;
        result |= low;
        return result;
    }

    // Convert absl::int128 to boost::multiprecision::cpp_int (preserving sign)
    inline boost::multiprecision::cpp_int to_cpp_int(absl::int128 val) {
        bool negative = val < 0;
        absl::uint128 abs_val = negative ? -val : static_cast<absl::uint128>(val);
        boost::multiprecision::cpp_int result = to_cpp_int(abs_val);
        if (negative) result = -result;
        return result;
    }

    // Greatest common divisor for absl::uint128 (Euclidean algorithm)
    inline absl::uint128 gcd(absl::uint128 a, absl::uint128 b) noexcept {
        while (b != 0) {
            absl::uint128 t = b;
            b = a % b;
            a = t;
        }
        return a;
    }

    // Check if a * b would overflow absl::int128
    inline bool would_overflow_mul(absl::int128 a, absl::int128 b) noexcept {
        // Use compiler built-ins when available
#if defined(__GNUC__) || defined(__clang__)
        absl::int128 result;
        return __builtin_mul_overflow(a, b, &result);
#elif defined(_MSC_VER) && defined(_M_IX86) || defined(_M_X64)
    // MSVC provides _Mul128 for 64-bit, but for 128-bit we need to use __mul128?
    // Actually _Mul128 works on __int64, not __int128. Use fallback.
        (void)a; (void)b;
        // Fallback to manual check
        // We'll just use the fallback for MSVC as well for simplicity
        // Since the fallback is safe, we'll use it universally for MSVC.
        // The fallback is below.
#endif

    // Fallback: use division to check
        if (a == 0 || b == 0) return false;
        bool negative = (a < 0) ^ (b < 0);
        absl::uint128 aa = a < 0 ? -a : a;
        absl::uint128 bb = b < 0 ? -b : b;
        absl::uint128 max_val = negative
            ? static_cast<absl::uint128>(absl::int128(1) << 127)   // 2^127
            : static_cast<absl::uint128>(absl::int128(1) << 127) - 1; // 2^127 - 1
        if (aa > max_val / bb) return true;
        return false;
    }

    // Check if a + b would overflow absl::int128
    inline bool would_overflow_add(absl::int128 a, absl::int128 b) noexcept {
        // Use compiler built-ins
#if defined(__GNUC__) || defined(__clang__)
        absl::int128 result;
        return __builtin_add_overflow(a, b, &result);
#elif defined(_MSC_VER)
        (void)a; (void)b;
        // Fallback
#endif

    // Fallback: check sign bits
        if ((b > 0) && (a > (absl::int128(1) << 127) - 1 - b)) return true;
        if ((b < 0) && (a < -((absl::int128(1) << 127)) - b)) return true;
        return false;
    }

    // Least common multiple for absl::uint128 (result may not fit in 128 bits)
    inline boost::multiprecision::cpp_int lcm(absl::uint128 a, absl::uint128 b) {
        if (a == 0 || b == 0) return boost::multiprecision::cpp_int(0);
        boost::multiprecision::cpp_int aa = to_cpp_int(a);
        boost::multiprecision::cpp_int bb = to_cpp_int(b);
        boost::multiprecision::cpp_int g = to_cpp_int(gcd(a, b));
        return (aa / g) * bb;
    }

    // Convert absl::int128 to decimal string
    inline std::string int128_to_string(absl::int128 n) {
        if (n == 0) return "0";
        bool negative = n < 0;
        absl::uint128 u = negative ? -n : static_cast<absl::uint128>(n);
        std::string result;
        while (u > 0) {
            result.push_back(static_cast<char>('0' + static_cast<int>(u % 10)));
            u /= 10;
        }
        if (negative) result.push_back('-');
        std::reverse(result.begin(), result.end());
        return result;
    }

    // Convert absl::uint128 to decimal string
    inline std::string uint128_to_string(absl::uint128 n) {
        if (n == 0) return "0";
        std::string result;
        while (n > 0) {
            result.push_back(static_cast<char>('0' + static_cast<int>(n % 10)));
            n /= 10;
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

    //-----------

    inline absl::int128 int128_from_string(const std::string& s) {
        absl::int128 result = 0;
        bool negative = false;
        size_t i = 0;
        if (!s.empty() && s[0] == '-') {
            negative = true;
            i = 1;
        }
        for (; i < s.size(); ++i) {
            result = result * 10 + (s[i] - '0');
        }
        return negative ? -result : result;
    }

    inline absl::uint128 uint128_from_string(const std::string& s) {
        absl::uint128 result = 0;
        for (char c : s) {
            result = result * 10 + (c - '0');
        }
        return result;
    }
} // namespace delta::internal