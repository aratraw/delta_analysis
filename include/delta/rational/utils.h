// utils.h
#pragma once

#include <absl/numeric/int128.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/number.hpp>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>

// Cross‑platform 64‑bit ctz
#ifdef _MSC_VER
#include <intrin.h>
static inline int ctz64(uint64_t x) {
    unsigned long index;
    if (_BitScanForward64(&index, x)) return static_cast<int>(index);
    return 64;
}
#else
static inline int ctz64(uint64_t x) {
    return __builtin_ctzll(x);
}
#endif

namespace delta::internal {

    // ============================================================================
    // 1. Тип dumb_int с отключёнными expression templates (et_off)
    // ============================================================================
    using dumb_int = boost::multiprecision::number<
        boost::multiprecision::cpp_int_backend<>,
        boost::multiprecision::et_off
    >;

    // ============================================================================
    // 1b. Вспомогательные 128-битные функции (ctz, степень двойки, GCD)
    // ============================================================================
    inline int ctz128(absl::uint128 x) {
        uint64_t lo = static_cast<uint64_t>(x);
        if (lo != 0) return ctz64(lo);
        uint64_t hi = static_cast<uint64_t>(x >> 64);
        return 64 + ctz64(hi);
    }

    inline bool is_power_of_two(absl::uint128 x) {
        return x != 0 && (x & (x - 1)) == 0;
    }

    // Оптимизированный бинарный GCD (алгоритм Стейна с ctz)
    inline absl::uint128 binary_gcd(absl::uint128 a, absl::uint128 b) noexcept {
        if (a == 0) return b;
        if (b == 0) return a;
        if (a == b) return a;
        int shift = ctz128(a | b);
        a >>= shift;
        b >>= shift;
        while (a != b) {
            if (a > b) std::swap(a, b);
            b -= a;
            b >>= ctz128(b);
        }
        return a << shift;
    }

    // ============================================================================
    // 2. Конвертации между absl::int128 / absl::uint128 и dumb_int
    // ============================================================================

    inline dumb_int to_dumb_int(absl::uint128 val) {
        if (val == 0) return dumb_int(0);
        uint64_t low = static_cast<uint64_t>(val);
        uint64_t high = static_cast<uint64_t>(val >> 64);
        dumb_int result(high);
        result <<= 64;
        result |= low;
        return result;
    }

    inline dumb_int to_dumb_int(absl::int128 val) {
        if (val == 0) return dumb_int(0);
        bool neg = val < 0;
        absl::uint128 abs_val = neg ? -val : static_cast<absl::uint128>(val);
        dumb_int result = to_dumb_int(abs_val);
        if (neg) result = -result;
        return result;
    }

    inline bool fits_in_uint128(const dumb_int& x) {
        if (x == 0) return true;
        if (x < 0) return false;
        return boost::multiprecision::msb(x) < 128;
    }

    inline bool fits_in_int128(const dumb_int& x) {
        if (x == 0) return true;
        bool neg = x < 0;
        dumb_int abs_x = neg ? -x : x;
        if (!fits_in_uint128(abs_x)) return false;
        dumb_int limit = dumb_int(1) << 127;
        if (neg) {
            return abs_x <= limit;
        }
        else {
            return abs_x <= limit - 1;
        }
    }

    inline absl::uint128 dumb_int_to_uint128(const dumb_int& x) {
        if (x == 0) return 0;
        uint64_t low = static_cast<uint64_t>(x & ((dumb_int(1) << 64) - 1));
        uint64_t high = static_cast<uint64_t>(x >> 64);
        return (static_cast<absl::uint128>(high) << 64) | low;
    }

    inline absl::int128 dumb_int_to_int128(const dumb_int& x) {
        if (x == 0) return 0;
        bool neg = x < 0;
        dumb_int abs_x = neg ? -x : x;
        if (!fits_in_uint128(abs_x)) {
            throw std::overflow_error("dumb_int_to_int128: value out of range");
        }
        absl::uint128 u = dumb_int_to_uint128(abs_x);
        if (neg) {
            if (u > static_cast<absl::uint128>(std::numeric_limits<absl::int128>::max()) + 1) {
                throw std::overflow_error("dumb_int_to_int128: negative overflow");
            }
            return -static_cast<absl::int128>(u);
        }
        else {
            if (u > static_cast<absl::uint128>(std::numeric_limits<absl::int128>::max())) {
                throw std::overflow_error("dumb_int_to_int128: positive overflow");
            }
            return static_cast<absl::int128>(u);
        }
    }

    // ============================================================================
    // 3. Быстрое удаление малых простых множителей из числителя и знаменателя
    // ============================================================================
    inline void extract_small_primes(dumb_int& num, dumb_int& den) {
        auto shift_num = boost::multiprecision::lsb(num);
        auto shift_den = boost::multiprecision::lsb(den);
        auto shift = std::min(shift_num, shift_den);
        if (shift > 0) {
            num >>= shift;
            den >>= shift;
        }
        constexpr std::array<unsigned, 8> primes = { 3, 5, 7, 11, 13, 17, 19, 23 };
        for (unsigned p : primes) {
            while (num % p == 0 && den % p == 0) {
                num /= p;
                den /= p;
            }
        }
    }

    // ============================================================================
    // 4. Проверки переполнения для 128-битной арифметики
    // ============================================================================
    inline bool would_overflow_mul(absl::int128 a, absl::int128 b) noexcept {
#if defined(__GNUC__) || defined(__clang__)
        absl::int128 result;
        return __builtin_mul_overflow(a, b, &result);
#else
        if (a == 0 || b == 0) return false;
        bool negative = (a < 0) ^ (b < 0);
        absl::uint128 aa = a < 0 ? -a : a;
        absl::uint128 bb = b < 0 ? -b : b;
        absl::uint128 max_val = negative
            ? static_cast<absl::uint128>(std::numeric_limits<absl::int128>::min())
            : static_cast<absl::uint128>(std::numeric_limits<absl::int128>::max());
        return aa > max_val / bb;
#endif
    }

    inline bool would_overflow_mul(absl::uint128 a, absl::uint128 b) noexcept {
#if defined(__GNUC__) || defined(__clang__)
        unsigned __int128 result;
        return __builtin_mul_overflow(a, b, &result);
#else
        if (a == 0 || b == 0) return false;
        absl::uint128 max_val = (std::numeric_limits<absl::uint128>::max)();
        return a > max_val / b;
#endif
    }

    inline bool would_overflow_add(absl::int128 a, absl::int128 b) noexcept {
#if defined(__GNUC__) || defined(__clang__)
        absl::int128 result;
        return __builtin_add_overflow(a, b, &result);
#else
        if (b > 0) {
            return a > std::numeric_limits<absl::int128>::max() - b;
        }
        else if (b < 0) {
            return a < std::numeric_limits<absl::int128>::min() - b;
        }
        return false;
#endif
    }

    // ============================================================================
    // 5. Строковые преобразования (только для отладки)
    // ============================================================================
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

} // namespace delta::internal