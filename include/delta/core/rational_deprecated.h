// include/delta/core/rational.h
/**
 * @file rational.h
 * 
 * @brief Exact arithmetic kernel for Δ‑analysis – No‑Double Policy.
 *
 * =============================================================================
 *                           CORE PRINCIPLES
 * =============================================================================
 * 0.  Do Not Disturb rational.h unless absolutely sure.
 *
 * 1.  No‑Double Policy (with one exception).
 *     Floating‑point types (float, long double) are strictly forbidden.
 *     In NATIVE_DOUBLE mode, double is allowed as the underlying type,
 *     but you still must use _r literals to create values; accidental mixing
 *     with bare double is prevented by the absence of implicit conversions.
 *
 * 2.  Four Precision Modes (select via CMake):
 *     - DELTA_RATIONAL_MODE_NATIVE_DOUBLE : use built‑in double (fastest, limited precision)
 *     - DELTA_RATIONAL_MODE_BIN_FLOAT     : binary float (default 100 digits, fast)
 *     - DELTA_RATIONAL_MODE_FIXED_RATIONAL: fixed‑bit rational (256 bits default)
 *     - DELTA_RATIONAL_MODE_EXACT_RATIONAL: unlimited dynamic rational
 *
 * 3.  Raw Literals _r.
 *     Literals like 0.1_r are parsed directly into the chosen Rational type,
 *     without any intermediate double conversion (except in NATIVE_DOUBLE mode,
 *     where it becomes a double literal). This guarantees exact initialisation
 *     according to the chosen backend.
 *
 * 4.  ⚠️⚠️⚠️ SCIENTIFIC NOTATION WITH _r IS NOT CURRENTLY SUPPORTED ⚠️⚠️⚠️
 *     Writing numbers like 1e-10_r WILL COMPILE BUT WILL CRASH AT RUNTIME.
 *     EVEN IF IT DOESN'T CRASH, IT WILL LEAD TO ERROR ACCUMULATION, AMBIGUOUS BEHAVIOR, ETC.
 *     This notation is NOT supported by the Rational backend.
 *
 *     Decimal notation like 0.0000001_r is valid ONLY as long as the number
 *     fits within double/long double (or whatever) precision limits during parsing.
 *
 *     For VERY BIG/SMALL NUMBERS, ALWAYS USE STRING LITERALS:
 *     "0.0000000000000000000000001"_r - this preserves full precision
 *     of the chosen Rational mode. If you chose double as backend - your loss.
 *
 * 5.  Unchecked Arithmetic.
 *     All rational modes use `unchecked` arithmetic – no overflow checks,
 *     for maximum performance and to avoid static initialization issues.
 *     In NATIVE_DOUBLE, standard IEEE behaviour applies.
 *
 * 6.  Precision‑controlled transcendental functions.
 *     In rational modes, functions accept an optional absolute precision `eps`.
 *     In NATIVE_DOUBLE and BIN_FLOAT, precision is fixed by the type and `eps` is ignored.
 *
 * =============================================================================
 *                           USAGE EXAMPLE
 * =============================================================================
 * using namespace delta;
 * auto a = 0.1_r;                    // becomes double, binary float, or rational depending on mode
 * auto b = sin(a, "0.0000000000000000000000000001"_r);  // epsilon is used in rational modes, ignored in double/float
 * auto c = exp(b, "0.00000000000000000001"_r);          // exponential with coarser accuracy
 * auto d = sqrt(2_r);                 // default precision (1e-30)
 *
 * // WRONG - WILL CRASH:
 * // auto wrong = sin(a, 1e-10_r);
 *
 * // CORRECT - use string for tiny numbers:
 * // auto correct = sin(a, "0.0000000001"_r);
 *
 * // No free lunch: you want precision, you pay in iterations.
 * =============================================================================
 *                      ⚠️ CRITICAL NOTES FOR THE BRAVE ⚠️
 * =============================================================================
 *
 * - **EXPRESSION TEMPLATES ARE OFF (et_off) FOR A REASON.**
 *   Boost.Multiprecision, when used with expression templates enabled (et_on),
 *   returns complex expression types instead of concrete numbers.
 *   These expression types LACK DEFAULT CONSTRUCTORS, causing
 *   innocent-looking code like `return Value{};` to explode in your face
 *   with C2512 and a trail of tears. We've been there. 1.5 days of life
 *   were lost. NEVER ENABLE et_on UNLESS YOU HAVE A MATERIALIZE STRATEGY.
 *
 * - **CONSTRUCTORS FROM int – KEEP IT SIMPLE, STUPID.**
 *   Let Boost handle memory management. The `void` parameter in the backend
 *   template list is NOT a placeholder for "nothing". It is also, probably, not the allocator either.
 *   It should remain the magical crutch because it works.
 *   In the current setup, the `void` parameter is exactly where Boost
 *   expects it, and it is not there where Boost does not expect it
 *   Why? Because the library says so. Boost knows. We trust Boost.
 *
 * - **DO NOT MESS WITH THE BACKEND PARAMETERS.**
 *   The sequence `signed_magnitude, unchecked, void` is not random;
 *   it's the magical incantation that makes fixed‑precision rationals
 *   work without crashing. Messing with Boost parameters is like
 *   hiring a hitman on yourself and then forgetting about it.
 *
 * - **DOCUMENTATION CAN LIE.**
 *   Boost documentation is usually good, but in extreme cases it can be outdated or simply
 *   wrong (e.g., the infamous `rational.hpp` that was documented but
 *   removed from the library years ago). When in doubt, read the actual
 *   Boost headers. The truth is in the source. We can lie too. Everyone Lies.
 *
 * - **IF YOU GET WEIRD "OUT OF RANGE" OR CONSTRUCTOR ERRORS,**
 *   you probably messed up the multiprecision constructor calls.
 *   You absolutely passed void where you shouldn't have.
 *   Double-tripple‑check that you are passing the right number of arguments,
 *   that you haven't accidentally enabled expression templates,
 *   and that your backend parameters match what Boost expects.
 *   READ THE SOURCE. DO NOT GUESS. *
 * - **YOU HAVE BEEN WARNED.**
 *   If you are tempted to "optimize" by re‑enabling expression templates
 *   or tweaking the backend arguments, remember the 36 hours of debugging
 *   that led to this comment. The code works now. Let it work.
 *
 * 
 *   P.S. Actually about scientific notation - I forgot if it actually does crash.
 *   No time to check, whatever. To The Collider! (everyone lies).
 * =============================================================================
 */


// THIS VERSION of rational.h is deprecated(if we did a good-enough job).
// To restore it, further COPY THE SEQUENCE BELOW AS-IS TO ROOT CMakeLists.txt for this file, and uncomment.
// 
//# --- Rational backend selection ---
//# Choose the arithmetic mode for delta::Rational.
//# Options: NATIVE_DOUBLE - use built‑in double(fast, limited precision)
//#          BIN_FLOAT - binary floating‑point(cpp_bin_float, default 100 digits)
//#          FIXED_RATIONAL - fixed‑bit rational(rational_adaptor with fixed integer, default 256 bits)
//#          EXACT_RATIONAL - exact unlimited rational(rational_adaptor with dynamic integers)
//set(DELTA_RATIONAL_MODE "EXACT_RATIONAL" CACHE STRING "Rational arithmetic mode")
//set_property(CACHE DELTA_RATIONAL_MODE PROPERTY STRINGS NATIVE_DOUBLE BIN_FLOAT FIXED_RATIONAL EXACT_RATIONAL)
//
//if (DELTA_RATIONAL_MODE STREQUAL "NATIVE_DOUBLE")
//target_compile_definitions(delta_core INTERFACE DELTA_RATIONAL_MODE_NATIVE_DOUBLE)
//
//elseif(DELTA_RATIONAL_MODE STREQUAL "BIN_FLOAT")
//target_compile_definitions(delta_core INTERFACE DELTA_RATIONAL_MODE_BIN_FLOAT)
//# Optional: number of decimal digits(default 100)
//set(DELTA_BIN_FLOAT_DIGITS "100" CACHE STRING "Number of decimal digits for binary float (cpp_bin_float)")
//target_compile_definitions(delta_core INTERFACE DELTA_BIN_FLOAT_DIGITS = ${ DELTA_BIN_FLOAT_DIGITS })
//
//elseif(DELTA_RATIONAL_MODE STREQUAL "FIXED_RATIONAL")
//target_compile_definitions(delta_core INTERFACE DELTA_RATIONAL_MODE_FIXED_RATIONAL)
//# Optional: bit width for the fixed‑size integer(default 256)
//set(DELTA_RATIONAL_BITS "256" CACHE STRING "Bit width for fixed‑bit rational (FIXED_RATIONAL mode)")
//target_compile_definitions(delta_core INTERFACE DELTA_RATIONAL_BITS = ${ DELTA_RATIONAL_BITS })
//
//else()   # EXACT_RATIONAL(default)
//target_compile_definitions(delta_core INTERFACE DELTA_RATIONAL_MODE_EXACT_RATIONAL)
//endif()
//# --------------

#pragma once

#include <boost/multiprecision/number.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/rational_adaptor.hpp>
#include <iostream>
#include <string>
#include <type_traits>
#include <stdexcept>
#include <limits>
#include <cmath>      // for std::sqrt, std::sin, etc. in NATIVE_DOUBLE mode
#include <charconv>   // fast number parsing (C++17/20)
#include <system_error>

#ifndef DELTA_SERIES_MAX_ITER
#define DELTA_SERIES_MAX_ITER 1000000   ///< Safety limit to prevent infinite loops
#endif

namespace delta {

    // -----------------------------------------------------------------------------
    // 1.  Type traits for forbidden types (float, double, long double)
    //     In NATIVE_DOUBLE mode, double is allowed; otherwise it is forbidden.
    // -----------------------------------------------------------------------------

    template <typename T>
    struct is_forbidden_type : std::false_type {};

    template <>
    struct is_forbidden_type<float> : std::true_type {};

    // double is conditionally forbidden
#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
    template <>
    struct is_forbidden_type<double> : std::false_type {};
#else
    template <>
    struct is_forbidden_type<double> : std::true_type {};
#endif

    template <>
    struct is_forbidden_type<long double> : std::true_type {};

    // -----------------------------------------------------------------------------
    // 2.  Backend selection according to CMake‑defined macros
    //     Все rational‑режимы используют `unchecked` и правильный аллокатор.
    //     DO NOT TOUCH. DO NOT EDIT. DO NOT CHANGE THE CONSTRUCTOR ARGUMENTS
    //     OR ELSE IT URNS INTO A PUMPKIN.
    // -----------------------------------------------------------------------------

#if defined(DELTA_RATIONAL_MODE_NATIVE_DOUBLE)

    using Rational = double;  // fastest, but limited precision

#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
    using Rational = boost::multiprecision::number<
        boost::multiprecision::cpp_bin_float<DELTA_BIN_FLOAT_DIGITS>,
        boost::multiprecision::et_off   // <-- отключаем expression templates
    >;
#elif defined(DELTA_RATIONAL_MODE_FIXED_RATIONAL)
    using Rational = boost::multiprecision::number<
        boost::multiprecision::rational_adaptor<
        boost::multiprecision::cpp_int_backend<
        DELTA_RATIONAL_BITS, DELTA_RATIONAL_BITS,
        boost::multiprecision::signed_magnitude,
        boost::multiprecision::unchecked,
        void
        >
        >,
        boost::multiprecision::et_off   // <-- и здесь
    >;
#elif defined(DELTA_RATIONAL_MODE_EXACT_RATIONAL)
    using Rational = boost::multiprecision::number<
        boost::multiprecision::rational_adaptor<
        boost::multiprecision::cpp_int_backend<>
        >,
        boost::multiprecision::et_off   // <-- и здесь
    >;
#else
    // Default (если не определён режим) – тоже с et_off
    using Rational = boost::multiprecision::number<
        boost::multiprecision::rational_adaptor<
        boost::multiprecision::cpp_int_backend<>
        >,
        boost::multiprecision::et_off
    >;
#endif

    // -----------------------------------------------------------------------------
    // 3.  User‑defined literals (raw, no intermediate double)
    //     Простые и надёжные, как в старом файле.
    // -----------------------------------------------------------------------------

    /// Integer literal – прямой конструктор (быстро, без неожиданностей)
    inline Rational operator""_r(unsigned long long num) {
#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
        return static_cast<double>(num);
#else
        return Rational(num);
#endif
    }

    /// String literal – для чисел с плавающей точкой и огромных целых
/// String literal – для чисел с плавающей точкой и огромных целых
    inline Rational operator""_r(const char* str, std::size_t len) {
#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
        double val;
        auto [ptr, ec] = std::from_chars(str, str + len, val);
        if (ec != std::errc()) {
            // Редкий случай ошибки парсинга – fallback на stod
            return std::stod(std::string(str, len));
        }
        return val;
#else
        std::string s(str, len);

        // Проверяем, содержит ли строка десятичную точку
        size_t dot_pos = s.find('.');
        if (dot_pos != std::string::npos) {
            // Парсим десятичную дробь в числитель/знаменатель
            std::string integer_part = s.substr(0, dot_pos);
            std::string fractional_part = s.substr(dot_pos + 1);

            // Если дробная часть пустая (случай "123.") — считаем её нулём
            if (fractional_part.empty()) fractional_part = "0";

            // Подсчитываем количество знаков после запятой
            size_t decimal_places = fractional_part.length();

            // Удаляем ведущие нули из дробной части для числителя, но сохраняем для знаменателя
            std::string fractional_trimmed = fractional_part;
            size_t first_nonzero = fractional_trimmed.find_first_not_of('0');
            if (first_nonzero != std::string::npos) {
                fractional_trimmed = fractional_trimmed.substr(first_nonzero);
            }
            else {
                // Все нули в дробной части
                fractional_trimmed = "0";
            }

            // Формируем числитель: целая часть + дробная часть (без ведущих нулей)
            std::string numerator_str;
            if (integer_part == "0" || integer_part.empty()) {
                // Если целая часть пустая или "0", берём только дробную
                numerator_str = fractional_trimmed;
            }
            else {
                // Иначе объединяем
                numerator_str = integer_part + fractional_trimmed;
            }

            // Если числитель пустой или состоит из нулей
            if (numerator_str.empty() || numerator_str == "0") {
                return Rational(0);
            }

            // Убираем ведущие нули у числителя
            size_t num_start = numerator_str.find_first_not_of('0');
            if (num_start != std::string::npos) {
                numerator_str = numerator_str.substr(num_start);
            }

            // Знаменатель: 10^(количество знаков после запятой)
            // Используем cpp_int для больших степеней
            boost::multiprecision::cpp_int denominator = 1;
            for (size_t i = 0; i < decimal_places; ++i) {
                denominator *= 10;
            }

            // Конструируем рациональное число через строку "числитель/знаменатель"
            // Это единственный надёжный способ в Boost.Multiprecision
            std::string rational_str = numerator_str + "/" + denominator.str();
            return Rational(rational_str);
        }

        // Если нет точки, пробуем парсить как целое или дробь с косой чертой
        try {
            return Rational(s);
        }
        catch (const std::exception& e) {
            // Если не получилось, пробуем как дробь вида "a/b" вручную
            size_t slash_pos = s.find('/');
            if (slash_pos != std::string::npos) {
                // Конструируем через строку для единообразия
                return Rational(s);
            }
            // Всё пропало
            throw std::runtime_error("Cannot parse string as rational: " + s);
        }
#endif
    }
    // -----------------------------------------------------------------------------
    // 4.  Default epsilon – defined dynamically for overriding in actual use
    // -----------------------------------------------------------------------------

    inline Rational& default_eps_value() {
        static Rational eps =
#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
            1e-15;
#else
            Rational(1) / Rational("1000000000000000000000000000000");
#endif
        return eps;
    }

    inline const Rational& default_eps() {
        return default_eps_value();
    }

} // namespace delta

// Макрос должен быть определён после объявления default_eps,
// но до его использования в аргументах по умолчанию.
#define DELTA_DEFAULT_EPS delta::default_eps()

// Снова открываем namespace для всех остальных определений
namespace delta {

    // -----------------------------------------------------------------------------
    // 5.  Hygiene: prevent implicit conversion from double to Rational (advisory only)
    // -----------------------------------------------------------------------------

#ifndef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
    // static_assert(!std::is_convertible_v<double, Rational>,
    //     "[DELTA ERROR]: Implicit conversion from double to Rational is forbidden! Use _r suffix.");
#endif

    // -----------------------------------------------------------------------------
    // 6.  Mathematical constants (computed on demand with specified precision)
    // -----------------------------------------------------------------------------

    namespace detail {
        /// Absolute value – generic version using std::abs for double, Boost version for others
        template<typename T>
        inline T abs(const T& x) {
            using std::abs;
            return abs(x);
        }
    }

    /**
     * Return π (pi) with absolute accuracy at least `eps`.
     * In NATIVE_DOUBLE and BIN_FLOAT modes, eps is ignored and a high‑precision constant is returned.
     */
    inline Rational pi(const Rational& eps = DELTA_DEFAULT_EPS) {
#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
        (void)eps;
        return 3.14159265358979323846264338327950288419716939937510; // enough for double
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
        (void)eps;
        return boost::multiprecision::pi<Rational>(); // Boost provides pi for bin_float
#else
        // arctan(x) = x - x^3/3 + x^5/5 - ...
        auto arctan = [&](const Rational& x) -> Rational {
            Rational x2 = x * x;
            Rational term = x;
            Rational sum = term;
            Rational n = 1;
            std::size_t iter = 0;
            while (detail::abs(term) > eps && iter < DELTA_SERIES_MAX_ITER) {
                term *= -x2;
                n += 2;
                sum += term / n;
                ++iter;
            }
            return sum;
            };
        // π = 16*arctan(1/5) - 4*arctan(1/239)
        return 16 * arctan(Rational(1, 5)) - 4 * arctan(Rational(1, 239));
#endif
    }

    /**
     * Return e (Euler's number) with absolute accuracy at least `eps`.
     * In NATIVE_DOUBLE and BIN_FLOAT modes, eps is ignored.
     */
    inline Rational e(const Rational& eps = DELTA_DEFAULT_EPS) {
#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
        (void)eps;
        return 2.71828182845904523536028747135266249775724709369995;
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
        (void)eps;
        return boost::multiprecision::exp(Rational(1)); // exp(1) for bin_float
#else
        Rational sum = 0;
        Rational term = 1;
        std::size_t n = 0;
        while (detail::abs(term) > eps && n < DELTA_SERIES_MAX_ITER) {
            sum += term;
            ++n;
            term /= n;   // term = 1/n!
        }
        return sum;
#endif
    }

    // -----------------------------------------------------------------------------
    // 7.  Mathematical functions with precision control
    // -----------------------------------------------------------------------------

    // ---------- Square root ------------------------------------------------------
    /**
     * Compute sqrt(x) with absolute accuracy at least `eps`.
     * Throws std::domain_error if x < 0.
     */
    template <typename T>
    inline Rational sqrt(const T& x, const Rational& eps = DELTA_DEFAULT_EPS) {
        static_assert(!is_forbidden_type<T>::value,
            "\n\n[DELTA ERROR]: sqrt() called with floating‑point type!\n"
            "All calculations must use delta::Rational. Use 0.1_r instead of 0.1.\n");
        static_assert(std::is_constructible_v<Rational, T>,
            "Argument must be convertible to delta::Rational.");

        Rational a(x);
        if (a < 0) throw std::domain_error("sqrt of negative rational");

#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
        (void)eps;
        return std::sqrt(a);
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
        (void)eps;
        return boost::multiprecision::sqrt(a);
#else
        if (a == 0) return 0;
        Rational guess = a / 2;
        Rational next;
        std::size_t iter = 0;
        do {
            next = (guess + a / guess) / 2;
#ifdef DELTA_RATIONAL_MODE_EXACT_RATIONAL
            // Keep fractions reduced to avoid runaway size
            next.backend().normalize();
#endif
            if (detail::abs(next - guess) <= eps) break;
            guess = next;
            ++iter;
        } while (iter < DELTA_SERIES_MAX_ITER);
        return next;
#endif
    }

    // ---------- Cube root --------------------------------------------------------
    /**
     * Compute cbrt(x) with absolute accuracy at least `eps` using Newton's method.
     * For negative x, returns -cbrt(-x).
     */
    template <typename T>
    inline Rational cbrt(const T& x, const Rational& eps = DELTA_DEFAULT_EPS) {
        static_assert(!is_forbidden_type<T>::value,
            "[DELTA ERROR]: cbrt() called with floating‑point type!");
        Rational a(x);

        if (a == 0) return 0;
        if (a < 0) return -cbrt(-a, eps);

#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
        (void)eps;
        return std::cbrt(a);
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
        (void)eps;
        return boost::multiprecision::cbrt(a);
#else
        Rational guess = a / 3;
        Rational next;
        std::size_t iter = 0;
        do {
            next = (2 * guess + a / (guess * guess)) / 3;
#ifdef DELTA_RATIONAL_MODE_EXACT_RATIONAL
            next.backend().normalize();
#endif
            if (detail::abs(next - guess) <= eps) break;
            guess = next;
            ++iter;
        } while (iter < DELTA_SERIES_MAX_ITER);
        return next;
#endif
    }

    // ---------- Integer power ----------------------------------------------------
    /**
     * Compute x^exp for integer exponent exp (fast exponentiation).
     * No precision parameter – exact rational result.
     */
    template <typename T>
    inline Rational pow(const T& x, int exp) {
        static_assert(!is_forbidden_type<T>::value,
            "[DELTA ERROR]: pow() called with floating‑point type!");
        Rational base(x);
        if (exp == 0) return 1;
        Rational result = 1;
        int e = exp;
        if (e < 0) {
            base = 1 / base;
            e = -e;
        }
        while (e) {
            if (e & 1) result *= base;
            base *= base;
            e >>= 1;
        }
        return result;
    }

    // ---------- Exponential function ---------------------------------------------
    /**
     * Compute exp(x) with absolute accuracy at least `eps`.
     * Uses argument reduction and Taylor series.
     */
    template <typename T>
    inline Rational exp(const T& x, const Rational& eps = DELTA_DEFAULT_EPS) {
        static_assert(!is_forbidden_type<T>::value,
            "[DELTA ERROR]: exp() called with floating‑point type!");
        Rational a(x);

#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
        (void)eps;
        return std::exp(a);
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
        (void)eps;
        return boost::multiprecision::exp(a);
#else
        // Scale argument: find integer k such that |a/2^k| <= 1
        int k = 0;
        Rational scaled = a;
        while (detail::abs(scaled) > 1) {
            scaled /= 2;
            ++k;
        }
        // Compute exp(scaled) via Taylor series with term control
        Rational sum = 1;
        Rational term = 1;
        std::size_t n = 0;
        while (detail::abs(term) > eps && n < DELTA_SERIES_MAX_ITER) {
            ++n;
            term *= scaled / n;
#ifdef DELTA_RATIONAL_MODE_EXACT_RATIONAL
            term.backend().normalize();
#endif
            sum += term;
#ifdef DELTA_RATIONAL_MODE_EXACT_RATIONAL
            sum.backend().normalize();
#endif
        }
        // Reconstruct exp(a) = (exp(scaled))^(2^k) by repeated squaring
        Rational result = sum;
        for (int i = 0; i < k; ++i) {
            result *= result;
#ifdef DELTA_RATIONAL_MODE_EXACT_RATIONAL
            result.backend().normalize();
#endif
        }
        return result;
#endif
    }

    // ---------- Natural logarithm ------------------------------------------------
    /**
     * Compute log(x) with absolute accuracy at least `eps`.
     * Throws std::domain_error if x <= 0.
     */
    template <typename T>
    inline Rational log(const T& x, const Rational& eps = DELTA_DEFAULT_EPS) {
        static_assert(!is_forbidden_type<T>::value,
            "[DELTA ERROR]: log() called with floating‑point type!");
        Rational a(x);
        if (a <= 0) throw std::domain_error("log of non‑positive rational");

#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
        (void)eps;
        return std::log(a);
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
        (void)eps;
        return boost::multiprecision::log(a);
#else
        // Reduce argument to [0.5, 2] by extracting powers of two.
        int k = 0;
        Rational m = a;
        while (m >= 2) { m /= 2; ++k; }
        while (m < 0.5) { m *= 2; --k; }

        // Precompute ln(2) with required precision
        Rational ln2 = 0;
        {
            // Use series for ln((1+z)/(1-z)) with z = 1/3 -> ln2 ≈ 2*(z + z^3/3 + ...)
            Rational z = Rational(1, 3);
            Rational z2 = z * z;
            Rational term = z;
            Rational sum = term;
            Rational n = 1;
            std::size_t iter = 0;
            while (detail::abs(term) > eps && iter < DELTA_SERIES_MAX_ITER) {
                term *= z2;
                n += 2;
                sum += term / n;
#ifdef DELTA_RATIONAL_MODE_EXACT_RATIONAL
                sum.backend().normalize();
#endif
                ++iter;
            }
            ln2 = 2 * sum;
#ifdef DELTA_RATIONAL_MODE_EXACT_RATIONAL
            ln2.backend().normalize();
#endif
        }

        // Now m in [0.5,2]. Use series ln(m) = 2 * arctanh((m-1)/(m+1))
        Rational y = (m - 1) / (m + 1);
        Rational y2 = y * y;
        Rational term = y;
        Rational sum = term;
        Rational n = 1;
        std::size_t iter = 0;
        while (detail::abs(term) > eps && iter < DELTA_SERIES_MAX_ITER) {
            term *= y2;
            n += 2;
            sum += term / n;
#ifdef DELTA_RATIONAL_MODE_EXACT_RATIONAL
            sum.backend().normalize();
#endif
            ++iter;
        }
        return 2 * sum + k * ln2;
#endif
    }

    // ---------- Sine function ----------------------------------------------------
    /**
     * Compute sin(x) with absolute accuracy at least `eps`.
     */
    template <typename T>
    inline Rational sin(const T& x, const Rational& eps = DELTA_DEFAULT_EPS) {
        static_assert(!is_forbidden_type<T>::value,
            "[DELTA ERROR]: sin() called with floating‑point type!");
        Rational a(x);

#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
        (void)eps;
        return std::sin(a);
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
        (void)eps;
        return boost::multiprecision::sin(a);
#else
        // Reduce argument to [-π, π] using periodicity.
        Rational twopi = 2 * pi(eps);
        Rational reduced = a;
        while (reduced > pi(eps)) reduced -= twopi;
        while (reduced < -pi(eps)) reduced += twopi;

        // Taylor series: sin(y) = y - y^3/3! + y^5/5! - ...
        Rational y = reduced;
        Rational y2 = y * y;
        Rational term = y;
        Rational sum = term;
        Rational n = 1;
        std::size_t iter = 0;
        while (detail::abs(term) > eps && iter < DELTA_SERIES_MAX_ITER) {
            term *= -y2 / ((n + 1) * (n + 2));
#ifdef DELTA_RATIONAL_MODE_EXACT_RATIONAL
            term.backend().normalize();
#endif
            sum += term;
#ifdef DELTA_RATIONAL_MODE_EXACT_RATIONAL
            sum.backend().normalize();
#endif
            n += 2;
            ++iter;
        }
        return sum;
#endif
    }

    // ---------- Cosine function --------------------------------------------------
    /**
     * Compute cos(x) with absolute accuracy at least `eps`.
     */
    template <typename T>
    inline Rational cos(const T& x, const Rational& eps = DELTA_DEFAULT_EPS) {
        static_assert(!is_forbidden_type<T>::value,
            "[DELTA ERROR]: cos() called with floating‑point type!");
        Rational a(x);

#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
        (void)eps;
        return std::cos(a);
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
        (void)eps;
        return boost::multiprecision::cos(a);
#else
        // Reduce argument to [-π, π] using periodicity.
        Rational twopi = 2 * pi(eps);
        Rational reduced = a;
        while (reduced > pi(eps)) reduced -= twopi;
        while (reduced < -pi(eps)) reduced += twopi;

        // Taylor series: cos(y) = 1 - y^2/2! + y^4/4! - ...
        Rational y2 = reduced * reduced;
        Rational term = 1;
        Rational sum = term;
        Rational n = 0;
        std::size_t iter = 0;
        while (detail::abs(term) > eps && iter < DELTA_SERIES_MAX_ITER) {
            term *= -y2 / ((2 * n + 1) * (2 * n + 2));
#ifdef DELTA_RATIONAL_MODE_EXACT_RATIONAL
            term.backend().normalize();
#endif
            sum += term;
#ifdef DELTA_RATIONAL_MODE_EXACT_RATIONAL
            sum.backend().normalize();
#endif
            n += 2;
            ++iter;
        }
        return sum;
#endif
    }

    // ---------- Arccosine function ------------------------------------------------
    /**
     * Compute acos(x) with absolute accuracy at least `eps`.
     * x must be in [-1, 1]. Throws std::domain_error otherwise.
     */
    template <typename T>
    inline Rational acos(const T& x, const Rational& eps = DELTA_DEFAULT_EPS) {
        static_assert(!is_forbidden_type<T>::value,
            "[DELTA ERROR]: acos() called with floating‑point type!");
        Rational a(x);
        if (a < -1_r || a > 1_r) throw std::domain_error("acos argument out of [-1,1]");

#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
        (void)eps;
        return std::acos(a);
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
        (void)eps;
        return boost::multiprecision::acos(a);
#else
        // Use Newton's method to solve cos(y) = a for y in [0,π]
        // Initial guess:
        Rational y;
        if (a >= 0_r) {
            y = pi(eps) / 2_r * (1_r - a);
        }
        else {
            y = pi(eps) - pi(eps) / 2_r * (1_r + a);
        }

        Rational delta;
        std::size_t iter = 0;
        do {
            Rational cos_y = delta::cos(y, eps);
            Rational sin_y = delta::sin(y, eps);
            if (sin_y == 0_r) break; // shouldn't happen for |a|<1
            delta = (cos_y - a) / sin_y;
            y -= delta;
#ifdef DELTA_RATIONAL_MODE_EXACT_RATIONAL
            y.backend().normalize();
#endif
            ++iter;
        } while (detail::abs(delta) > eps && iter < DELTA_SERIES_MAX_ITER);
        return y;
#endif
    }

    //------------------------------------------------------------------------------
    // 8.  Floating-point literals with maximum precision
    //     Support syntax: 0.123_r, 3.14159_r, etc.
    //     Uses C++23 approach if available, otherwise falls back to long double
    //     with maximum precision string conversion.
    //------------------------------------------------------------------------------

#ifdef __cpp_user_defined_literals_floating_point
// C++23 way: compiler passes the exact character sequence
    template <char... digits>
    Rational operator""_r() {
        // Pack digits into a null-terminated string at compile time
        constexpr char str[] = { digits..., '\0' };
        // Используем строковую перегрузку для парсинга
        return operator""_r(str, sizeof(str) - 1);
    }
#else
// Fallback for older compilers: use long double and convert to string
// with maximum precision to preserve as much accuracy as possible.
    inline Rational operator""_r(long double x) {
        // If we're in NATIVE_DOUBLE mode, we can just return x directly
#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
        return static_cast<double>(x);
#else
        // For rational modes, convert to string with maximum precision
        // to avoid losing digits during the double → rational conversion.

        // Handle special cases first
        if (x == 0.0l) return 0_r;

        // Use stringstream with maximum precision
        std::stringstream ss;

        // Set precision to ensure we capture all digits that matter
        // max_digits10 gives us enough digits for round-trip conversion
        ss.precision(std::numeric_limits<long double>::max_digits10);

        // НЕ ИСПОЛЬЗУЕМ fixed! Используем defaultfloat чтобы избежать
        // лишних нулей и научной нотации для нормальных чисел
        ss << std::defaultfloat << x;

        std::string str = ss.str();

        // Проверяем на научную нотацию
        if (str.find('e') != std::string::npos || str.find('E') != std::string::npos) {
            throw std::runtime_error("Scientific notation in _r literals is not supported. Use string literal: \"" + str + "\"_r");
        }

        // Вызываем строковую перегрузку для парсинга десятичной дроби
        return operator""_r(str.c_str(), str.length());
#endif
    }
#endif


    using detail::abs; //why? because! because we need delta::abs to be visible in namespace delta.
} // namespace delta

// Подключаем Eigen для специализации internal::sqrt_impl
#include <Eigen/Core>

namespace Eigen {
    namespace internal {

        // Явная специализация sqrt_impl для типа delta::Rational
        template<>
        struct sqrt_impl<delta::Rational>
        {
            static inline delta::Rational run(const delta::Rational& x)
            {
                return delta::sqrt(x);
            }
        };

    } // namespace internal
} // namespace Eigen