// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// literals.h
// -----------------------------------------------------------------------------
// USER-DEFINED LITERALS FOR delta::Rational AND delta::GaussQi
// -----------------------------------------------------------------------------
//
// This header provides convenient literal syntax for creating Rational and
// GaussQi objects:
//
//   auto a = 0_r;           // Rational(0)
//   auto b = 42_r;          // Rational(42)
//   auto c = "0.5"_r;       // Rational(1/2)
//   auto d = "1/3"_r;       // Rational(1/3)
//
//   auto z1 = "1+2i"_qi;    // GaussQi(1,2)
//   auto z2 = "1/2-3/4i"_qi;// GaussQi(1/2, -3/4)
//   auto z3 = "0.333i"_qi;  // GaussQi(0, 333/1000)
//   auto z4 = "i"_qi;       // GaussQi(0,1)
//   auto z5 = "-i"_qi;      // GaussQi(0,-1)
//   auto z6 = "2"_qi;       // GaussQi(2,0)
//
// -----------------------------------------------------------------------------
// SYNTAX FOR GAUSSQI LITERALS
// -----------------------------------------------------------------------------
//
// The string literal for _qi accepts the following forms (spaces are ignored):
//   - "a+bi"   – a is real part, b is imaginary coefficient (both optional)
//   - "a-bi"
//   - "a+i"
//   - "a-i"
//   - "bi"     – real part = 0
//   - "i"      – real = 0, imag = 1
//   - "-i"     – real = 0, imag = -1
//   - "a"      – purely real (no 'i' character)
//   - "(a,b)"  – alternative comma‑separated form
//
// Both a and b can be integers, decimals ("0.5"), or fractions ("1/3").
// The coefficient b is interpreted as the number multiplied by i, so "2i" means imag = 2,
// "1/2i" means imag = 1/2, etc.
//
// -----------------------------------------------------------------------------
// WHY 0.5_r IS NOT SUPPORTED (REPEATED FROM RATIONAL LITERALS)
// -----------------------------------------------------------------------------
// The syntax 0.5_r would involve a floating‑point literal of type double,
// which loses exactness. Therefore we require string literals for decimal
// fractions. The same reasoning applies to _qi: "0.5i"_qi is allowed,
// but 0.5i_qi is NOT (nor is it valid C++ syntax).
//
// -----------------------------------------------------------------------------
#pragma once

#include "rational_class.h"
#include "gauss_qi.h"
#include <string_view>
#include <algorithm>
#include <cctype>

namespace delta {

    // ========================================================================
    // Rational literal (integer and string forms)
    // ========================================================================

    inline Rational operator""_r(unsigned long long num) {
        return Rational(num);
    }

    inline Rational operator""_r(const char* str, std::size_t len) {
        return Rational(std::string(str, len));
    }

#ifdef __cpp_user_defined_literals_floating_point
    template<char... digits>
    inline Rational operator""_r() = delete;
#endif

    // ========================================================================
    // GaussQi literal: "a+bi"_qi, "i"_qi, "(a,b)"_qi, etc.
    // ========================================================================

    namespace internal {
        inline std::string remove_whitespace(std::string_view sv) {
            std::string result;
            result.reserve(sv.size());
            for (char c : sv) {
                if (!std::isspace(static_cast<unsigned char>(c)))
                    result.push_back(c);
            }
            return result;
        }

        // Исправленный парсер коэффициента при i
        inline Rational parse_imag_coefficient(const std::string& s) {
            if (s.empty() || s == "+") return Rational(1);
            if (s == "-") return Rational(-1);
            std::string coeff = s;
            if (coeff.front() == '+') coeff.erase(0, 1);
            return Rational(coeff);
        }
    } // namespace internal

    inline GaussQi operator""_qi(const char* str, std::size_t len) {
        std::string s = internal::remove_whitespace(std::string_view(str, len));
        if (s.empty())
            return GaussQi(Rational(0));

        if (s.front() == '(' && s.back() == ')') {
            std::string inner = s.substr(1, s.size() - 2);
            size_t comma = inner.find(',');
            if (comma != std::string::npos) {
                std::string re_str = inner.substr(0, comma);
                std::string im_str = inner.substr(comma + 1);
                return GaussQi(Rational(re_str), Rational(im_str));
            }
        }

        size_t i_pos = s.find('i');
        if (i_pos == std::string::npos) {
            return GaussQi(Rational(s));
        }

        std::string before_i = s.substr(0, i_pos);
        if (before_i.empty()) {
            return GaussQi(Rational(0), Rational(1));
        }

        size_t sep_pos = std::string::npos;
        bool found = false;
        for (size_t j = before_i.length(); j > 0; --j) {
            char c = before_i[j - 1];
            if (c == '+' || c == '-') {
                if (j == 1 && before_i.length() > 1 && (c == '-' || c == '+')) {
                    sep_pos = 0;
                    found = true;
                    break;
                }
                else if (j > 1) {
                    sep_pos = j - 1;
                    found = true;
                    break;
                }
            }
        }

        std::string real_part_str;
        std::string imag_coeff_str;

        if (!found) {
            imag_coeff_str = before_i;
            real_part_str = "";
        }
        else if (sep_pos == 0) {
            imag_coeff_str = before_i;
            real_part_str = "";
        }
        else {
            real_part_str = before_i.substr(0, sep_pos);
            imag_coeff_str = before_i.substr(sep_pos);
        }

        Rational real_part = real_part_str.empty() ? Rational(0) : Rational(real_part_str);
        Rational imag_part = internal::parse_imag_coefficient(imag_coeff_str);

        return GaussQi(real_part, imag_part);
    }
    // Integer literal: 42_qi → GaussQi(42, 0)
    inline GaussQi operator""_qi(unsigned long long num) {
        return GaussQi(Rational(num), Rational(0));
    }
} // namespace delta

namespace delta::literals {
    using delta::operator""_r;
    using delta::operator""_qi;
}