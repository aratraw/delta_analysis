// include/delta/rational/literals.h
#pragma once

#include "delta/rational/rational_class.h"
#include <string>

namespace delta {

    // -------------------------------------------------------------------------
    // Integer literal – constructs Rational directly from an unsigned long long
    // -------------------------------------------------------------------------
    inline Rational operator""_r(unsigned long long num) {
        return Rational(static_cast<absl::int128>(num));
    }

    // -------------------------------------------------------------------------
    // String literal – parses decimal or fractional representation
    // -------------------------------------------------------------------------
    inline Rational operator""_r(const char* str, std::size_t len) {
        std::string s(str, len);
        return Rational(s);
    }

#ifdef __cpp_user_defined_literals_floating_point
    // -------------------------------------------------------------------------
    // C++23 compile‑time string literal (exact digits, no intermediate double)
    // -------------------------------------------------------------------------
    template<char... digits>
    Rational operator""_r() {
        constexpr char str[] = { digits..., '\0' };
        return operator""_r(str, sizeof(str) - 1);
    }
#endif

} // namespace delta