#pragma once

#include "rational_class.h"
#include <string_view>

namespace delta {

    inline Rational operator""_r(unsigned long long num) {
        return Rational(static_cast<absl::int128>(num));
    }

    inline Rational operator""_r(const char* str, std::size_t len) {
        return Rational(std::string(str, len));
    }

#ifdef __cpp_user_defined_literals_floating_point
    template<char... digits>
    inline Rational operator""_r() {
        constexpr char str[] = { digits..., '\0' };
        return Rational(std::string(str));
    }
#endif

} // namespace delta