#pragma once

#include "rational_fwd.h"
#include "utils.h"

#include <absl/numeric/int128.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/integer.hpp>   // for lcm (used in operations.h)

#include <memory>
#include <stdexcept>
#include <variant>

namespace delta::internal {

    // ============================================================================
    // SmallStorage – 128‑bit rational with lazy normalization
    // ============================================================================
    struct SmallStorage {
        absl::int128 num;
        absl::uint128 den;
        bool reduced;

        constexpr SmallStorage() noexcept : num(0), den(1), reduced(true) {}

        constexpr explicit SmallStorage(absl::int128 n) noexcept
            : num(n), den(1), reduced(true) {
        }

        constexpr SmallStorage(absl::int128 n, absl::uint128 d) noexcept
            : num(n), den(d), reduced(false) {
        }

        void normalize() {
            if (reduced) return;
            if (den == 0) {
                throw std::domain_error("SmallStorage: denominator cannot be zero");
            }
            if (num == 0) {
                den = 1;
                reduced = true;
                return;
            }

            // Ensure denominator is positive
            if (den < 0) {
                den = -den;
                num = -num;
            }

            // Reduce by GCD
            absl::uint128 g = gcd(static_cast<absl::uint128>(num < 0 ? -num : num), den);
            if (g > 1) {
                num = static_cast<absl::int128>(static_cast<absl::uint128>(num < 0 ? -num : num) / g);
                if (num < 0) num = -num;
                den /= g;
            }
            reduced = true;
        }
    };

    // ============================================================================
    // BigStorageImpl – actual arbitrary‑precision storage (immutable)
    // ============================================================================
    struct BigStorageImpl {
        boost::multiprecision::cpp_int num;
        boost::multiprecision::cpp_int den;   // > 0

        BigStorageImpl() = default;

        BigStorageImpl(const boost::multiprecision::cpp_int& n,
            const boost::multiprecision::cpp_int& d)
            : num(n), den(d)
        {
            normalize();
        }

        void normalize() {
            if (den == 0) {
                throw std::domain_error("BigStorage: denominator cannot be zero");
            }
            if (num == 0) {
                den = 1;
                return;
            }

            // Ensure denominator is positive
            if (den < 0) {
                den = -den;
                num = -num;
            }

            // Reduce by GCD
            boost::multiprecision::cpp_int g = boost::multiprecision::gcd(num, den);
            if (g > 1) {
                num /= g;
                den /= g;
            }
        }
    };

    // ============================================================================
    // BigStorage – shared‑ptr wrapper for arbitrary‑precision rationals
    // ============================================================================
    struct BigStorage {
        std::shared_ptr<const BigStorageImpl> data;

        BigStorage() : data(std::make_shared<const BigStorageImpl>()) {}

        explicit BigStorage(const boost::multiprecision::cpp_int& n)
            : data(std::make_shared<const BigStorageImpl>(n, 1)) {
        }

        BigStorage(const boost::multiprecision::cpp_int& n,
            const boost::multiprecision::cpp_int& d)
            : data(std::make_shared<const BigStorageImpl>(n, d)) {
        }

        const boost::multiprecision::cpp_int& num() const noexcept { return data->num; }
        const boost::multiprecision::cpp_int& den() const noexcept { return data->den; }
    };

    // ============================================================================
    // Value – variant of immediate rationals
    // ============================================================================
    using Value = std::variant<SmallStorage, BigStorage>;

    // ----------------------------------------------------------------------------
    // Comparison operators for Value (full comparison, normalizing if needed)
    // ----------------------------------------------------------------------------

    // Helper: convert a Value to a pair of cpp_int (num, den) for comparison
    inline std::pair<boost::multiprecision::cpp_int, boost::multiprecision::cpp_int>
        normalize_to_cpp_int(const Value& v) {
        if (const auto* small = std::get_if<SmallStorage>(&v)) {
            SmallStorage s = *small;
            s.normalize();  // ensure reduced
            return { to_cpp_int(s.num), to_cpp_int(s.den) };
        }
        else if (const auto* big = std::get_if<BigStorage>(&v)) {
            return { big->num(), big->den() };
        }
        throw std::invalid_argument("Invalid variant state");
    }

    inline bool operator==(const Value& a, const Value& b) {
        auto [anum, aden] = normalize_to_cpp_int(a);
        auto [bnum, bden] = normalize_to_cpp_int(b);
        // Compare cross‑multiplication
        return anum * bden == bnum * aden;
    }

    inline bool operator<(const Value& a, const Value& b) {
        auto [anum, aden] = normalize_to_cpp_int(a);
        auto [bnum, bden] = normalize_to_cpp_int(b);
        return anum * bden < bnum * aden;
    }

    inline bool operator>(const Value& a, const Value& b) {
        return b < a;
    }

    inline bool operator<=(const Value& a, const Value& b) {
        return !(b < a);
    }

    inline bool operator>=(const Value& a, const Value& b) {
        return !(a < b);
    }

    inline std::string to_string(const SmallStorage& s) {
        SmallStorage norm = s;
        norm.normalize();
        if (norm.den == 1) return int128_to_string(norm.num);
        return int128_to_string(norm.num) + "/" + uint128_to_string(norm.den);
    }

    inline std::string to_string(const BigStorage& b) {
        if (b.den() == 1) return b.num().str();
        return b.num().str() + "/" + b.den().str();
    }

    inline std::string to_string(const Value& v) {
        if (const auto* s = std::get_if<SmallStorage>(&v))
            return to_string(*s);
        else
            return to_string(std::get<BigStorage>(v));
    }

} // namespace delta::internal