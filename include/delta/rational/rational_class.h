// include/delta/rational/rational_class.h
#pragma once

#include <absl/numeric/int128.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <sstream>
#include <stdexcept>

#include "delta/rational/context.h"   // для MAX_LAZY_DEPTH, global_eager_mode

#include "delta/rational/storage.h"
#include "delta/rational/utils.h"
#include "delta/rational/interval.h"

namespace delta {

    class Rational {
    public:
        // -------------------------------------------------------------------------
        // Constructors
        // -------------------------------------------------------------------------
        Rational() : storage_(internal::SmallStorage{}) {}

        explicit Rational(absl::int128 num) : storage_(internal::SmallStorage(num)) {}
        explicit Rational(absl::int128 num, absl::uint128 den) : storage_(internal::SmallStorage(num, den)) {}

        explicit Rational(const boost::multiprecision::cpp_int& num) : storage_(internal::BigStorage(num)) {}
        Rational(const boost::multiprecision::cpp_int& num, const boost::multiprecision::cpp_int& den)
            : storage_(internal::BigStorage(num, den)) {
        }

        explicit Rational(const std::string& s);

        // Convenience constructors for integer types
        Rational(long long num) : Rational(static_cast<absl::int128>(num)) {}
        Rational(unsigned long long num) : Rational(static_cast<absl::int128>(num)) {}
        Rational(int num) : Rational(static_cast<absl::int128>(num)) {}

        // New: two‑int constructor for small fractions (e.g., Rational(1,5))
        Rational(int num, int den) : Rational(static_cast<absl::int128>(num), static_cast<absl::uint128>(den)) {}

        explicit Rational(std::shared_ptr<internal::LazyNode> node) : storage_(std::move(node)) {}

        Rational(const Rational&) = default;
        Rational(Rational&&) = default;
        Rational& operator=(const Rational&) = default;
        Rational& operator=(Rational&&) = default;

        // -------------------------------------------------------------------------
        // Type queries
        // -------------------------------------------------------------------------
        bool is_lazy() const {
            return std::holds_alternative<std::shared_ptr<internal::LazyNode>>(storage_);
        }
        bool is_small() const {
            return std::holds_alternative<internal::SmallStorage>(storage_);
        }
        bool is_big() const {
            return std::holds_alternative<internal::BigStorage>(storage_);
        }

        const internal::SmallStorage* as_small() const {
            return std::get_if<internal::SmallStorage>(&storage_);
        }
        const internal::BigStorage* as_big() const {
            return std::get_if<internal::BigStorage>(&storage_);
        }
        const std::shared_ptr<internal::LazyNode>& as_lazy() const {
            return std::get<std::shared_ptr<internal::LazyNode>>(storage_);
        }

        // -------------------------------------------------------------------------
        // Depth of lazy tree (0 for non‑lazy)
        // -------------------------------------------------------------------------
        int depth() const {
            if (is_lazy()) return as_lazy()->depth;
            return 0;
        }

        // -------------------------------------------------------------------------
        // Approximate interval (used for fast comparisons)
        // -------------------------------------------------------------------------
        internal::Interval approx_interval() const {
            if (is_lazy()) {
                return as_lazy()->approx;
            }
            double val;
            if (is_small()) {
                const auto& s = *as_small();
                // Use normalised values to avoid huge intermediate results
                internal::SmallStorage copy = s;
                copy.normalize();
                val = static_cast<double>(copy.num) / static_cast<double>(copy.den);
            }
            else {
                const auto& b = *as_big();
                // Use Boost's decimal float for safe conversion to double
                boost::multiprecision::cpp_dec_float_100 num_float(b.num);
                boost::multiprecision::cpp_dec_float_100 den_float(b.den);
                val = static_cast<double>(num_float / den_float);
            }
            return internal::Interval(val);
        }

        // -------------------------------------------------------------------------
        // Force evaluation (returns a fully computed rational)
        // -------------------------------------------------------------------------
        Rational evaluate() const {
            return internal::evaluate(*this);
        }

        // -------------------------------------------------------------------------
        // String representation (for debugging)
        // -------------------------------------------------------------------------
        std::string to_string() const;

        // -------------------------------------------------------------------------
        // Internal storage access (for advanced use)
        // -------------------------------------------------------------------------
        const internal::Storage& storage() const { return storage_; }

    private:
        internal::Storage storage_;

        friend class internal::LazyNode;
        friend Rational internal::evaluate(const Rational&);
        friend Rational internal::simplify(const Rational&);
    };

    // -------------------------------------------------------------------------
    // String constructor implementation (simplified to avoid conversions)
    // -------------------------------------------------------------------------
    inline Rational::Rational(const std::string& s) {
        size_t slash = s.find('/');
        if (slash != std::string::npos) {
            // Fraction: numerator/denominator
            std::string num_str = s.substr(0, slash);
            std::string den_str = s.substr(slash + 1);
            boost::multiprecision::cpp_int num(num_str);
            boost::multiprecision::cpp_int den(den_str);
            if (den == 0) throw std::domain_error("zero denominator");
            // Always use BigStorage to avoid tricky conversions
            storage_ = internal::BigStorage(num, den);
        }
        else {
            // Decimal string: parse as rational
            std::string str = s;
            size_t dot = str.find('.');
            if (dot == std::string::npos) {
                // Integer
                boost::multiprecision::cpp_int num(str);
                storage_ = internal::BigStorage(num);
            }
            else {
                // Decimal: extract integer and fractional parts
                std::string int_part = str.substr(0, dot);
                std::string frac_part = str.substr(dot + 1);
                // Remove trailing zeros (optional)
                while (!frac_part.empty() && frac_part.back() == '0')
                    frac_part.pop_back();
                if (frac_part.empty()) {
                    // No fractional part -> integer
                    *this = Rational(int_part);
                    return;
                }
                boost::multiprecision::cpp_int numerator(int_part + frac_part);
                boost::multiprecision::cpp_int denominator = boost::multiprecision::pow(boost::multiprecision::cpp_int(10), frac_part.size());
                // Handle negative sign
                bool negative = false;
                if (int_part.size() > 0 && int_part[0] == '-') {
                    negative = true;
                    numerator = -numerator;
                }
                // Normalise (divide by gcd)
                boost::multiprecision::cpp_int g = boost::multiprecision::gcd(numerator, denominator);
                numerator /= g;
                denominator /= g;
                // Store
                storage_ = internal::BigStorage(numerator, denominator);
            }
        }
    }

    // -------------------------------------------------------------------------
    // to_string implementation
    // -------------------------------------------------------------------------
    inline std::string Rational::to_string() const {
        if (is_lazy()) {
            // Force evaluation for output (or we could output "lazy(...)").
            Rational ev = internal::evaluate(*this);
            return ev.to_string();
        }
        std::ostringstream oss;
        if (is_small()) {
            const auto& s = *as_small();
            // Ensure normalised for output
            internal::SmallStorage copy = s;
            copy.normalize();
            if (copy.den == 1) {
                oss << copy.num;
            }
            else {
                oss << copy.num << '/' << copy.den;
            }
        }
        else {
            const auto& b = *as_big();
            // BigStorage is already normalised by construction
            if (b.den == 1) {
                oss << b.num;
            }
            else {
                oss << b.num << '/' << b.den;
            }
        }
        return oss.str();
    }

    // -------------------------------------------------------------------------
    // Absolute value (defined after class)
    // -------------------------------------------------------------------------
    inline Rational abs(const Rational& x) {
        if (x < Rational(0)) return -x;
        return x;
    }

    // -------------------------------------------------------------------------
    // Arithmetic operators (lazy or eager)
    // -------------------------------------------------------------------------

    inline Rational operator+(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode) {
            return internal::eager_add(a, b);
        }
        absl::InlinedVector<std::shared_ptr<const Rational>, 2> args;
        args.emplace_back(std::make_shared<Rational>(a));
        args.emplace_back(std::make_shared<Rational>(b));
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::ADD, std::move(args), std::make_shared<const Rational>(default_eps()));
        Rational result(node);
        return internal::simplify(result);
    }

    inline Rational operator-(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode) {
            return internal::eager_sub(a, b);
        }
        absl::InlinedVector<std::shared_ptr<const Rational>, 2> args;
        args.emplace_back(std::make_shared<Rational>(a));
        args.emplace_back(std::make_shared<Rational>(b));
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::SUB, std::move(args), std::make_shared<const Rational>(default_eps()));
        Rational result(node);
        return internal::simplify(result);
    }

    inline Rational operator*(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode) {
            return internal::eager_mul(a, b);
        }
        absl::InlinedVector<std::shared_ptr<const Rational>, 2> args;
        args.emplace_back(std::make_shared<Rational>(a));
        args.emplace_back(std::make_shared<Rational>(b));
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::MUL, std::move(args), std::make_shared<const Rational>(default_eps()));
        Rational result(node);
        return internal::simplify(result);
    }

    inline Rational operator/(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode) {
            return internal::eager_div(a, b);
        }
        absl::InlinedVector<std::shared_ptr<const Rational>, 2> args;
        args.emplace_back(std::make_shared<Rational>(a));
        args.emplace_back(std::make_shared<Rational>(b));
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::DIV, std::move(args), std::make_shared<const Rational>(default_eps()));
        Rational result(node);
        return internal::simplify(result);
    }

    inline Rational operator-(const Rational& a) {
        if (internal::global_eager_mode) {
            return internal::eager_neg(a);
        }
        auto node = std::make_shared<internal::LazyNode>(internal::LazyOp::NEG, std::make_shared<Rational>(a), std::make_shared<const Rational>(default_eps()));
        Rational result(node);
        return internal::simplify(result);
    }

    // Compound assignment operators
    inline Rational& operator+=(Rational& a, const Rational& b) {
        a = a + b;
        return a;
    }
    inline Rational& operator-=(Rational& a, const Rational& b) {
        a = a - b;
        return a;
    }
    inline Rational& operator*=(Rational& a, const Rational& b) {
        a = a * b;
        return a;
    }
    inline Rational& operator/=(Rational& a, const Rational& b) {
        a = a / b;
        return a;
    }

    // -------------------------------------------------------------------------
    // Comparison operators
    // -------------------------------------------------------------------------

    inline bool operator==(const Rational& a, const Rational& b) {
        if (&a == &b) return true;
        if (a.is_lazy() && b.is_lazy()) {
            const auto& la = *a.as_lazy();
            const auto& lb = *b.as_lazy();
            if (internal::structurally_equal(la, lb)) return true;
        }
        internal::Interval ia = a.approx_interval();
        internal::Interval ib = b.approx_interval();
        if (!ia.overlaps(ib)) return false;
        Rational ea = internal::evaluate(a);
        Rational eb = internal::evaluate(b);
        if (ea.is_small() && eb.is_small()) {
            const auto& sa = *ea.as_small();
            const auto& sb = *eb.as_small();
            internal::SmallStorage sa_copy = sa;
            internal::SmallStorage sb_copy = sb;
            sa_copy.normalize();
            sb_copy.normalize();
            return sa_copy.num == sb_copy.num && sa_copy.den == sb_copy.den;
        }
        if (ea.is_big() && eb.is_big()) {
            const auto& ba = *ea.as_big();
            const auto& bb = *eb.as_big();
            return ba.num == bb.num && ba.den == bb.den;
        }
        if (ea.is_small() && eb.is_big()) {
            const auto& sa = *ea.as_small();
            const auto& bb = *eb.as_big();
            internal::SmallStorage sa_copy = sa;
            sa_copy.normalize();
            boost::multiprecision::cpp_int n = internal::to_cpp_int(sa_copy.num);
            boost::multiprecision::cpp_int d = internal::to_cpp_int(sa_copy.den);
            return n == bb.num && d == bb.den;
        }
        if (ea.is_big() && eb.is_small()) {
            const auto& ba = *ea.as_big();
            const auto& sb = *eb.as_small();
            internal::SmallStorage sb_copy = sb;
            sb_copy.normalize();
            boost::multiprecision::cpp_int n = internal::to_cpp_int(sb_copy.num);
            boost::multiprecision::cpp_int d = internal::to_cpp_int(sb_copy.den);
            return ba.num == n && ba.den == d;
        }
        return false;
    }

    inline bool operator!=(const Rational& a, const Rational& b) {
        return !(a == b);
    }

    inline bool operator<(const Rational& a, const Rational& b) {
        if (&a == &b) return false;
        if (a.is_lazy() && b.is_lazy()) {
            const auto& la = *a.as_lazy();
            const auto& lb = *b.as_lazy();
            if (internal::structurally_equal(la, lb)) return false;
        }
        internal::Interval ia = a.approx_interval();
        internal::Interval ib = b.approx_interval();
        if (ia.upper() < ib.lower()) return true;
        if (ia.lower() > ib.upper()) return false;
        Rational ea = internal::evaluate(a);
        Rational eb = internal::evaluate(b);
        if (ea.is_small() && eb.is_small()) {
            const auto& sa = *ea.as_small();
            const auto& sb = *eb.as_small();
            internal::SmallStorage sa_copy = sa;
            internal::SmallStorage sb_copy = sb;
            sa_copy.normalize();
            sb_copy.normalize();
            bool overflow = internal::would_overflow_mul(sa_copy.num, static_cast<absl::int128>(sb_copy.den)) ||
                internal::would_overflow_mul(sb_copy.num, static_cast<absl::int128>(sa_copy.den));
            if (!overflow) {
                absl::int128 lhs = sa_copy.num * static_cast<absl::int128>(sb_copy.den);
                absl::int128 rhs = sb_copy.num * static_cast<absl::int128>(sa_copy.den);
                return lhs < rhs;
            }
            else {
                boost::multiprecision::cpp_int n1 = internal::to_cpp_int(sa_copy.num);
                boost::multiprecision::cpp_int d1 = internal::to_cpp_int(sa_copy.den);
                boost::multiprecision::cpp_int n2 = internal::to_cpp_int(sb_copy.num);
                boost::multiprecision::cpp_int d2 = internal::to_cpp_int(sb_copy.den);
                return n1 * d2 < n2 * d1;
            }
        }
        if (ea.is_big() && eb.is_big()) {
            const auto& ba = *ea.as_big();
            const auto& bb = *eb.as_big();
            return ba.num * bb.den < bb.num * ba.den;
        }
        if (ea.is_small() && eb.is_big()) {
            const auto& sa = *ea.as_small();
            const auto& bb = *eb.as_big();
            internal::SmallStorage sa_copy = sa;
            sa_copy.normalize();
            boost::multiprecision::cpp_int n1 = internal::to_cpp_int(sa_copy.num);
            boost::multiprecision::cpp_int d1 = internal::to_cpp_int(sa_copy.den);
            return n1 * bb.den < bb.num * d1;
        }
        if (ea.is_big() && eb.is_small()) {
            const auto& ba = *ea.as_big();
            const auto& sb = *eb.as_small();
            internal::SmallStorage sb_copy = sb;
            sb_copy.normalize();
            boost::multiprecision::cpp_int n2 = internal::to_cpp_int(sb_copy.num);
            boost::multiprecision::cpp_int d2 = internal::to_cpp_int(sb_copy.den);
            return ba.num * d2 < n2 * ba.den;
        }
        return false;
    }

    inline bool operator>(const Rational& a, const Rational& b) {
        return b < a;
    }

    inline bool operator<=(const Rational& a, const Rational& b) {
        return !(a > b);
    }

    inline bool operator>=(const Rational& a, const Rational& b) {
        return !(a < b);
    }

} // namespace delta

namespace delta::internal {
    // Default epsilon value (перенесено из context.h)
    inline thread_local Rational default_eps_value = []() -> Rational {
        boost::multiprecision::cpp_int one(1);
        boost::multiprecision::cpp_int denom("1000000000000000000000000000000");
        return Rational(one, denom);
        }();
}

namespace delta {
    inline const Rational& default_eps() {
        return internal::default_eps_value;
    }
    inline void set_default_eps(const Rational& eps) {
        internal::default_eps_value = eps;
    }
}