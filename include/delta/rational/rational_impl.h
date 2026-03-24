#pragma once

#include "rational_class.h"
#include "expression_root_impl.h"
#include "eager.h"
#include "expression_root_factories.h"
#include "context.h"
#include "utils.h"

#include <absl/numeric/int128.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <cctype>
#include <stdexcept>
#include <string>
#include <variant>

namespace delta {

    // ----------------------------------------------------------------------------
    // Constructors
    // ----------------------------------------------------------------------------

    inline Rational::Rational() noexcept : storage_(internal::SmallStorage{}) {}

    inline Rational::Rational(absl::int128 num) : storage_(internal::SmallStorage(num)) {}

    inline Rational::Rational(absl::int128 num, absl::uint128 den)
        : storage_(internal::SmallStorage(num, den)) {
    }

    inline Rational::Rational(const boost::multiprecision::cpp_int& num)
        : storage_(internal::BigStorage(num)) {
    }

    inline Rational::Rational(const boost::multiprecision::cpp_int& num,
        const boost::multiprecision::cpp_int& den)
        : storage_(internal::BigStorage(num, den)) {
    }

    inline Rational::Rational(int num, int den)
        : Rational(static_cast<absl::int128>(num), static_cast<absl::uint128>(den)) {
    }

    inline Rational::Rational(const std::string& s) {
        // Parse "numerator/denominator" or "integer.fraction"
        size_t slash = s.find('/');
        if (slash != std::string::npos) {
            std::string num_str = s.substr(0, slash);
            std::string den_str = s.substr(slash + 1);
            boost::multiprecision::cpp_int num(num_str);
            boost::multiprecision::cpp_int den(den_str);
            if (den == 0) throw std::domain_error("Denominator cannot be zero");
            // Normalize sign: denominator positive
            if (den < 0) {
                den = -den;
                num = -num;
            }
            // Reduce
            boost::multiprecision::cpp_int g = boost::multiprecision::gcd(num, den);
            num /= g;
            den /= g;

            // Choose storage based on size
            if (num <= internal::to_cpp_int((std::numeric_limits<absl::int128>::max)()) &&
                den <= internal::to_cpp_int((std::numeric_limits<absl::uint128>::max)())) {
                // Convert through string to avoid direct static_cast
                storage_ = internal::SmallStorage(
                    internal::int128_from_string(num.str()),
                    internal::uint128_from_string(den.str())
                );
            }
            else {
                storage_ = internal::BigStorage(num, den);
            }
        }
        else {
            // Parse decimal: "0.125" or "123"
            size_t dot = s.find('.');
            if (dot == std::string::npos) {
                // Integer
                boost::multiprecision::cpp_int num(s);
                storage_ = internal::BigStorage(num);
            }
            else {
                std::string int_part = s.substr(0, dot);
                std::string frac_part = s.substr(dot + 1);
                // Remove trailing zeros to avoid huge denominators
                while (!frac_part.empty() && frac_part.back() == '0')
                    frac_part.pop_back();
                boost::multiprecision::cpp_int num(int_part + frac_part);
                boost::multiprecision::cpp_int den(1);
                for (size_t i = 0; i < frac_part.size(); ++i)
                    den *= 10;
                // Reduce
                boost::multiprecision::cpp_int g = boost::multiprecision::gcd(num, den);
                num /= g;
                den /= g;
                // Handle sign
                if (int_part.front() == '-') {
                    num = -num;
                }
                if (num <= internal::to_cpp_int((std::numeric_limits<absl::int128>::max)()) &&
                    den <= internal::to_cpp_int((std::numeric_limits<absl::uint128>::max)())) {
                    storage_ = internal::SmallStorage(
                        internal::int128_from_string(num.str()),
                        internal::uint128_from_string(den.str())
                    );
                }
                else {
                    storage_ = internal::BigStorage(num, den);
                }
            }
        }
    }

    // Private constructors
    inline Rational::Rational(internal::Value val) : storage_() {
        if (std::holds_alternative<internal::SmallStorage>(val)) {
            storage_ = std::get<internal::SmallStorage>(std::move(val));
        }
        else if (std::holds_alternative<internal::BigStorage>(val)) {
            storage_ = std::get<internal::BigStorage>(std::move(val));
        }
    }
    inline Rational::Rational(std::shared_ptr<const internal::ExpressionRoot> root)
        : storage_(std::move(root)) {
    }

    // ----------------------------------------------------------------------------
    // State queries
    // ----------------------------------------------------------------------------

    inline bool Rational::is_immediate() const noexcept {
        return std::holds_alternative<internal::SmallStorage>(storage_) ||
            std::holds_alternative<internal::BigStorage>(storage_);
    }

    inline bool Rational::is_lazy() const noexcept {
        return std::holds_alternative<std::shared_ptr<const internal::ExpressionRoot>>(storage_);
    }

    // ----------------------------------------------------------------------------
    // Raw accessors
    // ----------------------------------------------------------------------------

    inline const internal::SmallStorage* Rational::as_small() const noexcept {
        return std::get_if<internal::SmallStorage>(&storage_);
    }

    inline const internal::BigStorage* Rational::as_big() const noexcept {
        return std::get_if<internal::BigStorage>(&storage_);
    }

    inline const std::shared_ptr<const internal::ExpressionRoot>& Rational::as_lazy() const {
        return std::get<std::shared_ptr<const internal::ExpressionRoot>>(storage_);
    }

    // ----------------------------------------------------------------------------
    // Conversions
    // ----------------------------------------------------------------------------

    inline Rational Rational::lazy() const {
        if (is_lazy()) return *this;
        internal::Value val;
        if (auto* s = std::get_if<internal::SmallStorage>(&storage_)) {
            val = *s;
        }
        else if (auto* b = std::get_if<internal::BigStorage>(&storage_)) {
            val = *b;
        }
        else {
            // не должно произойти, так как is_lazy() == false
            throw std::logic_error("Rational::lazy: unexpected variant state");
        }
        auto root = std::make_shared<const internal::ExpressionRoot>(val);
        return Rational(std::move(root));
    }

    // ----------------------------------------------------------------------------
    // Simplification and evaluation
    // ----------------------------------------------------------------------------

    inline Rational Rational::simplify() const {
        if (is_immediate()) return *this;
        auto new_root = as_lazy()->simplify();
        // If the simplified tree is a single CONST node, collapse it to immediate
        if (new_root.nodes().size() == 1 && new_root.nodes()[0].op == internal::LazyOp::CONST) {
            return Rational(new_root.values()[new_root.nodes()[0].value_idx]);
        }
        return Rational(std::make_shared<const internal::ExpressionRoot>(std::move(new_root)));
    }

    inline internal::Value Rational::eval() const {
        if (is_immediate()) {
            if (auto* s = std::get_if<internal::SmallStorage>(&storage_)) {
                return *s;
            }
            else if (auto* b = std::get_if<internal::BigStorage>(&storage_)) {
                return *b;
            }
            else {
                throw std::logic_error("Rational::eval: unexpected variant state");
            }
        }
        // First simplify, then evaluate
        auto simplified = simplify();
        if (simplified.is_immediate()) {
            return simplified.eval();
        }
        return simplified.as_lazy()->eval();
    }

    // ----------------------------------------------------------------------------
    // Interval estimation
    // ----------------------------------------------------------------------------

    inline internal::Interval Rational::approx_interval() const noexcept {
        if (is_immediate()) {
            internal::Value val;
            if (auto* s = std::get_if<internal::SmallStorage>(&storage_)) {
                val = *s;
            }
            else if (auto* b = std::get_if<internal::BigStorage>(&storage_)) {
                val = *b;
            }
            else {
                return internal::Interval(); // fallback, не должно случиться
            }
            double d = internal::to_double(val);
            return internal::Interval(d);
        }
        return as_lazy()->nodes()[as_lazy()->root_index()].approx;
    }

    // ----------------------------------------------------------------------------
    // String conversion (to_string)
    // ----------------------------------------------------------------------------

    inline std::string to_string(const Rational& r) {
        if (r.is_immediate()) {
            internal::Value v;
            if (const auto* s = std::get_if<internal::SmallStorage>(&r.storage_)) {
                v = *s;
            }
            else if (const auto* b = std::get_if<internal::BigStorage>(&r.storage_)) {
                v = *b;
            }
            else {
                // is_immediate() гарантирует, что здесь SmallStorage или BigStorage
                throw std::logic_error("to_string: invalid immediate variant");
            }
            return internal::to_string(v);
        }
        // Ленивый случай: вычисляем значение и преобразуем
        return internal::to_string(r.eval());
    }

} // namespace delta