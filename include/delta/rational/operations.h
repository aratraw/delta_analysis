#pragma once

#include "rational_class.h"
#include "context.h"
#include "eager.h"
#include "expression_root_factories.h"
#include "comparisons.h"

#include <vector>
#include <algorithm>
#include <memory>
#include <functional>

namespace delta {

    // ----------------------------------------------------------------------------
    // Binary arithmetic operators
    // ----------------------------------------------------------------------------

    inline Rational operator+(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode || (a.is_immediate() && b.is_immediate())) {
            return internal::eager_add(a, b);
        }
        auto root = internal::make_add(a, b);
        return Rational(std::make_shared<const internal::ExpressionRoot>(std::move(root)));
    }

    inline Rational operator-(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode || (a.is_immediate() && b.is_immediate())) {
            return internal::eager_sub(a, b);
        }
        auto root = internal::make_sub(a, b);
        return Rational(std::make_shared<const internal::ExpressionRoot>(std::move(root)));
    }

    inline Rational operator*(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode || (a.is_immediate() && b.is_immediate())) {
            return internal::eager_mul(a, b);
        }
        auto root = internal::make_mul(a, b);
        return Rational(std::make_shared<const internal::ExpressionRoot>(std::move(root)));
    }

    inline Rational operator/(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode || (a.is_immediate() && b.is_immediate())) {
            return internal::eager_div(a, b);
        }
        auto root = internal::make_div(a, b);
        return Rational(std::make_shared<const internal::ExpressionRoot>(std::move(root)));
    }

    // ----------------------------------------------------------------------------
    // Unary arithmetic operator
    // ----------------------------------------------------------------------------

    inline Rational operator-(const Rational& a) {
        if (internal::global_eager_mode || a.is_immediate()) {
            return internal::eager_neg(a);
        }
        auto root = internal::make_neg(a);
        return Rational(std::make_shared<const internal::ExpressionRoot>(std::move(root)));
    }

    // ----------------------------------------------------------------------------
    // Compound assignment operators
    // ----------------------------------------------------------------------------

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

    // ----------------------------------------------------------------------------
    // Batch addition
    // ----------------------------------------------------------------------------

    inline Rational batch_add(const std::vector<Rational>& terms) {
        if (terms.empty()) return Rational(0);

        // Check if all terms are immediate
        bool all_immediate = std::all_of(terms.begin(), terms.end(),
            [](const Rational& r) { return r.is_immediate(); });

        if (all_immediate) {
            // Use arbitrary-precision arithmetic to avoid overflow
            boost::multiprecision::cpp_int common_denom(1);
            std::vector<boost::multiprecision::cpp_int> nums;
            nums.reserve(terms.size());

            for (const Rational& term : terms) {
                if (term.as_small()) {
                    internal::SmallStorage s = *term.as_small();
                    s.normalize();
                    nums.push_back(internal::to_cpp_int(s.num));
                    boost::multiprecision::cpp_int den = internal::to_cpp_int(s.den);
                    common_denom = boost::multiprecision::lcm(common_denom, den);
                }
                else if (term.as_big()) {
                    const auto& b = *term.as_big();
                    nums.push_back(b.num());
                    common_denom = boost::multiprecision::lcm(common_denom, b.den());
                }
                else {
                    // Should not happen
                    all_immediate = false;
                    break;
                }
            }

            if (all_immediate) {
                boost::multiprecision::cpp_int sum_num(0);
                for (size_t i = 0; i < terms.size(); ++i) {
                    boost::multiprecision::cpp_int factor = common_denom;
                    if (terms[i].as_small()) {
                        internal::SmallStorage s = *terms[i].as_small();
                        s.normalize();
                        factor /= internal::to_cpp_int(s.den);
                        sum_num += nums[i] * factor;
                    }
                    else if (terms[i].as_big()) {
                        const auto& b = *terms[i].as_big();
                        factor /= b.den();
                        sum_num += b.num() * factor;
                    }
                }
                boost::multiprecision::cpp_int g = boost::multiprecision::gcd(sum_num, common_denom);
                sum_num /= g;
                common_denom /= g;

                if (sum_num <= internal::to_cpp_int((std::numeric_limits<absl::int128>::max)()) &&
                    common_denom <= internal::to_cpp_int((std::numeric_limits<absl::uint128>::max)())) {
                    return Rational(
                        internal::int128_from_string(sum_num.str()),
                        internal::uint128_from_string(common_denom.str())
                    );
                }
                else {
                    return Rational(sum_num, common_denom);
                }
            }
        }

        // Build a balanced lazy addition tree
        std::function<internal::ExpressionRoot(int, int)> build_tree =
            [&](int l, int r) -> internal::ExpressionRoot {
            if (l == r) {
                // Convert single term to lazy ExpressionRoot
                auto root_ptr = internal::to_lazy_root(terms[l]);
                return *root_ptr;
            }
            int mid = l + (r - l) / 2;
            internal::ExpressionRoot left = build_tree(l, mid);
            internal::ExpressionRoot right = build_tree(mid + 1, r);
            return left.add(right);
            };

        internal::ExpressionRoot root = build_tree(0, static_cast<int>(terms.size()) - 1);
        return Rational(std::make_shared<const internal::ExpressionRoot>(std::move(root)));
    }

    // ----------------------------------------------------------------------------
    // Helper functions
    // ----------------------------------------------------------------------------

    inline Rational abs(const Rational& x) {
        return x < Rational(0) ? -x : x;
    }

    //commented out for duplication, delete completely after successful compiling.
    //inline std::string to_string(const Rational& r) {
    //    if (r.is_immediate()) {
    //        internal::Value v = std::visit([](const auto& val) -> internal::Value { return val; }, r.storage_);
    //        return internal::to_string(v);
    //    }
    //    else {
    //        return internal::to_string(r.eval());
    //    }
    //}

} // namespace delta