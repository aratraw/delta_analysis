// include/delta/rational/batch_arithmetic.h
#pragma once

#include "delta/rational/rational_class.h"
#include "delta/rational/evaluation.h"   // for evaluate()
#include "delta/rational/utils.h"        // for lcm, gcd
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>

namespace delta::internal {

    // -------------------------------------------------------------------------
    // Batch addition of rational numbers using a common denominator.
    // Assumes that all terms are already evaluated (non-lazy) or will be evaluated.
    // -------------------------------------------------------------------------
    inline Rational batch_add(const std::vector<Rational>& terms) {
        if (terms.empty()) return Rational(0);

        using boost::multiprecision::cpp_int;

        // First, evaluate all terms to ensure they are non-lazy.
        // We'll store evaluated versions and their denominators.
        struct EvaluatedTerm {
            cpp_int numerator;
            cpp_int denominator;
        };
        std::vector<EvaluatedTerm> ev_terms;
        ev_terms.reserve(terms.size());

        cpp_int common_denom = 1;

        for (const auto& t : terms) {
            Rational ev = t.is_lazy() ? evaluate(t) : t;

            cpp_int num, den;
            if (ev.is_small()) {
                const auto& s = *ev.as_small();
                // Normalize just in case
                internal::SmallStorage copy = s;
                copy.normalize();
                num = internal::to_cpp_int(copy.num);   // fixed
                den = internal::to_cpp_int(copy.den);   // fixed
            }
            else if (ev.is_big()) {
                const auto& b = *ev.as_big();
                num = b.num;
                den = b.den;
            }
            else {
                // Should not happen after evaluate
                continue;
            }

            // Update common denominator as LCM of current common_denom and this term's denominator
            cpp_int g = boost::multiprecision::gcd(common_denom, den);
            common_denom = (common_denom / g) * den;

            ev_terms.push_back({ std::move(num), std::move(den) });
        }

        // Accumulate numerator after scaling each term to common denominator
        cpp_int sum_num = 0;
        for (const auto& term : ev_terms) {
            cpp_int factor = common_denom / term.denominator;
            sum_num += term.numerator * factor;
        }

        // Reduce the result
        cpp_int g = boost::multiprecision::gcd(sum_num, common_denom);
        if (g > 1) {
            sum_num /= g;
            common_denom /= g;
        }

        // Ensure denominator is positive
        if (common_denom < 0) {
            common_denom = -common_denom;
            sum_num = -sum_num;
        }

        // Construct the result, letting Rational choose the storage type
        if (common_denom == 1) {
            return Rational(sum_num);
        }
        return Rational(sum_num, common_denom);
    }

} // namespace delta::internal