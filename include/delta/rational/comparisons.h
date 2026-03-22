// include/delta/rational/comparisons.h
#pragma once

#include "delta/rational/rational_class.h"
#include "delta/rational/interval.h"
#include "delta/rational/evaluation.h"   // for evaluate()
#include "delta/rational/context.h"
#include "delta/rational/simplify.h"

namespace delta::internal {
    // Forward declaration – defined in simplify.h
    bool structurally_equal(const LazyNode& a, const LazyNode& b);
}

namespace delta {

    // -------------------------------------------------------------------------
    // Equality
    // -------------------------------------------------------------------------
    inline bool operator==(const Rational& a, const Rational& b) {
        // 1. Identity (same object)
        if (&a == &b) return true;

        // 2. Structural equality of lazy trees (fast)
        if (a.is_lazy() && b.is_lazy()) {
            const auto& la = *a.as_lazy();
            const auto& lb = *b.as_lazy();
            if (internal::structurally_equal(la, lb)) return true;
        }

        // 3. Interval prediction (if intervals do not overlap -> definitely not equal)
        Interval ia = a.approx_interval();
        Interval ib = b.approx_interval();
        if (!ia.overlaps(ib)) return false;

        // 4. Eager evaluation (compute exactly)
        Rational ea = internal::evaluate(a);
        Rational eb = internal::evaluate(b);

        // Compare the stored values – after evaluation both are non‑lazy
        if (ea.is_small() && eb.is_small()) {
            const auto& sa = *ea.as_small();
            const auto& sb = *eb.as_small();
            // Normalise both to ensure correct comparison (they might be reduced already)
            const_cast<internal::SmallStorage&>(sa).normalize();
            const_cast<internal::SmallStorage&>(sb).normalize();
            return sa.num == sb.num && sa.den == sb.den;
        }
        if (ea.is_big() && eb.is_big()) {
            const auto& ba = *ea.as_big();
            const auto& bb = *eb.as_big();
            // BigStorage is always normalised
            return ba.num == bb.num && ba.den == bb.den;
        }
        // Mixed small/big: convert small to big for comparison
        if (ea.is_small() && eb.is_big()) {
            const auto& sa = *ea.as_small();
            const auto& bb = *eb.as_big();
            sa.normalize();
            boost::multiprecision::cpp_int n(sa.num);
            boost::multiprecision::cpp_int d(sa.den);
            return n == bb.num && d == bb.den;
        }
        if (ea.is_big() && eb.is_small()) {
            const auto& ba = *ea.as_big();
            const auto& sb = *eb.as_small();
            sb.normalize();
            boost::multiprecision::cpp_int n(sb.num);
            boost::multiprecision::cpp_int d(sb.den);
            return ba.num == n && ba.den == d;
        }
        // Should not happen (both non‑lazy)
        return false;
    }

    inline bool operator!=(const Rational& a, const Rational& b) {
        return !(a == b);
    }

    // -------------------------------------------------------------------------
    // Less than
    // -------------------------------------------------------------------------
    inline bool operator<(const Rational& a, const Rational& b) {
        // 1. Identity – cannot be less than itself
        if (&a == &b) return false;

        // 2. Structural equality – if structurally equal, then not less
        if (a.is_lazy() && b.is_lazy()) {
            const auto& la = *a.as_lazy();
            const auto& lb = *b.as_lazy();
            if (internal::structurally_equal(la, lb)) return false;
        }

        // 3. Interval prediction
        Interval ia = a.approx_interval();
        Interval ib = b.approx_interval();
        if (ia.upper() < ib.lower()) return true;   // a definitely less
        if (ia.lower() > ib.upper()) return false;  // a definitely greater

        // 4. Eager evaluation
        Rational ea = internal::evaluate(a);
        Rational eb = internal::evaluate(b);

        // Compare the rationals exactly
        if (ea.is_small() && eb.is_small()) {
            const auto& sa = *ea.as_small();
            const auto& sb = *eb.as_small();
            sa.normalize();
            sb.normalize();
            // Compare cross-multiplied to avoid division
            // a.num / a.den < b.num / b.den  <=> a.num * b.den < b.num * a.den
            // Use 128-bit multiplication if possible, else promote.
            bool overflow = internal::would_overflow_mul(sa.num, static_cast<absl::int128>(sb.den)) ||
                internal::would_overflow_mul(sb.num, static_cast<absl::int128>(sa.den));
            if (!overflow) {
                absl::int128 lhs = sa.num * static_cast<absl::int128>(sb.den);
                absl::int128 rhs = sb.num * static_cast<absl::int128>(sa.den);
                return lhs < rhs;
            }
            else {
                // Promote to big integers
                boost::multiprecision::cpp_int n1(sa.num);
                boost::multiprecision::cpp_int d1(sa.den);
                boost::multiprecision::cpp_int n2(sb.num);
                boost::multiprecision::cpp_int d2(sb.den);
                return n1 * d2 < n2 * d1;
            }
        }
        if (ea.is_big() && eb.is_big()) {
            const auto& ba = *ea.as_big();
            const auto& bb = *eb.as_big();
            return ba.num * bb.den < bb.num * ba.den;
        }
        // Mixed small/big: convert small to big
        if (ea.is_small() && eb.is_big()) {
            const auto& sa = *ea.as_small();
            const auto& bb = *eb.as_big();
            sa.normalize();
            boost::multiprecision::cpp_int n1(sa.num);
            boost::multiprecision::cpp_int d1(sa.den);
            return n1 * bb.den < bb.num * d1;
        }
        if (ea.is_big() && eb.is_small()) {
            const auto& ba = *ea.as_big();
            const auto& sb = *eb.as_small();
            sb.normalize();
            boost::multiprecision::cpp_int n2(sb.num);
            boost::multiprecision::cpp_int d2(sb.den);
            return ba.num * d2 < n2 * ba.den;
        }
        return false;
    }

    // -------------------------------------------------------------------------
    // Greater than
    // -------------------------------------------------------------------------
    inline bool operator>(const Rational& a, const Rational& b) {
        return b < a;
    }

    // -------------------------------------------------------------------------
    // Less than or equal
    // -------------------------------------------------------------------------
    inline bool operator<=(const Rational& a, const Rational& b) {
        return !(a > b);
    }

    // -------------------------------------------------------------------------
    // Greater than or equal
    // -------------------------------------------------------------------------
    inline bool operator>=(const Rational& a, const Rational& b) {
        return !(a < b);
    }

} // namespace delta