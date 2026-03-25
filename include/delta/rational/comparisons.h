#pragma once

#include "rational_class.h"
#include "simplify.h"          // for structurally_equal
#include "evaluation_core.h"   // for Value comparison (operator==, operator<)

namespace delta {

    // ----------------------------------------------------------------------------
    //  Equality
    // ----------------------------------------------------------------------------
    inline bool operator==(const Rational& a, const Rational& b) {
        // Same object (including same shared_ptr)
        if (&a == &b) return true;

        // Both lazy: structural comparison
        if (a.is_lazy() && b.is_lazy()) {
            const auto& root_a = *a.as_lazy();
            const auto& root_b = *b.as_lazy();

            // Same shared_ptr? Actually if they point to the same object, the pointers are equal
            if (a.as_lazy() == b.as_lazy()) return true;

            // If hashes match, do full structural equality
            if (root_a.hash() == root_b.hash()) {
                if (internal::structurally_equal(root_a, root_b)) {
                    return true;
                }
            }
            // If hashes differ, we still need to check numeric equality later
        }

        // Interval overlap test
        internal::Interval ia = a.approx_interval();
        internal::Interval ib = b.approx_interval();
        if (!ia.overlaps(ib)) {
            return false; // non‑overlapping intervals → cannot be equal
        }

        // Exact evaluation and comparison
        internal::Value va = a.to_value();
        internal::Value vb = b.to_value();
        return va == vb;
    }

    inline bool operator!=(const Rational& a, const Rational& b) {
        return !(a == b);
    }

    // ----------------------------------------------------------------------------
    //  Less than
    // ----------------------------------------------------------------------------
    inline bool operator<(const Rational& a, const Rational& b) {
        // Same object → false
        if (&a == &b) return false;

        // Interval prediction
        internal::Interval ia = a.approx_interval();
        internal::Interval ib = b.approx_interval();

        if (ia.upper() < ib.lower()) return true;
        if (ia.lower() >= ib.upper()) return false;

        // Overlapping intervals → exact evaluation
        internal::Value va = a.to_value();
        internal::Value vb = b.to_value();
        return va < vb;
    }

    // ----------------------------------------------------------------------------
    //  Derived comparisons
    // ----------------------------------------------------------------------------
    inline bool operator<=(const Rational& a, const Rational& b) {
        return !(b < a);
    }

    inline bool operator>(const Rational& a, const Rational& b) {
        return b < a;
    }

    inline bool operator>=(const Rational& a, const Rational& b) {
        return !(a < b);
    }

} // namespace delta