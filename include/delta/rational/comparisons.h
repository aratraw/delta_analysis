// comparisons.h
#pragma once

#include "rational_class.h"
#include "expression_root.h"   // для ExpressionRoot
#include "simplify.h"          // для internal::structurally_equal
#include "evaluation_core.h"   // для сравнения Value

namespace delta {

    // ----------------------------------------------------------------------------
    //  Equality
    // ----------------------------------------------------------------------------
    inline bool operator==(const Rational& a, const Rational& b) {
        // Same object
        if (&a == &b) return true;

        // Both lazy: structural comparison
        if (a.is_lazy() && b.is_lazy()) {
            int idx_a = a.root_index();
            int idx_b = b.root_index();

            // Same root index? (было сравнение shared_ptr)
            if (idx_a == idx_b) return true;

            ExpressionRoot root_a(idx_a);
            ExpressionRoot root_b(idx_b);

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
            return false;
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