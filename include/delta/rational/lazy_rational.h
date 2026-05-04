// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// lazy_rational.h

// ---------------------------------------------------------------------------
// LAZYRATIONAL: MUTABLE LAZY EXPRESSION
// ---------------------------------------------------------------------------
//
// LazyRational is a move‑only type representing a lazy evaluation of an
// arithmetic‑transcendental expression over rational numbers.
// Copying is prohibited, moving is allowed. For explicit deep copying,
// use the .clone() method.
//
// ---------------------------------------------------------------------------
// MUTATION PHILOSOPHY: WHY acc + term, NOT acc = acc + term
// ---------------------------------------------------------------------------
//
// Let's be on point. 99% of all computations in any numerical library – ANY –
// boil down to one stupid simple pattern: **accumulating a sum (or product) in a loop**.
//
//   LazyRational acc;                  // CONST(0)
//   for (...) {
//       acc + term;                    // mutates acc, O(1) per iteration
//   }
//   Rational result = acc.eval();      // one evaluation at the end
//
// That's it. Determinant? Matrix multiplication? Series sum? Scalar product?
// Riemann sum for an integral? Finite difference? Iterative approximation?
// ALL OF THEM ARE SUMMATIONS. And when you add Taylor series, iterations,
// nested loops – it's SUMMATION OF SUMMATIONS OF SUMMATIONS OF SUMMATIONS.
//
// This is NOT a niche scenario. This is THE dominant workload of THE computational mathematics.
// Optimize this, and you are the king. Miss it, and your library is just another toy.
//
// Note: it is NOT `acc = acc + term`, just `acc + term`. Assignment is BLOCKED
// at compile time because LazyRational is move-only (copy assignment deleted).
//
// WHY MUTATION?
//   - `acc = acc + term` would copy the ENTIRE tree every iteration → O(N²).
//   - Immutable designs do those copies IMPLICITLY – you can't avoid them.
//   - Here, `acc + term` mutates `acc` in-place in O(1). The full evaluation
//     `acc.eval()` happens ONCE at the end in O(N).
//
// In 99.99999% of real use cases you have EXACTLY ONE LazyRational object per expression.
// All operands are eager values (Rational, int, literals) that get absorbed as leaf values:
//
//   acc + 10_r;     // Rational absorbed as leaf_value
//   acc / 3_r;      // Rational absorbed as leaf_value
//   acc * term_r;   // Rational absorbed as leaf_value
//
// The whole architecture is laser‑focused on this scenario. It does NOT mean it cannot do other scenarios
// like nested transcendentals, cool simplifications and whatnot - YES it can. 
// But if you do the BASE scenario faster - you're king.
// Algebraic simplification is orthogonal and happens later – 
// but the raw accumulation and final SUM calculation MUST be fast.
//
// So underline it: `acc = acc + term;` – THIS IS SHIT CODE. We blocked it.
// Copying is for clones, not for loops.
//
// If you think this is "niche", you have no idea about math. Get real.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HOW OPERATIONS WORK: CHAINS OF MUTATIONS
// ---------------------------------------------------------------------------
//
// Key principle: every mutating operator returns a REFERENCE to its left
// operand. This allows building chains of operations where ALL changes
// are applied to ONE object:
//
//   LazyRational acc;                  // CONST(0)
//   acc + 1_r + 2_r + 3_r + 4_r;      // mutates acc four times in a row!
//
// What happens under the hood:
//
//   1. acc + 1_r:
//      operator+(LazyRational& acc, const Rational& 1_r) is called.
//      Inside: acc.ensure_dirty(), then check the root.
//      If root is SUM, 1_r is added to leaf_values.
//      If root is not SUM, a new SUM node is created, the old root
//      becomes its child, and 1_r goes into leaf_values.
//      Returns acc& (reference to the same acc!).
//      Now acc = SUM(CONST(0), leaf_values=[1_r]).
//
//   2. acc + 2_r:
//      acc is the same object, its root is already SUM.
//      operator+(acc, 2_r) again adds 2_r to leaf_values of the same SUM node.
//      Returns acc&.
//      Now acc = SUM(CONST(0), leaf_values=[1_r, 2_r]).
//
//   3. acc + 3_r:
//      3_r is added to leaf_values.
//      Returns acc&.
//      Now acc = SUM(CONST(0), leaf_values=[1_r, 2_r, 3_r]).
//
//   4. acc + 4_r:
//      4_r is added to leaf_values.
//      Returns acc&.
//      Now acc = SUM(CONST(0), leaf_values=[1_r, 2_r, 3_r, 4_r]).
//
// Result: ONE acc object, ONE SUM node, all summands in leaf_values.
// All four operations have been executed in O(1) each, without a single
// allocation of a new LazyRational and without copying the tree.
//
// Transcendental functions work similarly:
//
//   LazyRational x = LazyRational(0.5_r);
//   Sin(x) + Cos(x.eval() * 2_r) + 1_r;
//
//   1. Sin(x) — creates a copy of x, mutates it into SIN(CONST(0.5)), returns it.
//   2. The obtained SIN is added to Cos(...) — mutates SIN into SUM(SIN, COS).
//   3. 1_r is added to SUM — goes into leaf_values of the same SUM.
//
// Wherever the operator returns a reference to the left operand, the chain
// keeps mutating ONE AND THE SAME object, creating no new ones.
//
// ---------------------------------------------------------------------------
// WHY a + 2 + 3 + 4 + 5 DOES NOT EXPLODE YOUR COMPILER
// ---------------------------------------------------------------------------
// Classic Expression Templates (Eigen, Blaze, Boost.Proto) build a type-level
// tree at compile time. That's great – until your tree depth exceeds ~100 nodes.
// Then the compiler hits its template recursion limit and dies. And if that
// expression lives inside a loop? You're screwed.
//
// LazyRational does something radically different:
//   1. a + 2   – mutates the left operand. No type trees. No templates.
//   2. ... + 3 – adds into the SAME object. No new LazyRational created.
//   3. ... + 4 – same object again. Zero template recursion. Zero compiler pain.
// 
//  Accumulate half a million numbers? Sure, it's tested.
//
// Bottom line: the expression tree lives at runtime, between assignments.
// And we don't need assignments anyway – `acc = acc + term` is BLOCKED
// (move-only semantics).
//
// Go ahead: write `a + 2 + 3 + 4 + 5 + ... + 100500`. The compiler won't blink.
// This is not "wrong". This is optimal for real numerical code.
// ---------------------------------------------------------------------------
// THE PRICE OF MUTATION: WHEN .clone() IS NEEDED
// ---
//
// Mutation means that operators modify their LEFT operand.
// If for any reason you need to use the same LazyRational in several
// sub‑expressions, you MUST explicitly create a copy via .clone():
//
//   LazyRational x = ...;
//   LazyRational expr = Sin(x.clone() * 2_r) + Cos(x.clone() + 1_r);
//
// Without .clone() the compiler may evaluate the sub‑expressions in any order,
// and x will be corrupted by the first mutating operator.
//
// In immutable libraries the same copies occur IMPLICITLY inside
// each operator. Here the user has FULL CONTROL over when to copy,
// and in 99.99999% of cases copies are simply not needed.
//
// ---------------------------------------------------------------------------
// ON MUTATION AND ARITHMETIC: WHAT MUTATES AND WHAT DOES NOT
// ---------------------------------------------------------------------------
//
// Operators with LazyRational are designed so that they always mutate the LEFT
// operand, even if the right operand is also LazyRational. This may be
// non‑obvious in the rare case when you have TWO LazyRationals:
//
//   LazyRational a = ...;
//   LazyRational b = ...;
//   a + b;   // a MUTATES (becomes SUM(a,b)), b is unchanged
//            // (its tree is imported into a via import_tree)
//
// After this operation a contains the sum, while b remains in its original
// state (its tree has been copied inside a).
//
// If you want to keep a and obtain a new LazyRational:
//
//   LazyRational c = a.clone() + b;   // a is untouched, c contains the sum
//
// ---
// SAFE OPERATORS (do NOT mutate the argument)
// ---
//   - Sin(x), Cos(x), Exp(x), Log(x), Sqrt(x), Acos(x) — take const&,
//     internally clone x, mutate the clone and return it.
//   - Unary minus: -x creates a new LazyRational (copies x).
//   - x.clone() — creates an explicit deep copy.
//   - x.eval()  — evaluates x to Rational (O(1) for CONST nodes).
//
// MUTATING OPERATORS (modify the left operand)
// ---
//   - a + b, a - b, a * b, a / b  (all binary arithmetic operations)
//   - a += b, a -= b, a *= b, a /= b
//
// ---------------------------------------------------------------------------

#pragma once

#include "rational_class.h"
#include "lazy_nodes.h"
#include "absl/container/inlined_vector.h"
#include "global_state.h"   // added for register_clean/unregister_clean
#include <vector>
#include <optional>          // for std::optional

#ifdef DELTA_TESTING
namespace delta::testing {
    class LazyRationalTestFixture;   // forward declaration
}
#endif

namespace delta {

    // forward declaration for friend function
    namespace internal {
        class Interval;
        void reset_pool();
        void collect_garbage();
    }
    internal::Interval compute_interval_dirty(const LazyRational& lr);

    class LazyRational {
    public:
        // ------------------------------------------------------------------------
        // Constructors and destructor
        // ------------------------------------------------------------------------
        LazyRational();                                    // dirty CONST(0)
        explicit LazyRational(const Rational& r);          // dirty CONST(r)
        explicit LazyRational(Rational&& r);               // dirty CONST(std::move(r))

        // Copying is prohibited (move‑only)
        LazyRational(const LazyRational&) = delete;
        LazyRational& operator=(const LazyRational&) = delete;

        // Move operations
        LazyRational(LazyRational&& other) noexcept;
        LazyRational& operator=(LazyRational&& other) noexcept;

        ~LazyRational();

        // ------------------------------------------------------------------------
        // Deep copying
        // ------------------------------------------------------------------------
        LazyRational clone() const;

        // ------------------------------------------------------------------------
        // Evaluate to Rational
        // ------------------------------------------------------------------------
        Rational eval(bool skip_simplify = false) const;

        // ------------------------------------------------------------------------
        // In‑place evaluation – turns the object into a clean tree with a single CONST node
        // ------------------------------------------------------------------------
        void eval_inplace(bool skip_simplify = false);

        // ------------------------------------------------------------------------
        // Simplification (canonicalisation in‑place)
        // ------------------------------------------------------------------------
        void simplify_inplace();          // Dirty -> Clean
        LazyRational simplify() const;    // returns a new Clean LazyRational (copying)

        // ------------------------------------------------------------------------
        // Approximate interval (does not require canonicalisation, cached)
        // ------------------------------------------------------------------------
        internal::Interval approx_interval() const;

        // ------------------------------------------------------------------------
        // Force conversion to dirty state
        // ------------------------------------------------------------------------
        void ensure_dirty();

        // ------------------------------------------------------------------------
        // State (for debugging)
        // ------------------------------------------------------------------------
        bool is_dirty() const { return state_ == State::Dirty; }
        bool is_clean() const { return state_ == State::Clean; }

        // ------------------------------------------------------------------------
        // Invalidate cached interval (called on mutations)
        // ------------------------------------------------------------------------
        void invalidate_interval() const { cached_interval_.reset(); }

        // ------------------------------------------------------------------------
        // Bulk insertion methods
        // ------------------------------------------------------------------------
        void append_values(std::vector<internal::Value>&& values);
        void append_nodes(std::vector<int>&& node_indices);

        // ------------------------------------------------------------------------
        // Access to constants (for testing via friends)
        // ------------------------------------------------------------------------
        int add_constant(const internal::Value& v);

        // ------------------------------------------------------------------------
        // Mutating operators (always modify the left operand)
        // ------------------------------------------------------------------------
        // Addition
        friend LazyRational& operator+(LazyRational& a, const LazyRational& b);
        friend LazyRational&& operator+(LazyRational&& a, const LazyRational& b);
        friend LazyRational& operator+(LazyRational& a, const Rational& b);
        friend LazyRational&& operator+(LazyRational&& a, const Rational& b);

        // Subtraction (a - b = a + NEG(b))
        friend LazyRational& operator-(LazyRational& a, const LazyRational& b);
        friend LazyRational&& operator-(LazyRational&& a, const LazyRational& b);
        friend LazyRational& operator-(LazyRational& a, const Rational& b);
        friend LazyRational&& operator-(LazyRational&& a, const Rational& b);

        // Multiplication
        friend LazyRational& operator*(LazyRational& a, const LazyRational& b);
        friend LazyRational&& operator*(LazyRational&& a, const LazyRational& b);
        friend LazyRational& operator*(LazyRational& a, const Rational& b);
        friend LazyRational&& operator*(LazyRational&& a, const Rational& b);

        // Division (a / b = a * RECIP(b))
        friend LazyRational& operator/(LazyRational& a, const LazyRational& b);
        friend LazyRational&& operator/(LazyRational&& a, const LazyRational& b);
        friend LazyRational& operator/(LazyRational& a, const Rational& b);
        friend LazyRational&& operator/(LazyRational&& a, const Rational& b);

        // Unary minus (creates a new LazyRational)
        friend LazyRational operator-(const LazyRational& a);

        // ------------------------------------------------------------------------
        // Friend operators with Rational on the left (Rational + LazyRational etc.)
        // ------------------------------------------------------------------------
        friend LazyRational& operator+(const Rational& a, LazyRational& b);
        friend LazyRational&& operator+(const Rational& a, LazyRational&& b);
        friend LazyRational operator-(const Rational& a, const LazyRational& b);
        friend LazyRational operator-(const Rational& a, LazyRational&& b);
        friend LazyRational& operator*(const Rational& a, LazyRational& b);
        friend LazyRational&& operator*(const Rational& a, LazyRational&& b);
        friend LazyRational operator/(const Rational& a, const LazyRational& b);
        friend LazyRational operator/(const Rational& a, LazyRational&& b);
        friend LazyRational mutating_unary_minus(LazyRational&& a);

        // Compound assignment operators
        friend LazyRational& operator+=(LazyRational& a, const LazyRational& b);
        friend LazyRational& operator+=(LazyRational& a, const Rational& b);
        friend LazyRational& operator-=(LazyRational& a, const LazyRational& b);
        friend LazyRational& operator-=(LazyRational& a, const Rational& b);
        friend LazyRational& operator*=(LazyRational& a, const LazyRational& b);
        friend LazyRational& operator*=(LazyRational& a, const Rational& b);
        friend LazyRational& operator/=(LazyRational& a, const LazyRational& b);
        friend LazyRational& operator/=(LazyRational& a, const Rational& b);

        // ------------------------------------------------------------------------
        // Comparisons (implicitly cause canonicalisation, modify objects)
        // ------------------------------------------------------------------------
        friend bool operator==(const LazyRational& a, const LazyRational& b);
        friend bool operator!=(const LazyRational& a, const LazyRational& b);
        friend bool operator<(const LazyRational& a, const LazyRational& b);
        friend bool operator<=(const LazyRational& a, const LazyRational& b);
        friend bool operator>(const LazyRational& a, const LazyRational& b);
        friend bool operator>=(const LazyRational& a, const LazyRational& b);

        // ------------------------------------------------------------------------
        // Friend functions for lazy transcendentals (all overloads)
        // ------------------------------------------------------------------------
        friend LazyRational lazy_sqrt(const LazyRational& x, const Rational& eps);
        friend LazyRational lazy_sqrt(const Rational& x, const Rational& eps);
        friend LazyRational lazy_exp(const LazyRational& x, const Rational& eps);
        friend LazyRational lazy_exp(const Rational& x, const Rational& eps);
        friend LazyRational lazy_log(const LazyRational& x, const Rational& eps);
        friend LazyRational lazy_log(const Rational& x, const Rational& eps);
        friend LazyRational lazy_sin(const LazyRational& x, const Rational& eps);
        friend LazyRational lazy_sin(const Rational& x, const Rational& eps);
        friend LazyRational lazy_cos(const LazyRational& x, const Rational& eps);
        friend LazyRational lazy_cos(const Rational& x, const Rational& eps);
        friend LazyRational lazy_acos(const LazyRational& x, const Rational& eps);
        friend LazyRational lazy_acos(const Rational& x, const Rational& eps);
        friend LazyRational lazy_pi(const Rational& eps);
        friend LazyRational lazy_e(const Rational& eps);
        friend LazyRational lazy_pow(const LazyRational& base, const LazyRational& exponent, const Rational& eps);
        friend LazyRational lazy_pow(const Rational& base, const LazyRational& exponent, const Rational& eps);
        friend LazyRational lazy_pow(const Rational& base, const Rational& exponent, const Rational& eps);
        friend LazyRational lazy_pow(const LazyRational& base, const Rational& exponent, const Rational& eps);
        friend LazyRational lazy_pow(const LazyRational& base, int exponent);

        // Friend to access dirty tree for interval computation
        friend internal::Interval compute_interval_dirty(const LazyRational& lr);

        friend void internal::reset_pool();
        friend void internal::collect_garbage();

#ifdef DELTA_TESTING
        friend class delta::testing::LazyRationalTestFixture;
#endif

    private:
        enum class State { Dirty, Clean };
        mutable State state_ = State::Dirty;

        // For Dirty (mutable for lazy canonicalisation in const methods)
        mutable std::vector<internal::DirtyNode> nodes_;
        mutable std::vector<internal::Value> constants_;
        mutable int root_ = -1;

        // For Clean:
        mutable int clean_index_ = -1;      // mutable for canonicalisation in const methods

        // Cached interval (nullopt if not computed or tree has changed)
        mutable std::optional<internal::Interval> cached_interval_;

        // ------------------------------------------------------------------------
        // Private methods
        // ------------------------------------------------------------------------
        void canonicalize() const;          // Dirty -> Clean, changes state_ and clean_index_
        int import_tree(const LazyRational& other);
        int new_dirty_node(internal::LazyOp op, absl::InlinedVector<int32_t, 2> children,
            int value_idx = -1, int eps_idx = -1);
        void append_sum_children(int sum_node, const LazyRational& other);
        void append_product_children(int prod_node, const LazyRational& other);

        // ------------------------------------------------------------------------
        // Registration/deregistration in the global clean object registry
        // ------------------------------------------------------------------------
        void register_clean() {
            internal::register_clean(this);
        }
        void unregister_clean() {
            internal::unregister_clean(this);
        }
    };

    std::ostream& operator<<(std::ostream& os, const LazyRational& lr);

} // namespace delta

#include "lazy_rational_impl.h"