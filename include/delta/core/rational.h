// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/core/rational.h
// =========================================================================================================
//   WHY Rational, not double, is the PRIMARY SCALAR TYPE FOR Δ‑ANALYSIS
// =========================================================================================================
//
// In the Δ‑analysis framework, the continuum is not a pre‑existing set of points.
// It emerges as the *limit* of an infinite refinement process: grids are refined
// level by level, and the final continuum objects (ℝⁿ, smooth functions, etc.)
// are invariants of that process.  This design imposes **fundamental requirements**
// on the underlying scalar type that floating‑point doubles simply cannot satisfy.
//
// ---------------------------------------------------------------------------------------------------------
// 1.  UNBOUNDED REFINEMENT – THE KILLER ARGUMENT AGAINST DOUBLE
// ---------------------------------------------------------------------------------------------------------
//
// A typical Δ‑path uses dyadic or barycentric refinement: from a coarse grid,
// each refinement level halves the edge lengths.  After m levels, the smallest
// representable coordinate difference is 2^{-m} (or a similar geometric factor).
//
// A double has only 53 bits of mantissa.  When m exceeds 53, 2^{-m} becomes
// smaller than 1e-16 – **the next refinement step adds points that are
// indistinguishable from existing points** when stored as double.  The refinement
// effectively stops.  The continuum limit is never approached beyond ~50 levels.
//
// But Δ‑analysis *requires* the ability to refine without a built‑in bound.
// The continuum, by definition, is the idealised limit of an infinite process.
// If the implementation forces a hard stop after 50 refinements, the “continuum”
// is merely an illusion created by the finite precision of the arithmetic.
//
// Rational numbers have no such limitation.  A number k/2^m is stored exactly
// as a pair of integers (k, 2^m).  No matter how large m becomes, the value
// remains exact.  Therefore the refinement can continue arbitrarily far,
// and the limit behaviour can be studied correctly.
//
// Moreover, Boost.Multiprecision (the backend of our Rational) recognises
// powers of two and uses **bit shifts** internally for multiplication and
// division by 2^k.  This means that operations like a / 2^m are extremely
// efficient – often as fast as working with integers.  There is no gradual
// loss of performance as the denominator grows, as long as it stays a power
// of two.
//
// ---------------------------------------------------------------------------------------------------------
// 2.  EXACT INVARIANTS ARE THE BACKBONE OF Δ‑ANALYSIS
// ---------------------------------------------------------------------------------------------------------
//
// The framework heavily relies on *exact* algebraic identities on every finite grid:
//   •  d(d(ω)) = 0                  (nilpotence of exterior derivative)
//   •  ∫ (f Δg - g Δf) dV = ∫_∂ (f ∇g - g ∇f)·n dS   (Green’s second identity)
//   •  summation by parts
//   •  curl grad f = 0, div curl v = 0
//   •  consistency under subdivision: e.g. for a 1‑form, ω(e) = ω(e1) + ω(e2)
//      when edge e is split into e1 and e2.
//
// With double arithmetic, none of these identities hold even approximately.
// Rounding errors accumulate and break the exact cancellations that are
// built into the discrete operators.  Consequently, you can never be sure
// whether a failed test indicates a real bug in the algorithm or just
// floating‑point noise.
//
// Rational arithmetic guarantees that the identities hold *exactly* on each
// finite grid (modulo possible overflow, which is avoided by using
// arbitrary‑precision integers).  This makes debugging and verification
// possible, and ensures that the entire mathematical machinery of Δ‑analysis
// is realised faithfully at every finite stage.
//
// ---------------------------------------------------------------------------------------------------------
// 3.  PREDICTABLE COMPARISONS AND TESTING – THE “SPEAKING ERROR” EFFECT
// ---------------------------------------------------------------------------------------------------------
//
// With double, the simple test `EXPECT_EQ(a, b)` is meaningless; you must
// replace it by fuzzy comparisons with an arbitrarily chosen epsilon.
// The choice of epsilon is never rigorous and often masks real errors.
//
// With Rational, `a == b` is a well‑defined, deterministic predicate.
// This enables:
//   •  TDD with strict equality checks.
//   •  Automatic verification of the discrete Green’s identities.
//   •  Detecting unintended modifications of fields during refinement.
//
// But the real power becomes visible when a test *fails*.  Suppose you expect
// a result approximately 1/6, but the code produces a huge irreducible fraction
// like 1/2.  Immediately you know: the discrepancy is not rounding noise,
// it is structural.  The exact value 1/2 tells you that your expectation
// probably omitted a factor 2 somewhere, or that a contribution is counted
// twice.  If the unexpected result equals the sum of two simple fractions
// (e.g. 1/4 + 1/8 = 3/8), you can directly look for the code segment that
// introduces those specific rational numbers (1/4 and 1/8).  The error itself
// points you to the bug.
//
// This “speaking error” property is absent in floating‑point: 0.1666667 vs 0.5
// could be anything – rounding, cancellations, or a real mistake.  You cannot
// reverse‑engineer the cause from the numbers.
//
// ---------------------------------------------------------------------------------------------------------
// 4.  DECOUPLING DIFFERENT SOURCES OF UNCERTAINTY IN REAL‑WORLD MODELS
// ---------------------------------------------------------------------------------------------------------
//
// In any realistic application, multiple error sources coexist:
//   •  Modelling error (e.g. simplified physics)
//   •  Measurement noise (input data)
//   •  Discretisation error (grid, time step)
//   •  Iterative solver tolerance
//   •  Rounding errors (if using double)
//
// With double, all these are tangled together.  You cannot tell whether a
// discrepancy of 1e‑8 comes from the grid being too coarse, from a large
// condition number, or from accumulated rounding.
//
// With Rational (and exact algebraic operations), the only remaining numerical
// approximations are:
//   •  The **truncation error** of transcendental series (sqrt, exp, log, trig),
//      which is controlled by a user‑supplied epsilon.
//   •  The limitations of the discrete model itself (grid refinement level).
//
// All other sources of “noise” are eliminated.  Therefore, when you compare
// simulation results with reference data, any mismatch can be traced back to
// *either* the model inadequacy *or* insufficient refinement – never to
// arithmetic flakiness.  This clean separation is invaluable for calibration,
// validation, and uncertainty quantification.
//
// ---------------------------------------------------------------------------------------------------------
// 5.  SIMPLE INTERACTION WITH THE CONSTRUCTIVE CORE 𝒦
// ---------------------------------------------------------------------------------------------------------
//
// Δ‑analysis explicitly restricts addresses to points whose coordinates are
// *actualisable* (e.g. terminating decimals, dyadic rationals, or more generally
// the universal constructive core 𝒦* = ℚ\{0}).  Rational numbers are the natural
// representation for such points: they can be stored exactly, reduced to lowest
// terms, and tested for membership in the chosen core.
//
// Doubles cannot represent even simple fractions like 1/3 exactly, and they
// cannot distinguish between a genuine zero coordinate (which is excluded from
// 𝒦) from a non‑zero coordinate that became zero due to rounding.  This breaks
// the fundamental ontology of Δ‑analysis.
//
// ---------------------------------------------------------------------------------------------------------
// 6.  PERFORMANCE COMPROMISE – BUT FOR THE RIGHT REASONS
// ---------------------------------------------------------------------------------------------------------
//
// Double is undeniably faster.  However, in Δ‑analysis speed is a secondary
// concern during development and verification.  Once the algorithms are
// debugged and the invariants are proven on rationals, one can optionally
// introduce a template parameter `typename Scalar` and instantiate the
// same code with `double` for large‑scale production runs.  This is a
// **compile‑time decision**, not a philosophical contradiction.
//
// Therefore, the **primary scalar type** of the library is Rational, because
// the library’s raison d’être – the rigorous construction of continuum limits
// from discrete processes – cannot be realised with double.  Floating‑point
// support is a possible optimisation, not the foundation.
//
// ---------------------------------------------------------------------------------------------------------
// 7.  BOTTOM LINE
// ---------------------------------------------------------------------------------------------------------
//
// Double kills the very idea of unbounded refinement, destroys the exact
// algebraic invariants, and forces fuzzy comparisons that make verification
// unreliable.  Its error contamination prevents clean separation of modelling,
// discretisation, and arithmetic uncertainties.  Δ‑analysis without Rational
// is not Δ‑analysis – it is just another finite‑difference library with a
// fancy name.
//
// Hence, **Rational is the targeted, natural, and only defensible scalar type**
// for the core of the Δ‑analysis library.
//
// =========================================================================================================
// =========================================================================================================
//   WHY RATIONAL, NOT double – THE FIELD CLOSURE PRINCIPLE
// =========================================================================================================
//
// A **field** is a set equipped with addition, subtraction, multiplication, and division
// (by non‑zero elements) such that all results stay within the set.  The rational numbers ℚ
// form a field: if a = p/q and b = r/s (with integers p,q,r,s, q,s ≠ 0), then
//   • a ± b = (ps ± rq)/(qs)
//   • a * b = (pr)/(qs)
//   • a / b = (ps)/(qr)  (b ≠ 0)
// are again rational numbers.  Our Rational class stores numerator and denominator
// exactly, using arbitrary‑precision integers.  Therefore every arithmetic operation
// on Rational yields another Rational – **exactly and without approximation**.
//
// Double (binary floating‑point) does **not** have this property.  Its representable
// numbers are a discrete subset of ℚ (numbers of the form m·2^e with a bounded mantissa).
// For example, 0.1 is not exactly representable; neither are 0.2, 0.3, etc.  The sum of
// two representable numbers often falls outside the set.  Hence **double is not even a ring**,
// let alone a field.
//
// ---------------------------------------------------------------------------------------------------------
// CONSEQUENCE: WITHOUT FIELD CLOSURE, THERE IS NO GEOMETRY
// ---------------------------------------------------------------------------------------------------------
//
// Geometry deals with points, vectors, coordinates, and transformations:
//   • Points have coordinates.
//   • Vectors are added, subtracted, scaled.
//   • Coordinates are added to vectors to obtain new points.
//   • Lengths and inner products involve squaring and summing coordinates.
//
// All these operations require closure of the underlying numeric type:
//   • If you add two coordinates, you must get a coordinate.
//   • If you multiply a coordinate by a scalar, you must get a coordinate.
//   • If you compute a squared distance (x₁−x₂)² + (y₁−y₂)², the result must be
//     a valid element of the field.
//
// With double, this fails already at the first step: the sum of two coordinates that
// are exactly representable may not be representable, forcing rounding.  The rounding
// errors accumulate, break exact algebraic identities (e.g. the parallelogram law,
// the Pythagorean theorem), and ultimately destroy any hope of a consistent geometric
// model.  You cannot speak of a “vector space” over a set that is not closed under
// addition and scalar multiplication.  You cannot define a metric that respects the
// field structure.  In short, **double does not support geometry** – it only supports
// approximate, error‑prone simulations that happen to be “close enough” for some
// engineering purposes.
//
// Δ‑analysis demands a rigorous geometric foundation: points, vectors, and coordinates
// must belong to a field (or at least a ring) that is closed under all necessary
// operations.  Rational provides exactly that.  Double does not, and no amount of
// rounding or epsilon tuning can fix this fundamental deficiency.
//
// Therefore, **Rational is the only logical choice** for the scalar type in a
// library that aims to implement a genuine geometric system.
//
// =========================================================================================================
// =========================================================================================================
//   OBJECTION: “transcendental functions break field closure – you cannot have exact √2”
// =========================================================================================================
//
// The criticism: “Rational is a field, but you introduce sqrt, sin, exp with tolerance ε.
// This loses exactness – you cannot compute √2 exactly.  Geometry without exact diagonals
// is meaningless.  So you are no better than double.”
//
// This objection presupposes the existence of √2 as a *completed object* that must be
// represented.  In Δ‑analysis we reject that presupposition.
//
// =========================================================================================================
//   THERE IS NO “TRUE VALUE” OF AN IRRATIONAL NUMBER
// =========================================================================================================
//
// An irrational number (√2, π, e, …) does **not** exist as an independent object in
// the constructive universe.  What exists are:
//   • A rule that defines a Cauchy sequence of rational numbers.
//   • At each finite stage, a concrete rational number that approximates according
//     to that rule.
//   • The limit (the “true” irrational) is a regulative idea, not a constructible point.
//
// Therefore, when we compute `sqrt(2, eps)`, we are **not** approximating some
// pre‑existing √2 that lives outside the rationals.  Instead, we are executing the
// rule “produce a rational number r such that r^2 is within eps of 2”.  The result r
// is the only meaningful object; there is no hidden “true” value behind it.
//
// ---------------------------------------------------------------------------------------------------------
//   CONSEQUENCE: THE FIELD ℚ IS CONSTRUCTIVELY CLOSED
// ---------------------------------------------------------------------------------------------------------
//
// For any rational inputs, the operations +, −, ×, / produce rational outputs exactly.
// For transcendental operations, the output is **by definition** a rational number
// (computed by series, binary splitting, etc.) that is guaranteed to satisfy the
// requested tolerance.  There is no claim that the output equals an abstract
// irrational object; there is only the rational output itself.
//
// Thus, the field ℚ is *constructively closed* under all operations we define:
// the result always lands in ℚ.  The concept of “error relative to a true value”
// is a convenient way to reason about the coherence of sequences, but it does
// not introduce any non‑rational entity into the computational substrate.
//
// ---------------------------------------------------------------------------------------------------------
//   WHY THIS IS FUNDAMENTALLY DIFFERENT FROM double
// ---------------------------------------------------------------------------------------------------------
//
// Double pretends to represent √2 as a fixed binary fraction (≈1.4142135623730951)
// and implicitly assumes that this is an “approximation” to a pre‑existing real
// number.  The error is hard‑coded, cannot be refined without changing the data type,
// and the operations (+,-,*,/) on double are not even exact for rationals.
//
// With Rational, `sqrt(2, 1e-6)`, `sqrt(2, 1e-12)`, `sqrt(2, 1e-30)` produce
// different rational numbers.  The sequence is under our control, and the limit
// (the regulative idea) is never mistaken for an actual object.  The arithmetic
// operations stay exact, and the transcendental operations produce rational
// results that belong to the same field ℚ.  No foreign “real” numbers ever enter
// the system.
//
// ---------------------------------------------------------------------------------------------------------
//   GEOMETRY WITHOUT “TRUE” IRRATIONAL LENGTHS
// ---------------------------------------------------------------------------------------------------------
//
// The objection that “geometry without exact diagonals is meaningless” implicitly
// assumes that a perfect square with side length 1 exists in reality and that its
// diagonal must have length √2 as an element of a pre‑existing continuum.  Neither
// holds in a constructive framework.
//
//   • Any physical square is made of finitely many elementary units (atoms, cells).
//     Its side is a rational number given by a measurement with finite precision.
//   • The diagonal is a rational length (by the Pythagorean theorem applied to
//     rational sides) whose square may not be exactly 2; but we can refine the
//     measurement (or the conceptual construction) to make it as close to 2 as desired.
//   • The notion of an “ideal square” with exactly rational sides and exactly
//     irrational diagonal is a mathematical fantasy – useful for reasoning, but
//     not a constructible reality.
//
// Δ‑analysis embraces this: geometry is the study of rational approximations and
// their limits.  The field ℚ, together with parametrically accurate transcendental
// functions, provides all the necessary constructive power.
//
// ---------------------------------------------------------------------------------------------------------
//   CONCLUSION: CONSTRUCTIVE CLOSURE IS THE ONLY RELEVANT CLOSURE
// ---------------------------------------------------------------------------------------------------------
//
// The demand that a field be closed under “taking √2” is the demand for algebraic
// closure – which ℚ does not have, and which is irrelevant for computational geometry.
// What matters is that every operation defined on rationals yields a rational result.
// That holds for +, -, ×, / exactly, and for transcendental functions with a tolerance
// parameter.  The tolerance parameter does not introduce irrational objects; it only
// quantifies the refinement level of the constructive process.
//
// Therefore, the alleged contradiction disappears.  Rational is not an approximation
// of an ideal continuum; it is the genuine constructive field.  Double, on the other
// hand, fails even at exact addition of simple rationals and embeds a false belief
// in the existence of “true” real constants.  The choice of Rational as the primary
// scalar type is not a compromise – it is the only coherent choice for a library
// that takes constructivity seriously.
//
// =========================================================================================================
#pragma once

// Main Rational and LazyRational classses and implementation
#include "delta/rational/rational_class.h"
#include "delta/rational/lazy_rational.h"
// Custom Literal (_r)
#include "delta/rational/literals.h"

// Transcendental functions (sqrt, exp, log, sin, cos, acos, pi, e, pow)
#include "delta/rational/transcendentals.h"
#include "delta/rational/context.h"
// Eigen Integration
#include "delta/rational/eigen_integration.h"


// =========================================================================================================
//                    COMPREHENSIVE TECHNICAL REFERENCE – delta::rational
// =========================================================================================================
//
// This header (`delta/core/rational.h`) unifies the entire rational computing sub‑library.
// Treating it as a black box, the rest of the project can use arbitrary‑precision rational
// arithmetic, lazy expression trees, and transcendental functions without delving into
// internal details.  However, to use the sub‑library efficiently and correctly you must
// understand its dual eager/lazy architecture, the design rationale behind move‑only
// LazyRational, and the performance implications of different usage patterns.
//
// ---------------------------------------------------------------------------------------------------------
// 1.  HIGH‑LEVEL ARCHITECTURE
// ---------------------------------------------------------------------------------------------------------
//
// The library consists of two main public classes:
//
//   • Rational  – a strictly *eager*, arbitrary‑precision rational number
//                 (backed by Boost.Multiprecision cpp_int).
//
//   • LazyRational – a *lazy*, move‑only expression graph that accumulates
//                    operations (arithmetic + transcendental) and is evaluated
//                    once, potentially after high‑level algebraic simplification.
//
// Eager functions (`sqrt`, `exp`, `log`, …) return Rational and compute
// immediately using series expansions (or a fast float‑fallback for modest
// precision).  Lazy versions (`Sqrt`, `Exp`, …) build a graph node and defer
// computation to evaluation time, enabling symbolic simplifications.
//
// All heavy internal machinery (node pool, caches, series implementations) is
// hidden in the `delta::internal` namespace, so regular users only interact
// with `Rational`, `LazyRational`, the free functions and the literal suffix.
//
// ---------------------------------------------------------------------------------------------------------
// 2.  CORE TYPE: Rational
// ---------------------------------------------------------------------------------------------------------
//
// #include <delta/rational/rational_class.h>
//
// Construction:
//   Rational()             // 0
//   Rational(int)
//   Rational(long long)
//   Rational(unsigned long long)
//   Rational(cpp_int)      // Boost cpp_int
//   Rational(cpp_int num, cpp_int den)
//   Rational(std::string)  // "3.14", "22/7", "123456789"
//   Rational(Value)        // internal (for interop)
//
//   Literals (literals.h):
//       1_r                 // Rational(unsigned long long)
//       "3.14"_r            // Rational(const char*)
//       template<char...>   // compile‑time floating literal (if supported)
//
// Access:
//   .value()          → const internal::Value&   (raw backend, read‑only)
//   .numerator()      → Rational (the numerator part)
//   .denominator()    → Rational (the denominator part)
//   .to_double()      → double   (for quick approx)
//   .to_string()      → std::string
//   .as_lazy()        → LazyRational  (wraps constant into a lazy tree)
//   .approx_interval()→ Interval  (interval [value, value])
//
// Arithmetic:
//   +, -, *, / (binary, always eager, return new Rational)
//   +=, -=, *=, /= (also eager)
//   - (unary)
//   batch_add(vector<Rational>)  (LCM‑based summation, often faster than + in loop)
//   abs(Rational)
//
// Comparisons:
//   ==, !=, <, <=, >, >=  (with Rational; also with LazyRational via implicit eval)
//
// ---------------------------------------------------------------------------------------------------------
// 3.  CORE TYPE: LazyRational  (move‑only, mutable expression tree)
// ---------------------------------------------------------------------------------------------------------
//
// #include <delta/rational/lazy_rational.h>
// Full implementation in lazy_rational_impl.h.
//
// ---- Construction & state ----
//
//   LazyRational()           // dirty CONST(0)
//   LazyRational(const Rational&)
//   LazyRational(Rational&&)
//   // Copy is **deleted**.  Use .clone() for explicit deep copies.
//   // Move is allowed.
//
//   State:
//     .is_dirty() / .is_clean()
//     Dirty – flat list of nodes, can be mutated by operators.
//     Clean – node lives in the global NodePool, referenced by clean_index_.
//     Clean objects are automatically registered in a global set and participate in GC.
//
// ---- Key modifications (always change *this, even for LHS lvalue in binary ops) ----
//
//   Arithmetics with LazyRational& lhs, const Rational& / const LazyRational& rhs:
//       a + b     // **a is mutated**. If a is not SUM, it becomes SUM(a, b).
//                 // If a is already SUM, b's subtree is appended in O(1).
//       a - b     // implemented as a + NEG(b)
//       a * b     // similar, with PRODUCT
//       a / b     // a * RECIP(b)
//       a += b, a -= b, a *= b, a /= b   // same as above, return a&
//       -a        // **creates new LazyRational** (unary minus)
//       +, -, *, / with Rational on LHS also provided,
//               e.g. (const Rational&) + (LazyRational&) → b += a
//
//   Bulk append (for performance in loops):
//       .append_values(vector<Value>&&)    // push many leaf values into a SUM node
//       .append_nodes(vector<int>&&)       // push many child indices
//
// ---- Evaluation / simplification ----
//
//   .eval(bool skip_simplify = false) → Rational
//         If clean and root is CONST, returns constant. Otherwise
//         canonicalizes (dirty→clean) and then evaluates the clean DAG.
//         skip_simplify skips canonicalization and runs direct dirty evaluation
//         (faster if you don’t need symbolic optimizations).
//
//   .eval_inplace(bool skip_simplify = false)
//         Destroys the tree and replaces *this with a clean CONST node
//         holding the result.  Useful for one‑shot computations.
//
//   .simplify_inplace()   → forces canonicalization, object stays clean.
//   .simplify()           → returns a new clean LazyRational (cloning first).
//
// ---- Cloning ----
//   .clone() → LazyRational   // deep copy.  If clean, just increments refcount.
//
// ---- ensure_dirty() ----
//   If clean, materialises the DAG into a private dirty vector, removes from
//   clean registry.  All mutating operators call this first.
//
// ---- Interval approximation ----
//   .approx_interval() → Interval  (cached, recomputed on mutation)
//       Provides outward‑rounded double bounds, used for fast comparisons.
//
// ---- Import / append helpers (internal, but useful to know) ----
//   .import_tree(const LazyRational&)   // copy subtree into *this private nodes
//   .append_sum_children / append_product_children
//   .add_constant(const Value&) → idx
//   .new_dirty_node(...) → idx
//
// ---- Ownership & GC ----
//   Clean LazyRational objects are tracked in a global thread‑local set.
//   Decrementing refcounts and GC are automatic, but you can manually force
//   garbage collect via internal::force_garbage_collect() or reset_pool().
//   The pool size can be limited by internal::set_pool_max_size(size_t).
//   Default max_size = 1'000'000 nodes.  If exhausted, exception thrown.
//
// Design philosophy (CRUCIAL):
//   NEVER write `acc = acc + term;` (copy deleted).  Instead just `acc + term;`
//   Mutations accumulate O(1) per addition, and a single .eval() is O(N).
//   This is the primary performance advantage over immutable libraries.
//
//   When you need the same LazyRational in several branches, use .clone():
//      auto x = ...;
//      auto expr = Sin(x.clone() * 2_r) + Cos(x.clone() + 1_r);
//   Without .clone(), operators mutate x and you get undefined order effects.
//
//   Transcendental functions (Sin, Cos, Exp, …) **clone internally**,
//   so they never mutate their argument.
//
// ---------------------------------------------------------------------------------------------------------
// 4.  TRANSCENDENTAL FUNCTIONS
// ---------------------------------------------------------------------------------------------------------
//
// All functions take an optional `eps` parameter (Rational, default = 1e-30).
// eps specifies the *absolute* error bound: |true_value - result| ≤ eps.
// For extremely large values (exp(1000)) this may require enormous work;
// consider relative checking in user code if needed.
//
// ---- Eager (returns Rational) ----
//   sqrt(x, eps)           exp(x, eps)           log(x, eps)
//   sin(x, eps)            cos(x, eps)            acos(x, eps)
//   asin(x, eps)           atan(x, eps)           tan(x, eps)
//   pi(eps)                e(eps)
//   pow(base, exp, eps)    // rational exponent
//   pow(base, int)         // integer exponent (fast binary exponentiation)
//
// ---- Lazy (returns LazyRational) ----
//   Sqrt(x, eps)           Exp(x, eps)            Log(x, eps)
//   Sin(x, eps)            Cos(x, eps)             Acos(x, eps)
//   Pi(eps)                E(eps)
//   Pow(base, exp, eps)    // multiple overloads for combinations of
//                           // LazyRational / Rational / int.
//   lazy_sqrt, lazy_exp, … (same as above but explicit naming).
//
//   There are also lazy variants for asin, atan, tan in the code but currently
//   commented out (not yet implemented).  The eager versions are usable.
//
//   All lazy functions accept both LazyRational and Rational arguments.
//   They **do not mutate** the argument (they clone it).
//
// ---- Context (global epsilon) ----
//   delta::default_eps()          → Rational (1e-30 by default)
//   delta::set_default_eps(eps)   → change thread‑local default epsilon
//   delta::reset_default_eps()    → restore 1e-30
//   The default epsilon is used when the optional eps argument is omitted.
//   Internally stored as internal::Value in thread‑local variable.
//
// ---------------------------------------------------------------------------------------------------------
// 5.  UNDER THE HOOD: EVALUATION, SERIES, AND SIMPLIFICATION
// ---------------------------------------------------------------------------------------------------------
// (Not needed for casual use, but important for understanding performance.)
//
// ---- Node types (node_types.h, lazy_nodes.h) ----
//   LazyOp enum: CONST, SUM, PRODUCT, NEG, RECIP, SQRT, EXP, LOG, SIN, COS,
//                ACOS, PI, E, POW.
//   DirtyNode, TempNode, Node – internal representations with children, leaf_values,
//   eps_idx, etc.
//
// ---- Evaluation (evaluate_impl.h) ----
//   Template function evaluate_tree traverses a vector of nodes in post‑order,
//   computes each node via the corresponding eager_* function.
//   Summation uses a pyramidal compact reduction (PCR) to minimize intermediate
//   growth.  There is an in‑place strategy for dirty evaluation and a copy strategy
//   for clean evaluation.
//
//   `evaluate(clean_index)` and `evaluate_dirty(nodes)` are the main entry points.
//
// ---- Series implementations (evaluation_core.h) ----
//   - series sqrt: Newton's method with argument scaling.
//   - series exp:  Taylor series with argument reduction (divide by 2^k) and
//                  fast binary squaring, rigorous eps scaling.
//   - series log:  range reduction to [1/2,2] using ln2, atanh series.
//   - sin/cos:     binary splitting of Maclaurin series, exact π reduction.
//   - arctan, arsin, arccos: binary splitting, reduction formulas.
//   - pi:          Chudnovsky algorithm with binary splitting, cached per eps.
//   - e:           simple series sum(1/n!).
//   - tan:         sin/cos.
//
//   HYBRID_THRESHOLD = 1e-35: for eps ≥ 1e-35, float‑paths using cpp_dec_float_100
//   are used for sin, cos, exp, acos, pi, asin, atan, tan (not for sqrt, log, e).
//   This speeds up moderate precision without sacrificing correctness.
//
// ---- Symbolic simplification (simplify_impl.h) ----
//   When a LazyRational is canonicalized (dirty→clean), the tree is first converted
//   to TempNode and then `simplify_tree` applies:
//     - Flattening of nested SUM/PRODUCT.
//     - Removal of 0 in SUM, 1 in PRODUCT.
//     - Grouping identical scalars into multiplications.
//     - Collapsing identical sub‑trees (A+A → 2*A, A*A → A^2).
//     - Distribution: a*b + a*c → a*(b+c) (if products exist).
//     - Cancellation: x + NEG(x) → 0, x * RECIP(x) → 1.
//     - Inverse functions: NEG(NEG(x)) → x, EXP(LOG(x)) → x, etc.
//     - POW simplifications: 0^positive → 0, x^0 → 1, (x^a)^b → x^(a*b) for int exponents.
//   The simplification is *constructive* (builds new nodes, not numeric evaluation),
//   preserving the symbolic nature as much as possible.
//
// ---- NodePool & Garbage Collection (node_pool.h, global_state.h) ----
//   The global pool holds unique clean nodes (CONST, SUM, PRODUCT, unary).
//   Refcounting manages sharing; when a LazyRational is destroyed it decrements
//   the root refcount.  Nodes with refcount 0 are not immediately freed;
//   `collect_garbage()` compacts the pool by evaluating all live roots to CONST
//   and replacing them.  This is triggered automatically when the pool occupancy
//   exceeds gc_threshold (0.9 * max_size).  During canonicalization, if space is
//   insufficient GC is attempted; if still insufficient, a CanonicalizeGuard
//   temporarily expands the pool (or throws).
//
//   A global registry of all clean LazyRational objects is maintained so that
//   GC knows all roots.  `reset_pool()` completely resets the pool and
//   invalidates all existing clean LazyRationals (turning them into dirty zero).
//
// ---- Interval Arithmetic (interval.h) ----
//   Interval is a simple double‑based outward‑rounded interval class.
//   Used in comparisons (==, <, etc.) for quick reject before full evaluation.
//   Not exposed directly but LazyRational::approx_interval() returns it.
//
// ---------------------------------------------------------------------------------------------------------
// 6.  PERFORMANCE GUIDELINES
// ---------------------------------------------------------------------------------------------------------
// 1. Use eager arithmetic (`Rational + Rational` etc.) when you need a concrete
//    result and no further symbolic manipulation.
// 2. For accumulating sums or products in a loop, ALWAYS use the mutable
//    LazyRational pattern:  `LazyRational acc; for(...) acc + term;  acc.eval();`
//    This is O(N) vs O(N²) for repeated eager additions.
// 3. `batch_add(vector<Rational>)` is even faster for homogeneous numerator bunch.
// 4. When passing sub‑expressions to transcendental functions, decide:
//      - Simple constants → evaluate them first (use Rational arithmetics or eval()).
//      - Complex expressions that may cancel → keep lazy, using .clone() to avoid
//        mutating the original LazyRational.
//    Example: `Sin( (x.clone() * 2_r + 1_r) )`  – all of x*2+1 is kept lazy.
// 5. `eval()` on a CONST node is O(1); on a clean node it traverses the DAG.
//    If you will evaluate many times, consider calling .eval_inplace() to replace
//    the object with the constant result.
// 6. The canonicalization (automatic before evaluation) is the most expensive step.
//    If you are sure no simplification is needed, use `skip_simplify = true` in
//    eval()/eval_inplace().
// 7. The default epsilon 1e-30 is very strict.  For many practical applications
//    you can relax it to e.g. 1e-12 → drastically faster series computations.
//    Use `set_default_eps` or pass eps explicitly.
// 8. For huge arguments to exp (>> 20), the series path does aggressive argument
//    reduction and internal eps scaling; it is correct but may be slow.
//
// ---------------------------------------------------------------------------------------------------------
// 7.  INTEGRATION WITH Eigen
// ---------------------------------------------------------------------------------------------------------
//
// #include <delta/rational/eigen_integration.h>  (already included in this header)
//
// This provides Eigen::NumTraits<delta::Rational> so that Rational can be used
// as a scalar type in Eigen matrices.  Key points:
//   - epsilon() returns delta::default_eps()
//   - dummy_precision() returns delta::default_eps()
//   - A specialization of Eigen::internal::sqrt_impl for Rational calls delta::sqrt(x).
//   - AddCost, MulCost are set to 1 (not fully accurate but safe).
//
// You can write:
//   Eigen::Matrix<delta::Rational, 3, 3> M;
//   M << 1_r, 2_r, 3_r, ... ;
//   auto P = M.inverse();   // uses rational arithmetic, exact up to epsilon
//
// ---------------------------------------------------------------------------------------------------------
// 8.  THREAD SAFETY
// ---------------------------------------------------------------------------------------------------------
// All global state (NodePool, pi_cache, clean_registry, default eps, gc_disabled)
// is thread‑local (`thread_local`).  You may freely use different threads.
// However, LazyRational objects themselves are not synchronised – a single
// object must not be modified from multiple threads concurrently.
//
// ---------------------------------------------------------------------------------------------------------
// 9.  KNOWN LIMITATIONS & FUTURE WORK
// ---------------------------------------------------------------------------------------------------------
// - Lazy versions of `asin`, `atan`, `tan` are not yet exposed (enums exist
//   but the lazy node creation is commented out). Eager versions work fine.
// - No direct support for complex numbers or matrices beyond Eigen integration.
// - No serialisation of Rational or LazyRational.
// - The node pool can grow indefinitely if you create many long‑lived clean
//   objects; the automatic GC compacts but doesn’t shrink the vector.
// - Very large rational numbers (thousands of digits) naturally become slow;
//   there are no asymptotic optimisations beyond what Boost provides.
// - The `convert_to<T>` method for Rational supports `int`, `long long`, `dumb_int`,
//   `double` – no `float` directly.
//
// ---------------------------------------------------------------------------------------------------------
// 10. QUICK REFERENCE – COMMON USAGE PATTERNS
// ---------------------------------------------------------------------------------------------------------
//
// (A) Eager one‑shot computation:
//     Rational a = "1.5"_r;
//     Rational b = 2_r;
//     Rational c = sqrt(a * a + b * b);         // c ≈ 2.5 (exact rational sqrt with eps)
//
// (B) Accumulate in a loop with LazyRational:
//     LazyRational acc;
//     for (int i = 0; i < N; ++i) {
//         acc + sin(Rational(i+1));              // each sin is eager, added to lazy SUM
//     }
//     Rational total = acc.eval();              // single evaluation
//
// (C) Building a complex expression tree:
//     LazyRational x = LazyRational("0.5"_r);
//     auto expr = Sin( x.clone() * 2_r + 1_r )   // x*2+1 kept lazy
//                + Cos( x.eval() * 3_r );        // evaluete x to Rational then lazy Cos
//     Rational res = expr.eval();                // simplify + evaluate
//
// (D) Changing default epsilon for whole thread:
//     set_default_eps("1/100000000000000000000000"_r);
//     // all subsequent calls to sqrt, sin, etc. without explicit eps use this truncation epsilon.
//     reset_default_eps();
//
// (E) Using with Eigen:
//     #include <delta/core/rational.h>
//     Eigen::Matrix<Rational, 2, 2> M;
//     M(0,0) = "1/2"_r; M(0,1) = "1/3"_r;
//     M(1,0) = "1/4"_r; M(1,1) = "1/5"_r;
//     auto Minv = M.inverse();   // exact rational inverse (with approximations for sqrt)
//
// =========================================================================================================
//                           END OF TECHNICAL REFERENCE
// =========================================================================================================

