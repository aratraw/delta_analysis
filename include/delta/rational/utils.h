// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// utils.h
// -----------------------------------------------------------------------------
// UTILITY TYPES – ESPECIALLY ＤＵＭＢ＿ＩＮＴ
// -----------------------------------------------------------------------------
// This tiny file is one of the most important in the entire library.
// It defines `dumb_int` – a `cpp_int` with expression templates disabled
// (`et_off`). This type is used for numerators, denominators, and anywhere
// we need raw integer values without Boost's lazy expression machinery.
//
// -----------------------------------------------------------------------------
// WHY `et_off` IS CRITICAL – THE "SIESTA" PARABLE
// -----------------------------------------------------------------------------
// Boost.Multiprecision, by default, uses expression templates (`et_on`).
// With `et_on`, every arithmetic operation returns a **lazy expression**
// object rather than a concrete number. This is great for optimising chains
// like `a*b + c*d` – but it is **disastrous** when you build your own
// lazy evaluation system on top of it.
//
// Imagine you go to a Spanish bank to make a deposit. You hand over your money,
// but the clerk says: "Siesta – come back later." Then you go to another clerk,
// same answer. Nothing actually gets done. That's `et_on` in our context:
//
//   - Our library already has a lazy layer (LazyRational, canonicalisation,
//     simplification, etc.). We control exactly when evaluation happens.
//   - If `cpp_int` also plays lazy, every `.eval()` or assignment triggers
//     **immediate** construction of Boost's own lazy expression trees,
//     which then get evaluated immediately anyway – but with huge overhead.
//   - Benchmarks show that with `et_on`, all rational arithmetic becomes
//     2–3× slower. Expression templates buy us nothing; they just add
//     indirection, temporary objects, and lambda‑heavy evaluation.
//
// With `et_off`, `cpp_int` behaves as a plain, eager integer type.
// Arithmetic is performed immediately, exactly when we ask for it.
// This matches our own lazy hierarchy perfectly: **we decide when to compute,
// not Boost.**
//
// -----------------------------------------------------------------------------
// WHAT HAPPENS IF YOU CHANGE `et_on`?
// -----------------------------------------------------------------------------
// If you change the definition below to `et_on`, the library will still
// compile and pass almost all tests – but every rational operation will
// become 2–3× slower. Since arithmetic is the backbone of everything,
// the overall slowdown will be catastrophic (5–10× for typical workloads).
//
// Therefore: **NEVER enable expression templates for `dumb_int`.**
// This is not a suggestion – it is a hard requirement.
//
// -----------------------------------------------------------------------------
// USAGE NOTE
// -----------------------------------------------------------------------------
// `dumb_int` is used for:
//   - Numerator and denominator of `Value` (via `numerator()` and `denominator()`)
//   - External interfaces that need to pass integers to/from Rational
//   - Hashing and comparisons where we want to avoid expression template overhead
//   - basically any intermit computations in namespace delta::internal
//
// The helper `dumb_int_to_string` is provided for debugging only. Never use strings in hot workloads, kids. 
// -----------------------------------------------------------------------------

#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/number.hpp>
#include <string>
#include <algorithm>

namespace delta::internal {

    // ----------------------------------------------------------------------------
    // Type `dumb_int`: a `cpp_int` with expression templates OFF.
    // Used for numerators/denominators of Value and in external interfaces.
    // ----------------------------------------------------------------------------
    using dumb_int = boost::multiprecision::number<
        boost::multiprecision::cpp_int_backend<>,
        boost::multiprecision::et_off   // CRITICAL – DO NOT CHANGE TO `et_on`
    >;

    // ----------------------------------------------------------------------------
    // Debug helper – convert a dumb_int to string.
    // May be removed if not used; kept for convenience.
    // ----------------------------------------------------------------------------
    inline std::string dumb_int_to_string(const dumb_int& n) {
        return n.str();
    }

} // namespace delta::internal