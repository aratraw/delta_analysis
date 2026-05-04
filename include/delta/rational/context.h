// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// -----------------------------------------------------------------------------
// DEFAULT EPSILON – THE MASTER KNOB AND SYNTACTIC SUGAR
// -----------------------------------------------------------------------------
// The library provides a global default epsilon that is used by all
// transcendental functions (sqrt, exp, log, sin, cos, acos, asin, atan, tan,
// pow, pi, e) when the epsilon parameter is omitted.
//
// This allows you to write:
//   Rational x = sin(cos(2 * pi()));
// instead of:
//   Rational x = sin(cos(2 * pi(eps), eps), eps);
//
// The default value is 1e-30 (1/10^30). This is a safe balance between
// speed and accuracy for most applications. You can change it globally
// with set_default_eps().
//
// !!! CRITICAL WARNINGS !!!
// 1. NEVER set epsilon to zero. Zero means "infinitely accurate", which
//    leads to infinite loops in series expansions. The library will not
//    protect you from this. Take care that your code does not
//    shoot itself in the foot.
// 2. NEVER allow epsilon to become zero implicitly through improper
//    initialization (e.g., uninitialized Rational, default-constructed
//    Rational(0) without explicit eps).
// 3. DO NOT make the default epsilon thread‑local. The global default
//    is meant to be the same across threads for reproducible results.
//    Changing it in one thread will affect all threads – this is by design.
//    Different eps for different threads is mathematically meaningless
// 4. DO NOT initialize the default epsilon via a lambda that may not
//    be called (static initialization order fiasco). The current direct
//    string literal initialization is safe and reliable.
//
// In short: don't overcomplicate. Use the default as is, or set it once
// at the beginning of your program. Do not try to be clever.
//
// !!! BEST PRACTICE FOR TESTING !!!
// The default epsilon is a GLOBAL parameter. It persists across function
// calls and translation units. If a test changes it (e.g., via set_default_eps()),
// it may affect subsequent tests in unpredictable ways – especially if they
// rely on the original epsilon for convergence.
//
// Therefore, ALWAYS call reset_default_eps() in SetUp and TearDown of your
// tests, EVEN IF the test does not explicitly use or modify the
// default epsilon. This ensures a clean, reproducible state for each test.
//
// If your test appears to be "stuck" – it is NOT stuck. No infinite recursion
// exists in this library. The test is honestly computing something enormous.
// With 80% probability, the default epsilon that entered the test was zero
// or an absurdly small value. Why? Either improper initialization, or mutation
// of the global state without resetting to default 20 tests ago. Print the
// default epsilon that the test received. You will most likely find it to be
// zero or ridiculously tiny. This insight cost a lot of nerves and hours to learn.
// -----------------------------------------------------------------------------
#pragma once

#include "rational_fwd.h"
#include "storage.h"

namespace delta::internal {
    // default eps = 1e-30 = 1/10^30
    inline Value default_eps_value = Value("1/1000000000000000000000000000000");

    inline void reset_default_eps() {
        default_eps_value = Value("1/1000000000000000000000000000000");
    }
} // namespace delta::internal

namespace delta {
    inline Rational default_eps() {
        assert(internal::default_eps_value > 0);
        return Rational(internal::default_eps_value);
    }

    inline void set_default_eps(const Rational& eps) {
        internal::default_eps_value = eps.value();
    }

    inline void reset_default_eps() {
        internal::reset_default_eps();
    }
} // namespace delta