// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// rational_class.h
// -----------------------------------------------------------------------------
// EAGER RATIONAL NUMBERS WITH ARBITRARY PRECISION
// -----------------------------------------------------------------------------
//
// This header defines the Rational class – an eager, immutable rational number
// type. Under the hood it wraps internal::Value (based on Boost.Multiprecision
// rational_adaptor). All arithmetic operations are performed immediately
// (eagerly) and produce a new Rational object.
//
// Key properties:
//   - Fractions are always stored in normalized form (gcd reduced, denominator > 0).
//   - Copyable and movable (no special move‑only restrictions).
//   - Provides conversions from integers, strings, and cpp_int.
//   - Supports all basic arithmetic (+, -, *, /) and comparisons.
//   - Can be converted to double (lossy) or to string (exact).
//   - Provides interval approximation (double‑based) for fast comparisons.
//   - Integrates with LazyRational via .as_lazy().
//
// Design decisions:
//   - Normalization happens in every constructor that takes numerator/denominator.
//     This ensures that operator== and hashing work correctly.
//   - Arithmetic operators are implemented as non‑member friends for symmetry.
//   - Compound assignments (+=, -=, etc.) modify the left operand in‑place
//     (no copy) – they return a reference to the modified object.
//   - batch_add() provides an efficient way to sum many Rationals using a
//     common denominator technique (reduces intermediate swell).
//   - The class is intentionally simple; all heavy transcendental work is
//     delegated to free functions (sqrt, sin, exp, etc.) defined in
//     transcendentals.h.
//
// -----------------------------------------------------------------------------
// THREAD SAFETY
// -----------------------------------------------------------------------------
// Rational objects are independent and immutable after construction (except
// when modified via compound assignments). No global state is touched,
// so they are thread‑safe for distinct instances.
// -----------------------------------------------------------------------------

#pragma once

#include "rational_fwd.h"
#include "storage.h"

#include <boost/multiprecision/cpp_int.hpp>
#include <string>
#include <vector>

namespace delta {

    class Rational {
    public:
        // ------------------------------------------------------------------------
        // Constructors
        // ------------------------------------------------------------------------

        // Default constructor: 0.
        Rational() noexcept;

        // Integer constructors (signed and unsigned).
        Rational(int num);
        Rational(long long num);
        Rational(unsigned long long num);

        // Constructor from numerator and denominator.
        // Automatically normalises (gcd reduction, makes denominator positive).
        // Throws if denominator == 0.
        explicit Rational(long long num, long long den);

        // Constructors from Boost.Multiprecision cpp_int.
        explicit Rational(const boost::multiprecision::cpp_int& num);
        explicit Rational(const boost::multiprecision::cpp_int& num, const boost::multiprecision::cpp_int& den);

        // Construct from string: accepts "123", "123/456", "0.5", "1.23e-4".
        explicit Rational(const std::string& s);

        // Construct from internal dumb_int.
        explicit Rational(const internal::dumb_int& num);
        explicit Rational(const internal::dumb_int& num, const internal::dumb_int& den);

        // Internal constructor from Value (used by eager transcendental functions).
        explicit Rational(internal::Value val);

        // Copy and move
        Rational(const Rational&);
        Rational(Rational&&) noexcept;
        Rational& operator=(const Rational&);
        Rational& operator=(Rational&&) noexcept;
        ~Rational();

        // ------------------------------------------------------------------------
        // Conversion to LazyRational
        // ------------------------------------------------------------------------
        LazyRational as_lazy() const;

        // ------------------------------------------------------------------------
        // Access to numerator and denominator (normalised)
        // ------------------------------------------------------------------------
        Rational numerator() const;     // returns a Rational with denominator 1
        Rational denominator() const;   // always positive

        // ------------------------------------------------------------------------
        // Conversions to double and string
        // ------------------------------------------------------------------------
        double to_double() const;       // approximate, may lose precision
        std::string to_string() const;  // exact rational representation

        // Interval approximation (double‑based, used for comparisons with LazyRational)
        internal::Interval approx_interval() const;

        // ------------------------------------------------------------------------
        // Access to internal Value (for internal use only)
        // ------------------------------------------------------------------------
        const internal::Value& value() const noexcept { return storage_; }

        // ------------------------------------------------------------------------
        // Arithmetic operators (always eager, return a new Rational)
        // ------------------------------------------------------------------------
        friend Rational operator+(const Rational& a, const Rational& b);
        friend Rational operator-(const Rational& a, const Rational& b);
        friend Rational operator*(const Rational& a, const Rational& b);
        friend Rational operator/(const Rational& a, const Rational& b);
        friend Rational operator-(const Rational& a);   // unary minus

        // Compound assignment operators (modify left operand in‑place)
        friend Rational& operator+=(Rational& a, const Rational& b);
        friend Rational& operator-=(Rational& a, const Rational& b);
        friend Rational& operator*=(Rational& a, const Rational& b);
        friend Rational& operator/=(Rational& a, const Rational& b);

        // ------------------------------------------------------------------------
        // Comparisons
        // ------------------------------------------------------------------------
        friend bool operator==(const Rational& a, const Rational& b);
        friend bool operator!=(const Rational& a, const Rational& b);
        friend bool operator<(const Rational& a, const Rational& b);
        friend bool operator<=(const Rational& a, const Rational& b);
        friend bool operator>(const Rational& a, const Rational& b);
        friend bool operator>=(const Rational& a, const Rational& b);

        // ------------------------------------------------------------------------
        // Batch addition – efficient summation of many Rationals
        // ------------------------------------------------------------------------
        // Uses a common denominator to avoid repeated normalisation.
        // Significantly faster than summing with + in a loop.
        friend Rational batch_add(const std::vector<Rational>& terms);

        // ------------------------------------------------------------------------
        // Absolute value
        // ------------------------------------------------------------------------
        friend Rational abs(const Rational& x);

        // ------------------------------------------------------------------------
        // Template conversion to numeric types (int, long long, double, etc.)
        // ------------------------------------------------------------------------
        // Throws if the conversion is impossible (e.g., not integer, out of range).
        template<typename T>
        T convert_to() const;

    private:
        internal::Value storage_;   // the actual rational data

        // Friends that need access to storage_ for eager transcendentals
        friend Rational eager_sqrt(const Rational& x, const Rational& eps);
        friend Rational eager_exp(const Rational& x, const Rational& eps);
        friend Rational eager_log(const Rational& x, const Rational& eps);
        friend Rational eager_sin(const Rational& x, const Rational& eps);
        friend Rational eager_cos(const Rational& x, const Rational& eps);
        friend Rational eager_acos(const Rational& x, const Rational& eps);
        friend Rational eager_asin(const Rational& x, const Rational& eps);
        friend Rational eager_atan(const Rational& x, const Rational& eps);
        friend Rational eager_tan(const Rational& x, const Rational& eps);
        friend Rational eager_pi(const Rational& eps);
        friend Rational eager_e(const Rational& eps);
        friend Rational eager_pow(const Rational& base, const Rational& exp, const Rational& eps);

        // In‑place operations for optimisation (used internally)
        friend void inplace_add(Rational& a, const Rational& b);
        friend void inplace_mul(Rational& a, const Rational& b);
    };

    // Output stream operator – prints the rational in "a/b" form (or just "a" if denominator == 1).
    std::ostream& operator<<(std::ostream& os, const Rational& r);

} // namespace delta

#include "rational_impl.h"