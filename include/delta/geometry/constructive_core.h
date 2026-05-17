// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/geometry/constructive_core.h
// ============================================================================
// constructive_core.h
// Constructive core of Δ‑analysis: points, vectors and operations on them
// ============================================================================
//
// This file defines the fundamental types for constructive description of
// space: Point and Vector, as well as legal operations on them.
// All calculations use delta::Rational – an exact fraction p/q (numerator/denominator)
// that does not depend on any base numeral system.
//
// ----------------------------------------------------------------------------
// 1. Why Rational and why no numeral system is needed?
// ----------------------------------------------------------------------------
// The classic problem of "finite representation of numbers" arises when we try
// to write a number in a fixed numeral system (e.g., decimal): 1/3 = 0.333...
// – an infinite string. However, Rational stores a number as a pair of integers
// (numerator, denominator). This is a finite and exact representation that does
// not depend on the base. Therefore, for us any non‑zero rational number is
// already a constructive address.
//
// ----------------------------------------------------------------------------
// 2. Universal constructive core K*
// ----------------------------------------------------------------------------
// According to section A4e, the universal constructive core K* = Q \ {0} –
// all non‑zero rational numbers. Since we work with Rational as a fraction,
// we automatically rely on K* and need no additional restrictions on the
// denominator (e.g., it is not required to be a product of powers of 2 and 5).
//
// ----------------------------------------------------------------------------
// 3. Point – constructive address
// ----------------------------------------------------------------------------
// A point represents a physical location. To be an address, a point must:
//   • have non‑zero coordinates (otherwise it is "nothing", cannot be pointed to);
//   • coordinates must be non‑zero rational numbers.
// Membership in the core K is checked by the function is_in_K(p).
//
// ----------------------------------------------------------------------------
// 4. Vector – free motion
// ----------------------------------------------------------------------------
// A vector is a displacement, velocity, force. It is not required to be non‑zero
// (the zero vector is allowed – "do nothing"). Vector coordinates can be any
// rational numbers, including zeros. Vectors form a full vector space with
// addition and scalar multiplication.
//
// ----------------------------------------------------------------------------
// 5. Relations between points and vectors
// ----------------------------------------------------------------------------
//   • Difference of two points yields a vector: p - q = v. Always valid.
//   • Sum of a point and a vector yields a new point ONLY if the result belongs
//     to the core K (i.e., all coordinates are non‑zero). Otherwise, the
//     operation returns std::nullopt.
//
// This is a fundamental difference from standard geometry:
//   • In standard geometry, point + vector is always a point.
//   • In Δ‑analysis, we cannot guarantee that addition will not produce a
//     zero coordinate (which is not an address) or an irrational number
//     (which is not Rational). Therefore, the result is optional – the
//     operation succeeds only when the new address remains constructive.
//
// ----------------------------------------------------------------------------
// 6. Why can't we just "do as in ordinary geometry"?
// ----------------------------------------------------------------------------
// The usual approach assumes ℝⁿ is given with all its points, including the
// origin and irrational points. This is convenient for mathematical analysis,
// but contradicts the constructive nature of physical reality:
//   • No real measurement can specify a point with a zero coordinate
//     (that is absence of place).
//   • Irrational coordinates cannot be written as a finite string.
//
// Δ‑analysis eliminates these problems by taking only constructive addresses
// (K*) as fundamental and explicitly checking the admissibility of results
// of operations. Using optional is a direct expression of this principle.
//
// ----------------------------------------------------------------------------
// 7. Usage in the library
// ----------------------------------------------------------------------------
// The Point and Vector types are used in all modules that work with discrete
// operators (gradient, divergence, curl), variational principles, etc.
// Operations on them strictly follow the axioms of Δ‑analysis.
//
// ============================================================================

#pragma once

#include <optional>
#include <set>
#include <cmath>
#include <Eigen/Dense>
#include <boost/multiprecision/cpp_int.hpp>
#include "delta/core/rational.h"
#include "delta/rational/eigen_integration.h"

namespace delta::geometry {

    // -------------------------------------------------------------------------
    // Helper functions for prime factor decomposition
    // -------------------------------------------------------------------------

    namespace detail {

        /**
         * @brief Returns the set of prime factors of a dumb_int.
         * @param n Integer to factorise.
         * @return Set of distinct prime divisors.
         */
        inline std::set<int> prime_factors(const delta::internal::dumb_int& n) {
            std::set<int> factors;
            if (n <= 1) return factors;
            delta::internal::dumb_int m = n;
            if (m % 2 == 0) {
                factors.insert(2);
                while (m % 2 == 0) m /= 2;
            }
            delta::internal::dumb_int p = 3;
            while (p * p <= m) {
                if (m % p == 0) {
                    factors.insert(p.convert_to<int>());
                    while (m % p == 0) m /= p;
                }
                p += 2;
            }
            if (m > 1) factors.insert(m.convert_to<int>());
            return factors;
        }

        /**
         * @brief Returns the set of prime factors of a Rational (must be integer).
         * @param x Rational number (must have denominator 1).
         * @return Set of distinct prime divisors.
         * @throws std::domain_error if x is not an integer.
         */
        inline std::set<int> prime_factors(const Rational& x) {
            if (x.denominator() != 1) {
                throw std::domain_error("prime_factors: argument must be an integer");
            }
            auto num = x.numerator();
            if (num < 0) num = -num;
            delta::internal::dumb_int n = num.convert_to<delta::internal::dumb_int>();
            return prime_factors(n);
        }

        /**
         * @brief Returns numerator and denominator of a Rational as a pair.
         * @param x Rational number.
         * @return Pair (numerator, denominator).
         */
        inline std::pair<Rational, Rational> get_numerator_denominator(const Rational& x) {
            return { x.numerator(), x.denominator() };
        }

    } // namespace detail

    // -------------------------------------------------------------------------
    // Finite base numbers – checks representability in a given base
    // -------------------------------------------------------------------------

    /**
     * @struct FiniteBaseNumbers
     * @brief Checks whether a rational number can be represented with a finite
     *        expansion in a given integer base.
     * @tparam Base The numeral system base (e.g., 2 for binary, 10 for decimal).
     * @note Only meaningful for integer bases.
     */
    template<int Base>
    struct FiniteBaseNumbers {
        /**
         * @brief Returns true iff x can be written as a finite string in base `Base`.
         * @param x Rational number to test.
         * @return true if representable, false otherwise.
         */
        static bool is_representable(const Rational& x) {
            if (x == 0) return false;
            auto [num, den] = detail::get_numerator_denominator(x);
            auto den_factors = detail::prime_factors(den);
            if (den_factors.empty()) return true;
            auto base_factors = detail::prime_factors(static_cast<delta::internal::dumb_int>(Base));
            for (int p : den_factors) {
                if (base_factors.find(p) == base_factors.end()) return false;
            }
            return true;
        }
    };

    // -------------------------------------------------------------------------
    // Universal core K* = Q \ {0}
    // -------------------------------------------------------------------------

    /**
     * @brief Returns true iff x is a non‑zero rational (i.e., belongs to K*).
     * @param x Rational number to test.
     * @return true if x != 0.
     */
    inline bool is_in_universal_core(const Rational& x) {
        return x != 0;
    }

    // -------------------------------------------------------------------------
    // Point – alias for Eigen::Matrix (a constructive address)
    // -------------------------------------------------------------------------

    /**
     * @brief A point in Δ‑analysis: a tuple of coordinates, each being a non‑zero Rational.
     * @tparam Scalar Numeric type (typically delta::Rational).
     * @tparam Dim Dimension of space.
     */
    template<typename Scalar, int Dim>
    using Point = Eigen::Matrix<Scalar, Dim, 1>;

    // -------------------------------------------------------------------------
    // Vector – separate class for free motions
    // -------------------------------------------------------------------------

    /**
     * @class Vector
     * @brief A vector represents displacement, velocity, or force.
     *        Zero coordinates are allowed (unlike points).
     * @tparam Scalar Numeric type (typically delta::Rational).
     * @tparam Dim Dimension of space.
     */
    template<typename Scalar, int Dim>
    class Vector {
        static_assert(Dim > 0, "Dimension must be positive");

    public:
        using vector_type = Eigen::Matrix<Scalar, Dim, 1>;

        /**
         * @brief Default constructor – zero vector.
         */
        Vector() : data_(vector_type::Zero()) {}

        /**
         * @brief Construct from Eigen vector.
         * @param data Eigen vector of dimension Dim.
         */
        Vector(const vector_type& data) : data_(data) {}

        /**
         * @brief Construct from any Eigen expression.
         * @tparam OtherDerived Eigen expression type.
         * @param other Expression convertible to vector_type.
         */
        template<typename OtherDerived>
        explicit Vector(const Eigen::MatrixBase<OtherDerived>& other) : data_(other) {}

        /**
         * @brief Construct from scalar components.
         * @tparam Args Variadic scalar types (must be convertible to Scalar).
         * @param args Exactly Dim scalar arguments.
         * @throws Compile-time error if number of arguments differs from Dim.
         * @example Vector<Rational,3>(1_r, 2_r, 3_r)
         */
        template<typename... Args, typename = std::enable_if_t<(std::is_convertible_v<Args, Scalar> && ...)>>
        explicit Vector(Args... args) {
            static_assert(sizeof...(Args) == Dim, "Wrong number of components");
            vector_type tmp;
            int i = 0;
            ((tmp[i++] = args), ...);
            data_ = tmp;
        }

        /**
         * @brief Access underlying Eigen vector (const).
         */
        const vector_type& data() const { return data_; }

        /**
         * @brief Access component (const).
         * @param i Index (0..Dim-1).
         */
        const Scalar& operator[](int i) const { return data_[i]; }

        /**
         * @brief Access component (mutable).
         * @param i Index (0..Dim-1).
         */
        Scalar& operator[](int i) { return data_[i]; }

        /**
         * @brief Equality comparison.
         */
        bool operator==(const Vector& other) const { return data_ == other.data_; }

        // Convenience accessors for low dimensions
        const Scalar& x() const { static_assert(Dim >= 1, "Dimension too low"); return data_[0]; }
        const Scalar& y() const { static_assert(Dim >= 2, "Dimension too low"); return data_[1]; }
        const Scalar& z() const { static_assert(Dim >= 3, "Dimension too low"); return data_[2]; }

        Scalar& x() { static_assert(Dim >= 1, "Dimension too low"); return data_[0]; }
        Scalar& y() { static_assert(Dim >= 2, "Dimension too low"); return data_[1]; }
        Scalar& z() { static_assert(Dim >= 3, "Dimension too low"); return data_[2]; }

        // -----------------------------------------------------------------
        // Arithmetic operators (component‑wise)
        // -----------------------------------------------------------------

        /**
         * @brief Vector addition.
         */
        Vector operator+(const Vector& other) const { return Vector(data_ + other.data_); }

        /**
         * @brief Vector subtraction.
         */
        Vector operator-(const Vector& other) const { return Vector(data_ - other.data_); }

        /**
         * @brief Scalar multiplication.
         */
        Vector operator*(Scalar s) const { return Vector(data_ * s); }

        /**
         * @brief Scalar division.
         * @throws Division by zero is allowed? Actually Scalar division may throw if Scalar is Rational and denominator=0.
         */
        Vector operator/(Scalar s) const { return Vector(data_ / s); }

        /**
         * @brief Unary minus.
         */
        Vector operator-() const { return Vector(-data_); }

        /**
         * @brief Vector addition assignment.
         */
        Vector& operator+=(const Vector& other) { data_ += other.data_; return *this; }

        /**
         * @brief Vector subtraction assignment.
         */
        Vector& operator-=(const Vector& other) { data_ -= other.data_; return *this; }

        /**
         * @brief Scalar multiplication assignment.
         */
        Vector& operator*=(Scalar s) { data_ *= s; return *this; }

        /**
         * @brief Scalar division assignment.
         */
        Vector& operator/=(Scalar s) { data_ /= s; return *this; }

        // -----------------------------------------------------------------
        // Geometric methods
        // -----------------------------------------------------------------

        /**
         * @brief Dot product.
         * @param other Another vector.
         * @return Scalar dot product.
         */
        Scalar dot(const Vector& other) const { return data_.dot(other.data_); }

        /**
         * @brief Cross product (only for 3D).
         * @param other Another 3D vector.
         * @return Cross product vector.
         * @note Compile-time error if Dim != 3.
         */
        Vector cross(const Vector& other) const {
            static_assert(Dim == 3, "cross only for 3D vectors");
            return Vector(data_.cross(other.data_));
        }

        /**
         * @brief Squared Euclidean norm.
         */
        Scalar squaredNorm() const { return data_.squaredNorm(); }

        /**
         * @brief Euclidean norm.
         */
        Scalar norm() const { return data_.norm(); }

        /**
         * @brief Normalised vector (unit vector).
         * @return Unit vector in the same direction; zero vector if norm is zero.
         */
        Vector normalized() const {
            Scalar n = norm();
            if (n == 0) return *this;
            return *this / n;
        }

    private:
        vector_type data_;   ///< Underlying Eigen vector storage.
    };

    // -------------------------------------------------------------------------
    // Free operators for Vector – commutative scalar multiplication
    // -------------------------------------------------------------------------

    /**
     * @brief Scalar multiplication (commutative).
     * @tparam Scalar Numeric type.
     * @tparam Dim Dimension.
     * @param s Scalar.
     * @param v Vector.
     * @return s * v.
     */
    template<typename Scalar, int Dim>
    Vector<Scalar, Dim> operator*(Scalar s, const Vector<Scalar, Dim>& v) {
        return v * s;
    }

    // -------------------------------------------------------------------------
    // Operations between points and vectors
    // -------------------------------------------------------------------------

    /**
     * @brief Difference of two points yields a vector. Always valid.
     * @tparam Scalar Numeric type.
     * @tparam Dim Dimension.
     * @param a First point.
     * @param b Second point.
     * @return Vector from b to a.
     */
    template<typename Scalar, int Dim,
        std::enable_if_t<(Dim > 0), int> = 0>
        Vector<Scalar, Dim> operator-(const Point<Scalar, Dim>& a, const Point<Scalar, Dim>& b) {
        return Vector<Scalar, Dim>(a.array() - b.array());
    }

    /**
     * @brief Checks whether a point belongs to the constructive core K*
     *        (i.e., all coordinates are non‑zero).
     * @tparam Dim Dimension.
     * @param p Point with Rational coordinates.
     * @return true if all coordinates != 0.
     */
    template<int Dim>
    bool is_in_K(const Eigen::Matrix<Rational, Dim, 1>& p) {
        for (int i = 0; i < Dim; ++i) {
            if (p[i] == 0) return false;
        }
        return true;
    }

    /**
     * @brief Point + Vector.
     * @tparam Scalar Numeric type.
     * @tparam Dim Dimension.
     * @param p Point.
     * @param v Vector.
     * @return New point if result belongs to K*, otherwise std::nullopt.
     * @note The operation is admissible only when all coordinates of the
     *       result are non‑zero (constructive addresses). This is a fundamental
     *       difference from standard geometry – in Δ‑analysis, moving a point
     *       may produce a non‑address (or irrational).
     */
    template<typename Scalar, int Dim>
    std::optional<Point<Scalar, Dim>> operator+(const Point<Scalar, Dim>& p,
        const Vector<Scalar, Dim>& v) {
        Point<Scalar, Dim> new_coords = p + v.data();
        if (is_in_K(new_coords)) {
            return new_coords;
        }
        return std::nullopt;
    }

    /**
     * @brief Vector + Vector (always valid).
     * @tparam Scalar Numeric type.
     * @tparam Dim Dimension.
     * @param u First vector.
     * @param v Second vector.
     * @return Sum vector.
     */
    template<typename Scalar, int Dim>
    Vector<Scalar, Dim> operator+(const Vector<Scalar, Dim>& u, const Vector<Scalar, Dim>& v) {
        return Vector<Scalar, Dim>(u.data() + v.data());
    }

    // -------------------------------------------------------------------------
    // Helper functions for tests (construct points/vectors from literals)
    // -------------------------------------------------------------------------

    /**
     * @brief Construct a point from scalar components (test helper).
     * @tparam Dim Dimension.
     * @tparam Args Variadic scalar types.
     * @param args Exactly Dim scalar arguments.
     * @return Point with given coordinates.
     */
    template<int Dim, typename... Args>
    Point<Rational, Dim> make_point(Args... args) {
        Point<Rational, Dim> p;
        int i = 0;
        ((p[i++] = static_cast<Rational>(args)), ...);
        return p;
    }

    /**
     * @brief Construct a vector from scalar components (test helper).
     * @tparam Dim Dimension.
     * @tparam Args Variadic scalar types.
     * @param args Exactly Dim scalar arguments.
     * @return Vector with given components.
     */
    template<int Dim, typename... Args>
    Vector<Rational, Dim> make_vector(Args... args) {
        return Vector<Rational, Dim>(static_cast<Rational>(args)...);
    }

} // namespace delta::geometry