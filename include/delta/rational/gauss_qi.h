// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0
/**
 * RATIONAL_INFINITY_MANIFESTO
 * @brief Theoretical Framework: The Convergence of Complex Algebra and Infinite-Dimensional Rationality.
 *
 * ======================================================================================================
 * THE RATIONAL CAPTURE THEORY: COMPLEXITY AS AN EMERGENT PROPERTY OF INFINITE DIMENSION
 * ======================================================================================================
 *
 * 1. THE ORTHOGONALITY POSTULATE
 * Traditionally, the imaginary unit (i) is treated as an axiomatic extension to the real field,
 * defined by i² = -1. Geometrically, (i) is viewed as an axis orthogonal to the 1D real line.
 * As we expand the real space to R^N, the complex plane is often treated as an external "extra"
 * dimension that remains transcendentally orthogonal to the entire N-dimensional manifold.
 *
 * 2. THE N -> INF LIMIT AND AXIAL ABSORPTION
 * The core of this research proposes that as N (the number of rational dimensions) approaches infinity,
 * the "External Orthogonality" of the complex plane is annihilated. In the limit Q^inf, the space
 * becomes "dense with directions." Every possible orthogonal orientation—including that which we
 * perceive as the imaginary axis—is already contained within the infinite set of rational basis vectors.
 *
 * CONSEQUENCE: The imaginary unit is no longer a "mythical" square root of a negative number.
 * Instead, it is a structural identifier for a specific rotational mapping between two
 * rational axes within the Q^inf Hilbert-like space.
 *
 * 3. FROM ALGEBRAIC IMAGINATION TO GEOMETRIC DETERMINISM
 * This framework shifts the definition of Complex Numbers from "Algebraic Extensions" to
 * "Rational Projections":
 *    - THE CAPTURE: At N=inf, the imaginary axis is "captured" by the rational manifold.
 *    - THE MAPPING: A GaussQi number (a + bi) is reinterpreted as a 2D projection of an
 *      infinite-dimensional rational vector where 'i' is simply a coordinate shift operator (e_n -> e_{n+1}).
 *    - THE OPERATOR: Multiplication by (i) is no longer a symbolic operation but a
 *      topological permutation of rational data across the infinite basis.
 *
 * 4. NEW MATHEMATICAL HORIZONS: TRANSCENDENTAL RATIONALITY
 * By embedding GaussQi into Q^inf, we challenge the necessity of Real Analysis (R).
 * If the complex plane is merely a slice of the infinite rational web, then:
 *    - Transcendental functions (exp, log, sin) are not limits of real numbers but
 *      discrete combinatorial paths through the infinite-dimensional rational lattice.
 *    - Precision becomes absolute: Since we remain within the rational field (Q),
 *      the "errors" in complex transcendental approximations are merely artifacts
 *      of truncating the infinite-dimensional vector.
 *
 * 5. RESEARCH OBJECTIVES
 * - Mapping the Gaussian field Q(i) onto the regular representation of infinite-dimensional shift matrices.
 * - Developing "Dimension-Agnostic" arithmetic where complex logic emerges from index-permutation.
 * - Proving that all "imaginary" phenomena in physics are low-dimensional shadows of high-dimensional
 *   rational interactions.
 *
 * SUMMARY:
 * Imaginarity is a localized illusion resulting from low-dimensional constraints.
 * In the limit of infinite rational dimensions, all complex structures are absorbed
 * into a single, deterministic, and purely rational geometry.
 * ======================================================================================================
 */
 /**
  * 6. THE FINITE REDUCTION PRINCIPLE (Q.E.D.)
  *
  * The most profound implication of the Rational Capture Theory is the
  * Deterministic Approximation Law. Since a complex number is a projection
  * from Q^inf, any representation in a finite dimension N_j is a
  * lossy compression with a mathematically calculable error bound.
  *
  * THEOREM:
  * For every complex value z and every finite dimension N_j, there exists
  * a rational approximation in Q(i) such that the error epsilon is a
  * direct, deterministic function of the omitted dimensions (inf - N_j).
  *
  * Thus, numerical precision is not a struggle against randomness, but
  * a deliberate choice of geometric resolution within the rational manifold.
  *
  * Q.E.D.
  */
  /**
   * 7. THE COUNTABLE LIMIT AND DETERMINISTIC TRUNCATION
   *
   * A fundamental postulate of this framework is the rejection of non-countable
   * continuums. Since non-countable objects do not exist in a computable reality,
   * the infinite-dimensional rational space Q^inf is strictly COUNTABLE (ℵ₀).
   *
   * THE ITERATION-DIMENSION EQUIVALENCE:
   * In this model, the dimensionality of the projection (N_j) is equivalent to
   * the iteration count of the rational algorithm.
   * - Each new iteration is the "capture" of an additional orthogonal rational axis.
   * - The imaginary unit (i) is the limit of this process—a stable rotational
   *   state within the countable sequence of rational basis vectors.
   *
   * DETERMINISTIC ERROR BOUNDS:
   * The approximation error (ε) is no longer a stochastic rounding artifact.
   * It is defined as the "Residual Rational Mass"—the projection of the
   * remaining countable axes (from N_j+1 to ∞) that were truncated.
   *
   * CONCLUSION:
   * To compute a complex number is to perform a high-dimensional rational
   * projection. We can achieve any desired precision by simply increasing
   * the "Observation Depth" N_j. The "imaginary" is merely the "yet-to-be-computed"
   * rational remainder of the infinite sequence.
   *
   * THEORETICAL Q.E.D. COMPLETE.
   */


#ifndef DELTA_COMPLEX_GAUSS_QI_H
#define DELTA_COMPLEX_GAUSS_QI_H

#include "rational_class.h"
#include <string>
#include <utility>

namespace delta {

    class GaussQi {
    public:
        // Constructors
        GaussQi() = default;
        explicit GaussQi(const Rational& re);
        GaussQi(const Rational& re, const Rational& im);
        GaussQi(Rational&& re, Rational&& im);
        GaussQi(const GaussQi&) = default;
        GaussQi(GaussQi&&) = default;

        explicit GaussQi(long long re);
        GaussQi(long long re, long long im);
        explicit GaussQi(const std::string& str);
        GaussQi(const std::string& re_str, const std::string& im_str);
        GaussQi& operator=(const GaussQi&) = default;
        GaussQi& operator=(GaussQi&&) = default;
        // Accessors
        const Rational& real() const noexcept;
        const Rational& imag() const noexcept;
        void real(const Rational& r);
        void imag(const Rational& i);

        // Norm and conjugate
        Rational norm() const;
        GaussQi conj() const;

        // String representation
        std::string to_string() const;

        // Double approximation
        std::pair<double, double> to_double() const;

        // Compound assignments
        GaussQi& operator+=(const GaussQi& rhs);
        GaussQi& operator-=(const GaussQi& rhs);
        GaussQi& operator*=(const GaussQi& rhs);
        GaussQi& operator/=(const GaussQi& rhs);

        GaussQi& operator+=(const Rational& rhs);
        GaussQi& operator-=(const Rational& rhs);
        GaussQi& operator*=(const Rational& rhs);
        GaussQi& operator/=(const Rational& rhs);

        // Friends (binary arithmetic)
        friend GaussQi operator+(const GaussQi& lhs, const GaussQi& rhs);
        friend GaussQi operator-(const GaussQi& lhs, const GaussQi& rhs);
        friend GaussQi operator*(const GaussQi& lhs, const GaussQi& rhs);
        friend GaussQi operator/(const GaussQi& lhs, const GaussQi& rhs);

        friend GaussQi operator+(const GaussQi& lhs, const Rational& rhs);
        friend GaussQi operator+(const Rational& lhs, const GaussQi& rhs);
        friend GaussQi operator-(const GaussQi& lhs, const Rational& rhs);
        friend GaussQi operator-(const Rational& lhs, const GaussQi& rhs);
        friend GaussQi operator*(const GaussQi& lhs, const Rational& rhs);
        friend GaussQi operator*(const Rational& lhs, const GaussQi& rhs);
        friend GaussQi operator/(const GaussQi& lhs, const Rational& rhs);
        friend GaussQi operator/(const Rational& lhs, const GaussQi& rhs);

        friend GaussQi operator-(const GaussQi& z);
        friend bool operator==(const GaussQi& lhs, const GaussQi& rhs);
        friend bool operator!=(const GaussQi& lhs, const GaussQi& rhs);

    private:
        Rational re_;
        Rational im_;
    };

    // Free functions
    inline GaussQi conj(const GaussQi& z) { return z.conj(); }
    inline Rational norm(const GaussQi& z) { return z.norm(); }

    std::ostream& operator<<(std::ostream& os, const GaussQi& z);

} // namespace delta

#include "gauss_qi_impl.h"

#endif // DELTA_COMPLEX_GAUSS_QI_H