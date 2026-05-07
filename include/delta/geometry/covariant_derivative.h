// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/geometry/covariant_derivative.h
// ============================================================================
// COVARIANT DERIVATIVE ON A SIMPLICIAL COMPLEX
// ============================================================================
//
// Provides covariant derivative operators for scalar, vector, and (1,1)‑tensor
// fields along an oriented edge of a simplicial complex.  The field is stored
// as a TensorField<vertex_index, Scalar, Rank, Dim> on the vertices of the
// complex.  The connection provides the parallel transport matrices U_{i→j}.
//
// All formulas use the edge length computed from the supplied metric.
//
// For a scalar field f:
//   ∇_e f = (f(y) − f(x)) / ℓ(e)
//
// For a vector field v:
//   ∇_e v = (U_{x→y}^{-1} v_y − v_x) / ℓ(e)       (result is a vector at x)
//
// For a (1,1)‑tensor field M (matrix):
//   ∇_e M = (U_{x→y}^{-1} M_y U_{x→y} − M_x) / ℓ(e)
//
// CRITICAL: Never use `auto` to capture the result of Eigen::inverse() or
// any other expression template that depends on temporaries – it produces
// dangling references.  Always store the inverse in a plain matrix.
// ============================================================================
// ============================================================================
// WHY THIS FILE IS ARCHITECTURALLY UNAVOIDABLE AND MATHEMATICALLY CORRECT
// ============================================================================
//
// 1.  WHY WE NEED AN EXPLICIT COVARIANT DERIVATIVE
//     --------------------------------------------
//     The library already has a discrete exterior derivative (d) via
//     DiscreteForm, which works on k‑forms (cochains).  However, a connection
//     lives on the *tangent bundle* (or an associated vector bundle), and the
//     covariant derivative is the natural operator for tensor fields that
//     transform under the structure group.  Keeping the two concepts separate
//     follows the Δ‑analysis philosophy: each regulative idea (here: metric
//     vs. connection) defines its own differential calculus.
//
// 2.  CHOICE OF THE FORMULA
//     ----------------------
//     The formulas
//        ∇_e v = (U^{-1} v_y − v_x) / ℓ(e)          (vector)
//        ∇_e M = (U^{-1} M_y U − M_x) / ℓ(e)       (1,1)-tensor)
//     are the standard discrete realisations of the Leibniz rule and the
//     requirement that the connection is metric‑compatible (when U ∈ O(n))
//     or at least linear.  They are exact analogues of the continuous
//     expression  ∇_X T = lim_{ϵ→0} (τ^{-1} T(γ(ϵ)) − T(γ(0))) / ϵ,
//     where τ is parallel transport along the geodesic.
//
// 3.  WHY THE EDGE LENGTH APPEARS IN THE DENOMINATOR
//     -----------------------------------------------
//     The metric is used solely to compute ℓ(e) = d(x, y).  This is the
//     only place where the metric enters.  In a genuinely metric‑based
//     geometry, the parallel transport U already encodes the metric, so ℓ(e)
//     could be recovered from U.  However, we separate the two for
//     pedagogical clarity and to allow connections that are not metric‑
//     compatible.  In a future “metric‑first” refactoring (see the Sweep &
//     Scale plan), ℓ(e) could be computed from the connection itself if it
//     preserves a metric.
//
// 4.  THE AUTO‑TRAP (CRITICAL IMPLEMENTATION NOTE)
//     ---------------------------------------------
//     The line
//         auto Uinv = conn.get_transport(x, y).inverse();
//     creates an Eigen expression template that stores a reference to the
//     temporary matrix returned by get_transport().  The temporary dies at
//     the semicolon, and Uinv becomes a dangling reference.  In debug mode,
//     the memory often survives long enough to give correct results; in
//     release mode, the memory is reused immediately, causing a segfault.
//     This is not a bug in Eigen – it is a well‑documented consequence of
//     expression templates.  The fix is always to materialise:
//         Eigen::Matrix<Scalar,Dim,Dim> U = conn.get_transport(x, y);
//         Eigen::Matrix<Scalar,Dim,Dim> Uinv = U.inverse();
//     Do NOT use `auto` with any Eigen operation that returns a temporary.
//
// 5.  WHY THE FUNCTION IS OVERLOADED AND NOT TEMPLATE‑DISPATCHED ON RANK
//     -------------------------------------------------------------------
//     We use three separate function signatures (scalar=rank 0, vector=rank 1,
//     matrix=rank 2) instead of a single template parameterised by Rank.
//     This is intentional: the formulas are fundamentally different, and
//     deducing the return type correctly with SFINAE or if‑constexpr would
//     be much harder to read.  The extra verbosity is a price we pay for
//     clarity and compiler error messages that are understandable.
//
// 6.  EIGEN NAMING CONVENTION
//     ------------------------
//     The return types are `Eigen::Matrix<Scalar, Dim, 1>` (vector) and
//     `Eigen::Matrix<Scalar, Dim, Dim>` (matrix).  We deliberately do NOT
//     use our `geometry::Vector<Scalar,Dim>` alias here, because the result
//     is a plain coordinate tuple, not a geometric vector that needs to
//     belong to the constructive core.  The distinction between “algebraic
//     vector” and “geometric vector” is important in Δ‑analysis, and using
//     Eigen::Matrix directly signals that this is an algebraic quantity.
//
// 7.  FUTURE EXTENSIONS
//     ------------------
//     - Rank‑2 tensors of type (0,2) and (2,0) (e.g., metric fields).
//     - Covariant derivative for tensor fields on higher‑dimensional
//       complexes (already works generically thanks to the Dim parameter).
//     - Support for non‑metric connections (already partially supported,
//       because the metric only provides ℓ(e)).
// ============================================================================
#pragma once

#include "delta/core/rational.h"
#include "delta/geometry/tensor_field.h"
#include "delta/geometry/simplicial_complex.h"
#include "delta/geometry/connection.h"
#include "delta/core/regulative_idea.h"   // Metric concept

namespace delta::geometry {

    // -------------------------------------------------------------------------
    // Scalar field
    // -------------------------------------------------------------------------
    template<typename Complex, typename Scalar, typename Metric>
    Scalar covariant_derivative(
        const TensorField<typename Complex::vertex_index, Scalar, 0, Complex::Dimension>& field,
        const Connection<typename Complex::vertex_index, Scalar, Complex::Dimension>& /*conn*/,
        const std::array<typename Complex::vertex_index, 2>& edge,
        const Complex& mesh,
        const Metric& metric)
    {
        auto x = edge[0];
        auto y = edge[1];
        Scalar fx = field.at(x);
        Scalar fy = field.at(y);
        Scalar len = metric(mesh.vertex(x), mesh.vertex(y));
        if (len == Scalar(0))
            throw std::domain_error("covariant_derivative: zero‑length edge");
        return (fy - fx) / len;
    }

    // -------------------------------------------------------------------------
    // Vector field (rank 1)
    // -------------------------------------------------------------------------
    template<typename Complex, typename Scalar, int Dim, typename Metric>
    Eigen::Matrix<Scalar, Dim, 1> covariant_derivative(
        const TensorField<typename Complex::vertex_index, Scalar, 1, Dim>& field,
        const Connection<typename Complex::vertex_index, Scalar, Dim>& conn,
        const std::array<typename Complex::vertex_index, 2>& edge,
        const Complex& mesh,
        const Metric& metric)
    {
        auto x = edge[0];
        auto y = edge[1];
        auto vx = field.at(x);
        auto vy = field.at(y);
        auto len = metric(mesh.vertex(x), mesh.vertex(y));
        if (len == Scalar(0))
            throw std::domain_error("covariant_derivative: zero‑length edge");

        // Materialise the inverse – do NOT use `auto` here!
        Eigen::Matrix<Scalar, Dim, Dim> U = conn.get_transport(x, y);
        Eigen::Matrix<Scalar, Dim, Dim> Uinv = U.inverse();

        return (Uinv * vy - vx) / len;
    }

    // -------------------------------------------------------------------------
    // (1,1)‑tensor field (matrix)
    // -------------------------------------------------------------------------
    template<typename Complex, typename Scalar, int Dim, typename Metric>
    Eigen::Matrix<Scalar, Dim, Dim> covariant_derivative(
        const TensorField<typename Complex::vertex_index, Scalar, 2, Dim>& field,
        const Connection<typename Complex::vertex_index, Scalar, Dim>& conn,
        const std::array<typename Complex::vertex_index, 2>& edge,
        const Complex& mesh,
        const Metric& metric)
    {
        auto x = edge[0];
        auto y = edge[1];
        auto Mx = field.at(x);
        auto My = field.at(y);
        auto len = metric(mesh.vertex(x), mesh.vertex(y));
        if (len == Scalar(0))
            throw std::domain_error("covariant_derivative: zero‑length edge");

        // Materialise – do NOT use `auto` here!
        Eigen::Matrix<Scalar, Dim, Dim> U = conn.get_transport(x, y);
        Eigen::Matrix<Scalar, Dim, Dim> Uinv = U.inverse();

        return (Uinv * My * U - Mx) / len;
    }

} // namespace delta::geometry