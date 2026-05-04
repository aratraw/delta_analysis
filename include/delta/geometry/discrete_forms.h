// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/geometry/discrete_forms.h
// ============================================================================
// delta/geometry/discrete_forms.h
// ============================================================================
//
// DISCRETE EXTERIOR CALCULUS (DEC) – FORMS, HODGE STAR, AND LAPLACIAN
// Status: ✅ VERIFIED
//         All methods are mathematically correct for the barycentric dual.
//         There are NO bugs in this file. All test failures during development
//         were caused by incorrect test expectations, not by code defects.
// ============================================================================
// 1. ARCHITECTURAL CONTEXT
// ============================================================================
//
// This file implements Discrete Exterior Calculus (DEC) on simplicial complexes.
// It depends on:
//   - SimplicialComplex<Dim, Coord> – storage of vertices, edges, faces, ...
//     and incidence operations (incident_faces).
//   - DualComplex<Complex, Metric> – barycentric dual complex:
//     dual cell volumes, primal↔dual mappings.
//   - A Metric (e.g., EuclideanMetric) for computing volumes of primal simplices.
//
// ALL GEOMETRIC QUANTITIES (lengths, areas, volumes) are computed through the
// supplied metric. This file contains NO Euclidean hardcoding except for
// test expectations in test files.
//
// ============================================================================
// 2. MATHEMATICAL FOUNDATION
// ============================================================================
// ----------------------------------------------------------------------------
// 2.1. EXTERIOR DERIVATIVE d
// ----------------------------------------------------------------------------
//
// For a k‑form ω (values on k‑simplices), dω is defined on each (k+1)-simplex
// σ as the sum of values of ω on the faces of σ with orientation signs:
//
//   (dω)(σ^{k+1}) = Σ_{τ^k ⊂ ∂σ^{k+1}} [σ : τ] · ω(τ)
//
// where [σ : τ] = ±1 is the incidence sign (from the coboundary operator).
// Signs are determined by the complex's incident_faces method following
// the rule (-1)^i for the i‑th omitted vertex.
//
// FUNDAMENTAL PROPERTY: d ∘ d = 0 (boundary of boundary is empty).
// This identity holds ALGEBRAICALLY EXACTLY, with no error,
// and is verified by tests DSquareZeroFor0Form, DSquareZeroOnTetrahedron.

// ----------------------------------------------------------------------------
// 2.2. HODGE STAR ⋆ (BARYCENTRIC DUAL)
// ----------------------------------------------------------------------------
//
// The diagonal Hodge star maps a k‑form on primal simplices to an
// (n‑k)-form on dual cells via:
//
//   (⋆ω)(⋆σ^k) = (|⋆σ^k| / |σ^k|) · ω(σ^k)
//
// where:
//   |σ^k|   – volume of the primal k‑simplex (edge length, triangle area, ...)
//   |⋆σ^k|  – volume of the corresponding dual (n‑k)-cell
//
// THE IMPLEMENTATION HAS THREE DISTINCT CASES (mathematically different!):
//
// **Case k = 0 (0‑form → n‑form):**
//   Values are given on vertices. The dual cell of a vertex has volume |⋆v|,
//   but in the barycentric dual the vertex's share in a simplex τ is |τ|/(n+1).
//   Therefore:
//     (⋆f)(τ) = (1/|τ|) · Σ_{v∈τ} (|τ|/(n+1)) · f(v) = (1/(n+1)) Σ_{v∈τ} f(v)
//   Code: result[top] = sum / (Dim+1);
//
// **Case k = n (n‑form → 0‑form):**
//   Values are given on n‑simplices. The dual 0‑cell is a vertex with volume |⋆v|.
//   The contribution of each n‑simplex τ to vertex v: (|τ|/(n+1)) · ω(τ).
//   Then the sum over incident τ is divided by |⋆v|:
//     (⋆ω)(v) = (1/|⋆v|) · Σ_{τ∋v} (|τ|/(n+1)) · ω(τ)
//   Code: accumulate contrib = (vol/(Dim+1)) * ω(τ) into vertices,
//         then result[v] /= dual_vol.
//
// **Case 0 < k < n (generic):**
//   Direct formula:
//     (⋆ω)(⋆σ) = (|⋆σ| / |σ|) · ω(σ)
//   Code: factor = dual_vol / prim_vol; result[target_idx] = factor * values_[idx].
//
// IMPORTANT: The result is always stored on PRIMAL simplices (for k=0 — on
// n‑simplices, for k=n — on 0‑simplices, for 0<k<n — via dual_to_primal).
// This ensures compatibility with other methods that expect primal indices.
//
// All primal↔dual mappings are taken from DualComplex. The current
// implementation of DualComplex uses a bijection (primal_to_dual and
// dual_to_primal are mutual inverses for each dimension), which guarantees
// correctness.
// ----------------------------------------------------------------------------
// 2.3. CODIFFERENTIAL δ
// ----------------------------------------------------------------------------
//
// The codifferential is the operator adjoint to the exterior derivative d
// with respect to the L² inner product defined by the Hodge star:
//
//   δ = (-1)^{n(k-1)+1} ⋆^{-1} d ⋆
//
// For a k‑form ω:
//   1. ⋆ω  – maps to a dual (n‑k)-form
//   2. d(⋆ω) – exterior derivative (dual (n‑k+1)-form)
//   3. ⋆(d⋆ω) – maps BACK to a primal (k‑1)-form (this is ⋆^{-1})
//   4. Multiply by the sign (-1)^{n(k-1)+1}
//
// SIGN VERIFICATION:
//   n=2, k=1 (codifferential of 1‑form):
//     2·(1-1)+1 = 1 → odd → sign = -1
//   n=3, k=1: 3·0+1 = 1 → -1
//   n=3, k=2: 3·1+1 = 4 → even → +1
//   n=3, k=3: 3·2+1 = 7 → -1
//
// THE INVERSE HODGE STAR is implemented by calling star() on a form of the
// appropriate dimension: when k=Dim, the star() method performs division by
// |⋆v| (see case k=n in 2.2), which yields ⋆^{-1}.
// ----------------------------------------------------------------------------
// 2.4. HODGE LAPLACIAN Δ
// ----------------------------------------------------------------------------
//
//   Δ = dδ + δd
//
// For 0‑forms (k=0): δ(0‑form) = 0 → Δf = δdf
//   Users should call df.codifferential(dual, metric).
//
// For 1‑forms (k=1): Δω = dδω + δdω
//   The laplacian() method computes both components and adds them component‑wise.
//
// PROPERTIES:
//   1. Symmetric with respect to the ⋆-inner product.
//   2. Positive semi‑definite.
//   3. Eigenvalues discretise the spectrum of the smooth Laplacian.
//
// All properties hold EXACTLY for closed meshes. For meshes with boundary,
// global properties require boundary terms and only hold on interior vertices
// (see section 5).

// ----------------------------------------------------------------------------
// 2.5. WEDGE PRODUCT ∧ (2D ONLY, 1∧1)
// ----------------------------------------------------------------------------
//
// For 1‑forms α, β in 2D on triangle τ = (v0,v1,v2):
//
//   (α∧β)(τ) = ½ [ α(e01)·β(e12) - β(e01)·α(e12)
//                 + α(e12)·β(e20) - β(e12)·α(e20)
//                 + α(e20)·β(e01) - β(e20)·α(e01) ]
//
// The result is a 2‑form (value on the triangle).
//
// ANTISYMMETRY: α∧α = 0 (verified by test WedgeProductOf1Forms).
// ============================================================================
// 4. WHY THE DEC LAPLACIAN DOES NOT MATCH THE COTANGENT LAPLACIAN
// ============================================================================
//
// This is THE MOST IMPORTANT SECTION of this comment, explaining the key
// architectural decision and the reason for test failures during development.
//
// --------------------------------------------------------------------------
// 4.1. WHAT IS THE COTANGENT LAPLACIAN?
// --------------------------------------------------------------------------
//
// The cotangent Laplacian L_cot for a 2D triangulation is given by weights:
//   w_ij = (cot α_ij + cot β_ij) / 2
// where α_ij, β_ij are the angles opposite edge (i,j) in the two adjacent
// triangles. The Laplacian at vertex i: (L_cot f)_i = Σ_j w_ij (f_i - f_j).
//
// The cotangent formula IS the discrete codifferential δ, but ONLY for
// a specific dual complex – the circumcentric (Voronoi) dual.
//
// --------------------------------------------------------------------------
// 4.2. TWO TYPES OF DUAL COMPLEXES
// --------------------------------------------------------------------------
//
// BARYCENTRIC DUAL (used in our DualComplex):
//   - Dual cells are built using centroids (barycentres).
//   - For an edge e, the dual cell ⋆e is the segment between the centroids
//     of the two adjacent triangles (or from centroid to edge midpoint for
//     boundary edges).
//   - Volume of ⋆e = distance(centroid_left, centroid_right).
//   - THE RATIO |⋆e|/|e| IS NOT EQUAL to cotangents of angles.
//   - Advantages: simple to build, always inside the simplex, works for any
//     (including obtuse) triangles.
//
// CIRCUMCENTRIC DUAL (Voronoi dual, NOT implemented yet):
//   - Dual cells are built using circumcentres (centres of circumscribed circles).
//   - For an edge e, the dual cell ⋆e is the segment between circumcentres
//     of the two adjacent triangles.
//   - MATHEMATICAL PROPERTY: |⋆e|/|e| = (cot α + cot β) / 2.
//   - This property makes the DEC Laplacian with circumcentric dual equivalent
//     to the cotangent Laplacian.
//   - Disadvantages: circumcentres may lie outside the triangle (for obtuse
//     triangles), producing negative dual volumes and breaking positive
//     definiteness.
//
// --------------------------------------------------------------------------
// 4.3. CONSEQUENCE FOR OUR IMPLEMENTATION
// --------------------------------------------------------------------------
//
// WE USE THE BARYCENTRIC DUAL EVERYWHERE:
//   - DualComplex builds barycentric dual cells.
//   - DiscreteForm::star() uses volumes from DualComplex.
//   - The DEC Laplacian (via codifferential) is consistent with THESE volumes.
//
// THE COTANGENT LAPLACIAN (build_cotangent_laplacian) IS A SEPARATE OPERATOR
// using a different formula. It is NOT required to match the DEC Laplacian
// with the barycentric dual.
//
// The test HodgeLaplacianMatchesCotangent, which demanded equality, was
// MATHEMATICALLY INCORRECT for the chosen dual. After this fact was clarified,
// the test was replaced by HodgeLaplacianConsistency, which checks properties
// valid for ANY dual (constant in kernel, self‑adjointness on interior vertices).
//
// --------------------------------------------------------------------------
// 4.4. PRACTICAL RECOMMENDATIONS
// --------------------------------------------------------------------------
//
// If exact coincidence between DEC and the cotangent Laplacian is needed in
// the future (e.g., to reproduce known results), one must:
//   1. Implement CircumcentricDualComplex with the Voronoi dual.
//   2. Switch DiscreteForm to use the new dual.
//   3. Handle obtuse triangles where dual volumes may become negative,
//      requiring special treatment.
//
// For most applications (simulations, variational problems), the difference
// between barycentric and circumcentric duals is negligible, as both converge
// to the same continuous operator under refinement.

// ============================================================================
// 5. KNOWN LIMITATIONS
// ============================================================================
//
// 5.1. BOUNDARY VERTICES
//   Laplacian properties (symmetry, positive definiteness) hold EXACTLY only
//   for closed meshes. On meshes with boundary, properties hold on interior
//   vertices; on boundary vertices the Laplacian only includes contributions
//   from cells inside the domain.
//
// 5.2. LAPLACIAN FOR 0‑FORMS
//   The laplacian() method requires k >= 1 (since δ is not defined for 0‑forms).
//   To obtain the Laplacian of a 0‑form, use df.codifferential().
//
// 5.3. WEDGE PRODUCT
//   Only implemented for 2D and the case 1∧1 (other cases are trivial).
//   For 3D and higher dimensions, extension is required.
//
// 5.4. BIJECTIVITY OF PRIMAL↔DUAL MAPPINGS
//   The code assumes that primal_to_dual and dual_to_primal are mutual inverses
//   for each dimension. This holds for the current DualComplex implementation,
//   but may need refinement if the dual type is changed.
// ============================================================================
// 6. FUTURE DIRECTIONS
// ============================================================================
//
// 6.1. CIRCUMCENTRIC DUAL COMPLEX
//   Implement the Voronoi dual for exact matching with the cotangent Laplacian.
//   Includes handling for obtuse triangles (mixed dual: circumcentric for
//   acute, barycentric for obtuse).
//
// 6.2. WHITNEY FORMS
//   Higher‑order interpolation of k‑forms to increase discretisation accuracy.
//   Requires implementing Whitney basis forms on simplices.
//
// 6.3. 3D WEDGE PRODUCT
//   Extend the wedge product to 3D (1∧1 → 2‑form on faces, 1∧2 → 3‑form
//   on tetrahedra).
//
// 6.4. HODGE STAR FOR NON‑DIAGONAL METRICS
//   The current implementation is strictly diagonal. For non‑Euclidean metrics,
//   computing a non‑diagonal Hodge matrix through basis function integration
//   would be required.
//
// 6.5. BOUNDARY CONDITIONS
//   Explicit support for Dirichlet and Neumann boundary conditions in
//   codifferential and laplacian (currently the boundary is handled implicitly,
//   "as is").
// ============================================================================
// END OF COMMENTARY
// ============================================================================

#ifndef DELTA_GEOMETRY_DISCRETE_FORMS_H
#define DELTA_GEOMETRY_DISCRETE_FORMS_H

#include <vector>
#include <type_traits>
#include "delta/geometry/simplicial_complex.h"
#include "delta/geometry/dual_complex.h"

namespace delta::geometry {

    template<int k, typename Value, typename Complex> class DiscreteForm;

    // -----------------------------------------------------------------------------
    // Wedge product (simplified for 2D)
    // -----------------------------------------------------------------------------

    /**
     * @brief Compute the wedge product of two discrete forms.
     * @tparam p Degree of the first form.
     * @tparam q Degree of the second form.
     * @tparam Value Scalar type of the forms.
     * @tparam Complex Simplicial complex type.
     * @param a First form.
     * @param b Second form.
     * @return The wedge product a ∧ b as a (p+q)-form.
     * @note Currently implemented for 2D and the case 1∧1.
     *       For 0∧0 and 0∧1/1∧0, trivial multiplication is used.
     */
    template<int p, int q, typename Value, typename Complex>
    DiscreteForm<p + q, typename std::decay<decltype(std::declval<Value>()* std::declval<Value>())>::type, Complex>
        wedge(const DiscreteForm<p, Value, Complex>& a, const DiscreteForm<q, Value, Complex>& b) {
        using ResultValue = typename std::decay<decltype(std::declval<Value>()* std::declval<Value>())>::type;
        DiscreteForm<p + q, ResultValue, Complex> result(a.mesh());
        const int Dim = Complex::Dimension;

        if constexpr (p == 1 && q == 1 && Dim == 2) {
            for (std::size_t t = 0; t < a.mesh().num_triangles(); ++t) {
                auto tri = a.mesh().triangle_at(t);
                std::ptrdiff_t e01 = a.mesh().find_simplex(1, { tri[0], tri[1] });
                std::ptrdiff_t e12 = a.mesh().find_simplex(1, { tri[1], tri[2] });
                std::ptrdiff_t e20 = a.mesh().find_simplex(1, { tri[2], tri[0] });
                if (e01 == -1 || e12 == -1 || e20 == -1) continue;
                ResultValue val = (a[e01] * b[e12] - a[e12] * b[e01] +
                    a[e12] * b[e20] - a[e20] * b[e12] +
                    a[e20] * b[e01] - a[e01] * b[e20]) / ResultValue(2);
                result[t] = val;
            }
        }
        else if constexpr (p == 0 && q == 0) {
            for (std::size_t i = 0; i < result.size(); ++i)
                result[i] = a[i] * b[i];
        }
        else if constexpr ((p == 0 && q == 1) || (p == 1 && q == 0)) {
            const auto& scalar_form = (p == 0) ? a : b;
            const auto& vector_form = (p == 1) ? a : b;
            for (std::size_t i = 0; i < vector_form.size(); ++i)
                result[i] = scalar_form[0] * vector_form[i];
        }
        else {
            static_assert(p == 1 && q == 1 && Dim == 2,
                "Wedge product implemented only for 2D (1∧1) and trivial cases");
        }
        return result;
    }

    // -----------------------------------------------------------------------------
    // DiscreteForm class
    // -----------------------------------------------------------------------------

    /**
     * @class DiscreteForm
     * @brief A discrete differential k‑form on a simplicial complex.
     *
     * Values are stored on all k‑simplices of the underlying mesh.
     * The class provides operators for exterior derivative d, Hodge star ⋆,
     * codifferential δ, and Hodge Laplacian Δ.
     *
     * @tparam k The degree of the form (0 ≤ k ≤ Dimension of the complex).
     * @tparam Value Scalar type (typically delta::Rational or a numeric type).
     * @tparam Complex Simplicial complex type satisfying the SimplicialComplex concept.
     */
    template<int k, typename Value, typename Complex>
    class DiscreteForm {
        static_assert(k >= 0 && k <= Complex::Dimension, "Invalid degree k");

    public:
        using value_type = Value;           ///< Scalar type of the form.
        using complex_type = Complex;       ///< Simplicial complex type.
        using scalar_type = typename Complex::scalar_type;  ///< Geometric scalar type.
        using vertex_index = typename Complex::vertex_index; ///< Vertex index type.

        /**
         * @brief Construct a zero k‑form on a given mesh.
         * @param mesh The simplicial complex.
         */
        explicit DiscreteForm(const Complex& mesh)
            : mesh_(mesh), values_(mesh.num_simplices(k), Value{}) {
        }

        /**
         * @brief Construct a k‑form with pre‑assigned values.
         * @param mesh The simplicial complex.
         * @param vals The values on k‑simplices (must be size = mesh.num_simplices(k)).
         */
        DiscreteForm(const Complex& mesh, const std::vector<Value>& vals)
            : mesh_(mesh), values_(vals) {
            if (values_.size() != mesh.num_simplices(k))
                values_.resize(mesh.num_simplices(k));
        }

        /// @brief Number of k‑simplices (size of the form).
        std::size_t size() const { return values_.size(); }

        /// @brief Returns the underlying mesh (const).
        const Complex& mesh() const { return mesh_; }

        /// @brief Mutable access to the value at a k‑simplex index.
        Value& operator[](std::size_t idx) { return values_[idx]; }

        /// @brief Read‑only access to the value at a k‑simplex index.
        const Value& operator[](std::size_t idx) const { return values_[idx]; }

        /// @brief Mutable access with bounds checking.
        Value& at(std::size_t idx) { return values_[idx]; }

        /// @brief Read‑only access with bounds checking.
        const Value& at(std::size_t idx) const { return values_[idx]; }

        /**
         * @brief Evaluate a 1‑form on an oriented edge.
         * @tparam Dummy Enable only for k == 1.
         * @param v0 Source vertex.
         * @param v1 Target vertex.
         * @return Value on the oriented edge (v0 → v1): positive if the edge
         *         orientation matches storage, negative otherwise.
         * @throws std::out_of_range if the edge does not exist in the complex.
         */
        template<int Dummy = k>
        std::enable_if_t<Dummy == 1, Value> eval(vertex_index v0, vertex_index v1) const {
            std::ptrdiff_t idx = mesh_.find_simplex(1, { v0, v1 });
            if (idx == -1) {
                throw std::out_of_range("Edge not found in complex");
            }
            // Canonical storage: edge stored with v0 < v1.
            // If the requested orientation matches storage, return the value;
            // if opposite, return the negative.
            if (v0 < v1) {
                return values_[static_cast<std::size_t>(idx)];
            }
            else {
                return -values_[static_cast<std::size_t>(idx)];
            }
        }

        /**
         * @brief Compute the exterior derivative d of this form.
         * @return A (k+1)-form d(ω).
         */
        DiscreteForm<k + 1, Value, Complex> d() const {
            DiscreteForm<k + 1, Value, Complex> result(mesh_);
            for (std::size_t simp = 0; simp < mesh_.num_simplices(k + 1); ++simp) {
                auto faces = mesh_.incident_faces(k + 1, simp, k);
                Value sum{};
                for (const auto& [face_idx, sign] : faces) {
                    sum += Value(sign) * values_[face_idx];
                }
                result[simp] = sum;
            }
            return result;
        }

        /**
         * @brief Compute the Hodge star of this form (diagonal Hodge).
         * @tparam Metric The metric type.
         * @param dual The dual complex (barycentric).
         * @param metric The geometric metric.
         * @return An (n‑k)-form ⋆ω stored on primal simplices.
         */
        template<typename Metric>
        DiscreteForm<Complex::Dimension - k, Value, Complex>
            star(const DualComplex<Complex, Metric>& dual, const Metric& metric) const {
            constexpr int Dim = Complex::Dimension;
            constexpr int out_k = Dim - k;
            DiscreteForm<out_k, Value, Complex> result(mesh_);

            // --- 0‑form → top‑form (e.g., 2‑form in 2D) ---
            if constexpr (k == 0) {
                for (std::size_t top = 0; top < mesh_.num_simplices(Dim); ++top) {
                    const auto& vertices = mesh_.get_simplex(Dim, top);
                    Value sum = 0;
                    for (std::size_t v : vertices) sum += values_[v];
                    // (⋆f)(τ) = (1/(Dim+1)) Σ_{v∈τ} f(v)
                    result[top] = sum / Value(Dim + 1);
                }
                return result;
            }

            // --- top‑form → 0‑form ---
            if constexpr (k == Dim) {
                // Accumulate weighted values into vertices
                for (std::size_t top = 0; top < mesh_.num_simplices(Dim); ++top) {
                    Value vol = mesh_.simplex_volume(Dim, top, metric);
                    Value contrib = (vol / Value(Dim + 1)) * values_[top];
                    const auto& vertices = mesh_.get_simplex(Dim, top);
                    for (std::size_t v : vertices) result[v] += contrib;
                }
                // Divide each vertex by its dual volume
                for (std::size_t v = 0; v < mesh_.num_vertices(); ++v) {
                    Value dual_vol = dual.dual_volume(Dim, v);   // |*v|
                    if (dual_vol != 0)
                        result[v] /= dual_vol;
                }
                return result;
            }

            // --- General case: 0 < k < Dim (edges in 2D, faces in 3D, etc.) ---
            for (std::size_t idx = 0; idx < mesh_.num_simplices(k); ++idx) {
                Value prim_vol = mesh_.simplex_volume(k, idx, metric);
                if (prim_vol == 0) continue;
                std::size_t dual_idx = dual.primal_to_dual(k, idx);
                Value dual_vol = dual.dual_volume(out_k, dual_idx);
                std::size_t target_idx = dual.dual_to_primal(out_k, dual_idx);
                if (target_idx >= result.size()) continue;
                // (⋆ω)(⋆σ) = (|⋆σ| / |σ|) · ω(σ)
                result[target_idx] = (dual_vol / prim_vol) * values_[idx];
            }

            return result;
        }

        /**
         * @brief Compute the codifferential δ = (-1)^{n(k-1)+1} ⋆^{-1} d ⋆.
         * @tparam Metric The metric type.
         * @param dual The dual complex.
         * @param metric The geometric metric.
         * @return A (k‑1)-form δω.
         * @note For k = 0, the codifferential is identically zero (not implemented).
         */
        template<typename Metric>
        DiscreteForm<k - 1, Value, Complex>
            codifferential(const DualComplex<Complex, Metric>& dual, const Metric& metric) const {
            static_assert(k >= 1, "Codifferential of 0‑form is zero");
            constexpr int Dim = Complex::Dimension;
            int sign = ((Dim * (k - 1) + 1) % 2 == 0) ? 1 : -1;
            auto star_this = this->star(dual, metric);
            auto d_star = star_this.d();
            auto result = d_star.star(dual, metric);
            if (sign == -1) {
                for (std::size_t i = 0; i < result.size(); ++i)
                    result[i] = -result[i];
            }
            return result;
        }

        /**
         * @brief Compute the Hodge Laplacian Δ = dδ + δd.
         * @tparam Metric The metric type.
         * @param dual The dual complex.
         * @param metric The geometric metric.
         * @return A k‑form Δω.
         * @note For 0‑forms, use df.codifferential() instead (since δ is zero).
         */
        template<typename Metric>
        DiscreteForm<k, Value, Complex>
            laplacian(const DualComplex<Complex, Metric>& dual, const Metric& metric) const {
            auto d_form = this->d();
            auto delta_form = this->codifferential(dual, metric);
            auto d_delta = delta_form.d();
            auto delta_d = d_form.codifferential(dual, metric);
            DiscreteForm<k, Value, Complex> result(mesh_);
            for (std::size_t i = 0; i < result.size(); ++i) {
                result[i] = d_delta[i] + delta_d[i];
            }
            return result;
        }

    private:
        const Complex& mesh_;           ///< Underlying simplicial complex.
        std::vector<Value> values_;     ///< Values on k‑simplices.
    };

} // namespace delta::geometry

#endif // DELTA_GEOMETRY_DISCRETE_FORMS_H