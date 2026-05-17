// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/numerical/cotangent_laplacian.h
// ============================================================================
// COTANGENT LAPLACIAN AND LUMPED MASS MATRIX
// ============================================================================
//
// This file provides two fundamental discrete differential operators for
// 2D triangle meshes:
//
//   1. Cotangent Laplacian – the standard discretisation of the Laplace–Beltrami
//      operator on triangle meshes, widely used in geometry processing.
//   2. Lumped mass matrix – diagonal matrix representing vertex areas
//      (dual volumes) for computing inner products.
//
// ----------------------------------------------------------------------------
// MATHEMATICAL DEFINITION
// ----------------------------------------------------------------------------
//
// For an edge (i,j) with two incident triangles, let α and β be the angles
// opposite that edge. The cotangent weight is:
//
//      w_ij = (cot α + cot β) / 2
//
// For boundary edges, only the single triangle contributes (β = 0 → cot β = 0).
//
// The Laplacian matrix L (size V × V) is defined as:
//
//      L_ii = Σ_{j≠i} w_ij
//      L_ij = -w_ij          for i ≠ j
//
// This matrix is symmetric, positive semi‑definite, and satisfies:
//   - L·1 = 0 (constant functions are in the kernel)
//   - For a linear function on a regular triangulation, L approximates
//     the smooth Laplacian with second‑order accuracy.
//
// ----------------------------------------------------------------------------
// IMPLEMENTATION NOTES
// ----------------------------------------------------------------------------
//
// - The cotangent computation uses Heron's formula:
//       area = √(s(s-a)(s-b)(s-c))
//       cot(θ) = (b² + c² - a²) / (4·area)
//   where a = length of edge opposite angle θ.
//
// - All geometric quantities (edge lengths, areas) are computed using the
//   supplied metric. The metric must satisfy the Metric concept and return
//   distances (scalar_type).
//
// - The lumped mass matrix uses vertex dual volumes computed as one‑third
//   of the sum of areas of incident triangles (barycentric dual).
//
// - Both matrices are returned as Eigen::SparseMatrix<scalar_type>.
// ============================================================================

// ============================================================================
// TODO: COTANGENT LAPLACIAN – FUTURE IMPROVEMENTS
// ============================================================================
//
// 1. **Metric‑agnostic lumped mass matrix**
//    Currently, build_lumped_mass_matrix computes area directly from vertex
//    coordinates (assuming Euclidean embedding). For non‑Euclidean metrics,
//    area should be computed via Heron's formula using edge lengths from the
//    supplied metric. This would require passing a metric argument to the
//    function (currently missing).
//
// 2. **Boundary handling verification**
//    The current implementation correctly handles boundary edges by using
//    only one triangle (the other does not exist, so cot = 0). However, the
//    resulting Laplacian at boundary vertices may have different spectral
//    properties than the interior. Consider adding explicit Dirichlet/Neumann
//    boundary condition support (e.g., zero‑Dirichlet by removing boundary
//    vertices from the matrix).
//
// 3. **3D extension**
//    The cotangent formula also exists for tetrahedral meshes (opposite edge
//    angles). Extend the implementation to 3D with `build_cotangent_laplacian_3d`.
//    The formula: w_ij = Σ_t (cot α_t + cot β_t) / 2, summing over all tetrahedra
//    containing edge (i,j), where α_t and β_t are the dihedral angles opposite
//    the edge in the two adjacent tetrahedra (or one for boundary).
//
// 4. **Performance optimisation**
//    - The current implementation computes sqrt and square multiple times per
//      triangle. Could pre‑compute squared edge lengths and reuse.
//    - Triplet generation is O(ntriangles) but the number of triplets is at most
//      O(vertices + edges). Could reserve space in `triplets` vector to avoid
//      reallocations.
//    - For very large meshes, consider using `Eigen::SparseMatrix` with a
//      precomputed pattern (since the sparsity pattern is known in advance).
//
// 5. **Numerical stability for skinny triangles**
//    For very small angles, cot can become huge, leading to numerical issues.
//    Consider adding a heuristic to clamp cotangents or adjust weights for
//    degenerate/sliver triangles.
//
// ============================================================================

#ifndef DELTA_NUMERICAL_COTANGENT_LAPLACIAN_H
#define DELTA_NUMERICAL_COTANGENT_LAPLACIAN_H

#include <Eigen/Sparse>
#include <vector>
#include "delta/rational/eigen_integration.h"
#include "delta/geometry/simplicial_complex.h"
#include "delta/geometry/dual_complex.h"
#include "delta/core/regulative_idea.h"

namespace delta::numerical {

    /**
     * @brief Build the cotangent Laplacian matrix for a 2D triangle mesh.
     *
     * @tparam Complex SimplicialComplex<2> type.
     * @tparam Metric  A metric satisfying delta::Metric concept (must return distance).
     * @param mesh     The 2D triangle mesh.
     * @param metric   The metric used to measure distances.
     * @return Sparse matrix L (size = number of vertices).
     */
    template<typename Complex, typename Metric>
    Eigen::SparseMatrix<typename Complex::scalar_type>
        build_cotangent_laplacian(const Complex& mesh, const Metric& metric) {
        static_assert(Complex::Dimension == 2, "Cotangent Laplacian only for 2D complexes");
        using Scalar = typename Complex::scalar_type;
        using VertexIndex = typename Complex::vertex_index;

        std::size_t nv = mesh.num_vertices();
        std::vector<Eigen::Triplet<Scalar>> triplets;

        // Helper: squared edge length via metric
        auto sq_len = [&](VertexIndex i, VertexIndex j) -> Scalar {
            Scalar d = metric(mesh.vertex(i), mesh.vertex(j));
            return d * d;
            };

        // Process each triangle
        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto tri = mesh.triangle_at(t);
            VertexIndex i = tri[0], j = tri[1], k = tri[2];

            // Squared edge lengths: a opposite i, b opposite j, c opposite k
            Scalar a2 = sq_len(j, k);   // opposite i
            Scalar b2 = sq_len(k, i);   // opposite j
            Scalar c2 = sq_len(i, j);   // opposite k

            // Edge lengths (for Heron's formula)
            Scalar a = delta::sqrt(a2);
            Scalar b = delta::sqrt(b2);
            Scalar c = delta::sqrt(c2);
            Scalar s = (a + b + c) / 2;                     // semiperimeter
            Scalar area = delta::sqrt(s * (s - a) * (s - b) * (s - c)); // Heron
            if (area == 0) continue;   // degenerate triangle

            // Cotangents of the three angles
            // cot(angle opposite side a) = (b² + c² - a²) / (4·area)
            Scalar cot_i = (b2 + c2 - a2) / (4 * area);   // at vertex i (opposite edge jk)
            Scalar cot_j = (c2 + a2 - b2) / (4 * area);   // at vertex j
            Scalar cot_k = (a2 + b2 - c2) / (4 * area);   // at vertex k

            // For each edge, the weight is half the cotangent of the opposite angle
            // Contribution to stiffness: +cot/2 on diagonals, -cot/2 on off-diagonals
            auto add_edge = [&](VertexIndex u, VertexIndex v, Scalar cot) {
                triplets.emplace_back(u, v, -cot / 2);
                triplets.emplace_back(v, u, -cot / 2);
                triplets.emplace_back(u, u, cot / 2);
                triplets.emplace_back(v, v, cot / 2);
                };

            add_edge(i, j, cot_k);   // edge ij → opposite angle at k
            add_edge(j, k, cot_i);   // edge jk → opposite angle at i
            add_edge(k, i, cot_j);   // edge ki → opposite angle at j
        }

        Eigen::SparseMatrix<Scalar> L(nv, nv);
        L.setFromTriplets(triplets.begin(), triplets.end());
        return L;
    }

    /**
     * @brief Build the lumped mass matrix (vertex dual volumes) for a 2D triangle mesh.
     *
     * The lumped mass matrix M is diagonal, with M_ii = dual_volume(vertex i),
     * i.e., the area of the barycentric dual cell surrounding vertex i,
     * which equals the sum of one‑third of the area of each incident triangle.
     *
     * @tparam Complex SimplicialComplex<2> type.
     * @param mesh 2D triangle mesh.
     * @return Diagonal sparse matrix M (size = number of vertices).
     *
     * @note Currently uses Euclidean area from coordinates directly.
     *       For non‑Euclidean metrics, area should be computed via Heron's
     *       formula using metric edge lengths (see TODO in file header).
     */
    template<typename Complex>
    Eigen::SparseMatrix<typename Complex::scalar_type>
        build_lumped_mass_matrix(const Complex& mesh) {
        static_assert(Complex::Dimension == 2, "Lumped mass matrix only for 2D complexes");
        using Scalar = typename Complex::scalar_type;
        std::size_t nv = mesh.num_vertices();
        std::vector<Scalar> dual_volumes(nv, Scalar(0));

        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto tri = mesh.triangle_at(t);
            const auto& p0 = mesh.vertex(tri[0]);
            const auto& p1 = mesh.vertex(tri[1]);
            const auto& p2 = mesh.vertex(tri[2]);

            // Area = 0.5 * |(p1-p0) × (p2-p0)|
            Scalar cross = (p1.x() - p0.x()) * (p2.y() - p0.y()) -
                (p1.y() - p0.y()) * (p2.x() - p0.x());
            Scalar area = delta::abs(cross) / 2;
            Scalar one_third = area / 3;

            // Each vertex gets one‑third of the triangle area (barycentric dual)
            dual_volumes[tri[0]] += one_third;
            dual_volumes[tri[1]] += one_third;
            dual_volumes[tri[2]] += one_third;
        }

        std::vector<Eigen::Triplet<Scalar>> triplets;
        for (std::size_t i = 0; i < nv; ++i) {
            triplets.emplace_back(i, i, dual_volumes[i]);
        }

        Eigen::SparseMatrix<Scalar> M(nv, nv);
        M.setFromTriplets(triplets.begin(), triplets.end());
        return M;
    }

} // namespace delta::numerical

#endif // DELTA_NUMERICAL_COTANGENT_LAPLACIAN_H