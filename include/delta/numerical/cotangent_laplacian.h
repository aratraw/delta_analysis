// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0


// include/delta/numerical/cotangent_laplacian.h
#ifndef DELTA_NUMERICAL_COTANGENT_LAPLACIAN_H
#define DELTA_NUMERICAL_COTANGENT_LAPLACIAN_H

#include <Eigen/Sparse>
#include <vector>
#include "delta/geometry/simplicial_complex.h"
#include "delta/geometry/dual_complex.h"
#include "delta/core/regulative_idea.h"

namespace delta::numerical {

    /**
     * @brief Build the cotangent Laplacian matrix for a 2D simplicial complex.
     *
     * For each triangle with vertices i,j,k, let a=length(edge jk), b=length(edge ki), c=length(edge ij).
     * The area A is given by Heron's formula.
     * The cotangents are:
     *   cot(∠i) = (b² + c² - a²) / (4A)
     *   cot(∠j) = (c² + a² - b²) / (4A)
     *   cot(∠k) = (a² + b² - c²) / (4A)
     *
     * The cotangent Laplacian matrix L is defined as:
     *   L_{ii} = Σ_{j≠i} w_{ij}
     *   L_{ij} = -w_{ij}   for i≠j
     * where w_{ij} = (cot(γ_ij) + cot(δ_ij)) / 2, with γ_ij and δ_ij being the angles opposite edge (i,j)
     * in the two incident triangles (for boundary edges, only one triangle contributes).
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

        // Helper to get squared edge length
        auto sq_len = [&](VertexIndex i, VertexIndex j) -> Scalar {
            Scalar d = metric(mesh.vertex(i), mesh.vertex(j));
            return d * d;
            };

        // Process each triangle
        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto tri = mesh.triangle_at(t);
            VertexIndex i = tri[0], j = tri[1], k = tri[2];

            // Squared edge lengths
            Scalar a2 = sq_len(j, k);   // opposite i
            Scalar b2 = sq_len(k, i);   // opposite j
            Scalar c2 = sq_len(i, j);   // opposite k

            // Area via Heron
            Scalar a = delta::sqrt(a2);
            Scalar b = delta::sqrt(b2);
            Scalar c = delta::sqrt(c2);
            Scalar s = (a + b + c) / 2;
            Scalar area = delta::sqrt(s * (s - a) * (s - b) * (s - c));
            if (area == 0) continue;

            // Cotangents (divided by 2 for the final weight, we keep full cot and will scale later)
            Scalar cot_i = (b2 + c2 - a2) / (4 * area);
            Scalar cot_j = (c2 + a2 - b2) / (4 * area);
            Scalar cot_k = (a2 + b2 - c2) / (4 * area);

            // For each edge, add contribution to stiffness matrix
            // Edge (i,j): opposite angle k → cot_k
            // Edge (j,k): opposite angle i → cot_i
            // Edge (k,i): opposite angle j → cot_j

            // Off-diagonals get - (cot/2), diagonals accumulate + (cot/2)
            auto add_edge = [&](VertexIndex u, VertexIndex v, Scalar cot) {
                // u < v for consistent ordering (not strictly needed for triplets, but helps)
                triplets.emplace_back(u, v, -cot / 2);
                triplets.emplace_back(v, u, -cot / 2);
                triplets.emplace_back(u, u, cot / 2);
                triplets.emplace_back(v, v, cot / 2);
                };

            add_edge(i, j, cot_k);
            add_edge(j, k, cot_i);
            add_edge(k, i, cot_j);
        }

        Eigen::SparseMatrix<Scalar> L(nv, nv);
        L.setFromTriplets(triplets.begin(), triplets.end());
        return L;
    }

    /**
     * @brief Build the lumped mass matrix for a 2D simplicial complex.
     *
     * The lumped mass matrix M is diagonal, with M_{ii} = dual_volume(vertex i),
     * i.e., the area of the barycentric dual cell surrounding vertex i.
     * This is the sum of one‑third of the area of each triangle incident to the vertex.
     *
     * @tparam Complex SimplicialComplex<2> type.
     * @param mesh 2D triangle mesh.
     * @return Diagonal sparse matrix M (size = number of vertices).
     */
    template<typename Complex>
    Eigen::SparseMatrix<typename Complex::scalar_type>
        build_lumped_mass_matrix(const Complex& mesh) {
        static_assert(Complex::Dimension == 2, "Lumped mass matrix only for 2D complexes");
        using Scalar = typename Complex::scalar_type;
        std::size_t nv = mesh.num_vertices();
        std::vector<Scalar> dual_volumes(nv, Scalar(0));

        EuclideanMetric euclid; // assumption for area – but mesh vertices have coordinates, so we can compute area directly without metric
        // Actually we should use the metric, but for Euclidean mesh, EuclideanMetric gives correct edge lengths.
        // For general metric, we'd need to compute area via Heron using metric lengths.
        // Since mesh has coordinates, we can compute area directly from coordinates (which are rational).
        // But to be consistent, we use the same metric as passed to Laplacian? However mass matrix doesn't require metric argument in spec.
        // We'll compute area from vertex coordinates (assuming Euclidean embedding).
        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto tri = mesh.triangle_at(t);
            const auto& p0 = mesh.vertex(tri[0]);
            const auto& p1 = mesh.vertex(tri[1]);
            const auto& p2 = mesh.vertex(tri[2]);
            // area = 0.5 * |(p1-p0) x (p2-p0)|
            Scalar cross = (p1.x() - p0.x()) * (p2.y() - p0.y()) - (p1.y() - p0.y()) * (p2.x() - p0.x());
            Scalar area = delta::abs(cross) / 2;
            Scalar one_third = area / 3;
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