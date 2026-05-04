// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/geometry/dual_complex.h
// ===========================================================================
// BARYCENTRIC DUAL COMPLEX FOR SIMPLICIAL MESHES (2D AND 3D)
// ============================================================================
//
// This file implements the barycentric dual complex for a given primal
// simplicial complex. The dual is built by assigning to each primal simplex
// a dual cell of complementary dimension, constructed using barycentres
// (centroids) of simplices.
//
// ----------------------------------------------------------------------------
// DUALITY MAPPINGS
// ----------------------------------------------------------------------------
//
// For a simplicial complex of dimension Dim:
//   - Vertex (0‑simplex)           ↔ n‑cell (volume in dimension Dim)
//   - Edge (1‑simplex)             ↔ (n‑1)-cell (length in 3D, length in 2D)
//   - Triangle (2‑simplex)         ↔ (n‑2)-cell (length in 3D, point in 2D)
//   - Tetrahedron (3‑simplex, 3D)  ↔ 0‑cell (point)
//
// The dual complex provides:
//   - dual_volume(dim, idx): volume (or measure) of the dual cell for a given
//     primal simplex. For vertices, this is the area of the dual polygon (2D)
//     or volume of the dual polyhedron (3D). For higher‑dim simplices,
//     dimensions decrease accordingly.
//   - primal_to_dual(dim, primal_idx): maps a primal simplex to its dual cell.
//   - dual_to_primal(dim, dual_idx): inverse mapping (for consistency).
//
// ----------------------------------------------------------------------------
// GEOMETRIC CONSTRUCTION
// ----------------------------------------------------------------------------
//
// 2D (Triangulation):
//   - Vertex dual cell (2‑cell): polygon formed by barycentres of incident
//     triangles and edge midpoints. Volume = Σ (triangle area / 3).
//   - Edge dual cell (1‑cell): segment between barycentres of adjacent triangles
//     (or from barycentre to edge midpoint for boundary edges).
//   - Triangle dual cell (0‑cell): point (the triangle's barycentre),
//     volume = 1.
//
// 3D (Tetrahedralisation):
//   - Vertex dual cell (3‑cell): polyhedron formed by tetrahedron barycentres,
//     face barycentres, and edge midpoints. Volume = Σ (tetrahedron volume / 4).
//   - Edge dual cell (2‑cell): polygon whose vertices are tetrahedron
//     barycentres, face barycentres, and the edge midpoint.
//   - Face dual cell (1‑cell): segment between barycentres of incident tetrahedra
//     (or from barycentre to face centre for boundary faces).
//   - Tetrahedron dual cell (0‑cell): point (the tetrahedron's barycentre),
//     volume = 1.
//
// ----------------------------------------------------------------------------
// PROPERTIES
// ----------------------------------------------------------------------------
//
// - All dual cells are contained within the convex hull of the primal complex.
// - The mapping primal ↔ dual is a bijection (each primal simplex corresponds
//   to exactly one dual cell of complementary dimension).
// - The construction works for any metric (metric_ is used to compute distances
//   and areas/volumes).
//
// ----------------------------------------------------------------------------
// NOTES ON ACCURACY
// ----------------------------------------------------------------------------
//
// - For 3D edge dual volumes, a polygon triangulation approximation is used
//   (splitting into triangles from tetrahedron barycentre to face barycentres
//   to edge midpoint). This is exact for convex cells.
// - For boundary edges/faces, the dual cell extends only to the boundary
//   (using edge midpoint or face centre instead of a second tetrahedron
//   barycentre).
//
// ----------------------------------------------------------------------------
// TODO: VORONOI (CIRCUMCENTRIC) DUAL
// ----------------------------------------------------------------------------
//
// The current implementation uses the barycentric dual exclusively.
// For applications requiring exact matching with the cotangent Laplacian
// (e.g., reproducing known DEC results), a circumcentric dual based on
// Voronoi cells should be added. This would require:
//   - Computing circumcentres of triangles (2D) and tetrahedra (3D).
//   - Handling obtuse simplices where circumcentres lie outside the simplex
//     (e.g., using a mixed barycentric/circumcentric dual).
//   - Adjusting dual volumes accordingly.
//
// This is a separate header (e.g., voronoi_dual_complex.h) that could be
// added in the future. The interface would be identical to DualComplex,
// allowing drop‑in replacement in templates.
//
// ============================================================================

#ifndef DELTA_GEOMETRY_DUAL_COMPLEX_H
#define DELTA_GEOMETRY_DUAL_COMPLEX_H

#include <vector>
#include <unordered_map>
#include <optional>
#include <cmath>
#include "delta/geometry/simplicial_complex.h"
#include "delta/core/regulative_idea.h"
#include "delta/rational/transcendentals.h"

namespace delta::geometry {

    /**
     * @class DualComplex
     * @brief Barycentric dual complex for a simplicial mesh.
     *
     * Provides dual volumes and primal↔dual mappings for all dimensions.
     *
     * @tparam PrimalComplex Simplicial complex type (must satisfy SimplicialComplex concept).
     * @tparam Metric Metric type for computing distances and volumes.
     */
    template<typename PrimalComplex, typename Metric>
    class DualComplex {
    public:
        using scalar_type = typename PrimalComplex::scalar_type;
        using point_type = typename PrimalComplex::point_type;
        using vertex_index = typename PrimalComplex::vertex_index;
        using edge_type = typename PrimalComplex::edge_type;
        using triangle_type = typename PrimalComplex::triangle_type;
        using tetrahedron_type = typename PrimalComplex::tetrahedron_type;

        static constexpr int Dim = point_type::RowsAtCompileTime;

        /**
         * @brief Construct a barycentric dual complex from a primal mesh.
         * @param primal The primal simplicial complex.
         * @param metric The metric used for geometric computations.
         */
        DualComplex(const PrimalComplex& primal, const Metric& metric)
            : primal_(primal), metric_(metric) {
            build();
        }

        /**
         * @brief Number of dual cells of a given dimension.
         * @param dim Dimensionality of dual cells (0..Dim).
         * @return Number of dual cells.
         */
        std::size_t num_cells(int dim) const {
            if (dim < 0 || dim > Dim) return 0;
            if (dim >= static_cast<int>(dual_volumes_.size())) return 0;
            return dual_volumes_[dim].size();
        }

        /**
         * @brief Volume (measure) of the dual cell for a given primal simplex.
         * @param dim Dimension of the primal simplex (0..Dim).
         * @param idx Index of the primal simplex.
         * @return Volume of the corresponding dual cell.
         * @throws std::out_of_range if dim or idx are out of bounds.
         */
        scalar_type dual_volume(int dim, std::size_t idx) const {
            if (dim < 0 || dim > Dim || idx >= dual_volumes_[dim].size())
                throw std::out_of_range("DualComplex::dual_volume");
            return dual_volumes_[dim][idx];
        }

        /**
         * @brief Map a primal simplex to its dual cell index.
         * @param dim Dimension of the primal simplex.
         * @param primal_idx Index of the primal simplex.
         * @return Index of the dual cell.
         */
        std::size_t primal_to_dual(int dim, std::size_t primal_idx) const {
            if (dim < 0 || dim > Dim || primal_idx >= primal_to_dual_[dim].size())
                throw std::out_of_range("DualComplex::primal_to_dual");
            return primal_to_dual_[dim][primal_idx];
        }

        /**
         * @brief Map a dual cell back to its primal simplex.
         * @param dim Dimension of the primal simplex (not the dual cell!).
         * @param dual_idx Index of the dual cell.
         * @return Index of the primal simplex.
         * @note The mapping is a bijection: dual_to_primal(dim, primal_to_dual(dim, i)) == i.
         */
        std::size_t dual_to_primal(int dim, std::size_t dual_idx) const {
            if (dim < 0 || dim > Dim || dual_idx >= dual_to_primal_[dim].size())
                throw std::out_of_range("DualComplex::dual_to_primal");
            return dual_to_primal_[dim][dual_idx];
        }

    private:
        const PrimalComplex& primal_;
        Metric metric_;

        std::vector<std::vector<scalar_type>> dual_volumes_;      // [dim][idx]
        std::vector<std::vector<std::size_t>> primal_to_dual_;    // [dim][primal_idx] -> dual_idx
        std::vector<std::vector<std::size_t>> dual_to_primal_;    // [dim][dual_idx] -> primal_idx

        void build() {
            if constexpr (Dim == 2) {
                build_2d();
            }
            else if constexpr (Dim == 3) {
                build_3d();
            }
            else {
                static_assert(Dim == 2 || Dim == 3,
                    "DualComplex implemented only for dimensions 2 and 3");
            }
        }

        /**
         * @brief Build the barycentric dual for a 2D triangulation.
         */
        void build_2d() {
            std::size_t nv = primal_.num_vertices();
            std::size_t ne = primal_.num_edges();
            std::size_t nt = primal_.num_triangles();

            dual_volumes_.resize(3);
            primal_to_dual_.resize(3);
            dual_to_primal_.resize(3);
            dual_volumes_[2].resize(nv);
            dual_volumes_[1].resize(ne);
            dual_volumes_[0].resize(nt);
            primal_to_dual_[0].resize(nv);
            primal_to_dual_[1].resize(ne);
            primal_to_dual_[2].resize(nt);
            dual_to_primal_[2].resize(nv);
            dual_to_primal_[1].resize(ne);
            dual_to_primal_[0].resize(nt);

            // Triangle barycentres
            std::vector<point_type> tri_center(nt);
            for (std::size_t t = 0; t < nt; ++t) {
                auto tri = primal_.triangle_at(t);
                tri_center[t] = (primal_.vertex(tri[0]) + primal_.vertex(tri[1]) + primal_.vertex(tri[2])) / 3_r;
            }

            // Edge midpoints and incident triangles
            struct EdgeRec { point_type mid; std::size_t left; std::optional<std::size_t> right; };
            std::vector<EdgeRec> edges(ne);
            for (std::size_t e = 0; e < ne; ++e) {
                auto [v0, v1] = primal_.edge_at(e);
                edges[e].mid = (primal_.vertex(v0) + primal_.vertex(v1)) / 2_r;
                auto nbrs = primal_.edge_neighbors_2d(e);
                edges[e].left = nbrs.first;
                edges[e].right = nbrs.second;
            }

            // Dual volumes for vertices (2‑cells) = sum(incident triangle area / 3)
            for (std::size_t v = 0; v < nv; ++v) {
                scalar_type area = 0;
                for (std::size_t t = 0; t < nt; ++t) {
                    auto tri = primal_.triangle_at(t);
                    if (tri[0] == v || tri[1] == v || tri[2] == v) {
                        area += triangle_area(primal_.vertex(tri[0]), primal_.vertex(tri[1]), primal_.vertex(tri[2])) / 3_r;
                    }
                }
                dual_volumes_[2][v] = area;
                primal_to_dual_[0][v] = v;
                dual_to_primal_[2][v] = v;
            }

            // Dual lengths for edges (1‑cells)
            for (std::size_t e = 0; e < ne; ++e) {
                const auto& rec = edges[e];
                if (rec.right.has_value()) {
                    dual_volumes_[1][e] = metric_(tri_center[rec.left], tri_center[*rec.right]);
                }
                else {
                    dual_volumes_[1][e] = metric_(tri_center[rec.left], rec.mid);
                }
                primal_to_dual_[1][e] = e;
                dual_to_primal_[1][e] = e;
            }

            // Dual 0‑cells for triangles (measure = 1)
            for (std::size_t t = 0; t < nt; ++t) {
                dual_volumes_[0][t] = 1;
                primal_to_dual_[2][t] = t;
                dual_to_primal_[0][t] = t;
            }
        }

        /**
         * @brief Build the barycentric dual for a 3D tetrahedralisation.
         */
        void build_3d() {
            std::size_t nv = primal_.num_vertices();
            std::size_t ne = primal_.num_edges();
            std::size_t nf = primal_.num_triangles();
            std::size_t nt = primal_.num_tetrahedra();

            dual_volumes_.resize(4);
            primal_to_dual_.resize(4);
            dual_to_primal_.resize(4);
            dual_volumes_[3].resize(nv);
            dual_volumes_[2].resize(ne);
            dual_volumes_[1].resize(nf);
            dual_volumes_[0].resize(nt);
            primal_to_dual_[0].resize(nv);
            primal_to_dual_[1].resize(ne);
            primal_to_dual_[2].resize(nf);
            primal_to_dual_[3].resize(nt);
            dual_to_primal_[3].resize(nv);
            dual_to_primal_[2].resize(ne);
            dual_to_primal_[1].resize(nf);
            dual_to_primal_[0].resize(nt);

            // Tetrahedra barycentres and volumes
            std::vector<point_type> tet_center(nt);
            std::vector<scalar_type> tet_vol(nt);
            for (std::size_t t = 0; t < nt; ++t) {
                auto tet = primal_.tetrahedron_at(t);
                tet_center[t] = (primal_.vertex(tet[0]) + primal_.vertex(tet[1]) +
                    primal_.vertex(tet[2]) + primal_.vertex(tet[3])) / 4_r;
                tet_vol[t] = tetrahedron_volume(primal_.vertex(tet[0]), primal_.vertex(tet[1]),
                    primal_.vertex(tet[2]), primal_.vertex(tet[3]));
            }

            // Face barycentres
            std::vector<point_type> face_center(nf);
            for (std::size_t f = 0; f < nf; ++f) {
                auto tri = primal_.triangle_at(f);
                face_center[f] = (primal_.vertex(tri[0]) + primal_.vertex(tri[1]) + primal_.vertex(tri[2])) / 3_r;
            }

            // Edge midpoints
            std::vector<point_type> edge_mid(ne);
            for (std::size_t e = 0; e < ne; ++e) {
                auto [v0, v1] = primal_.edge_at(e);
                edge_mid[e] = (primal_.vertex(v0) + primal_.vertex(v1)) / 2_r;
            }

            // Incident tetrahedra for each vertex
            std::vector<std::vector<std::size_t>> vertex_tets(nv);
            for (std::size_t t = 0; t < nt; ++t) {
                auto tet = primal_.tetrahedron_at(t);
                for (int i = 0; i < 4; ++i) vertex_tets[tet[i]].push_back(t);
            }

            // Incident tetrahedra for each edge
            std::vector<std::vector<std::size_t>> edge_tets(ne);
            for (std::size_t e = 0; e < ne; ++e) {
                auto [v0, v1] = primal_.edge_at(e);
                for (std::size_t t : vertex_tets[v0]) {
                    auto tet = primal_.tetrahedron_at(t);
                    bool has0 = false, has1 = false;
                    for (int i = 0; i < 4; ++i) {
                        if (tet[i] == v0) has0 = true;
                        if (tet[i] == v1) has1 = true;
                    }
                    if (has0 && has1) edge_tets[e].push_back(t);
                }
            }

            // Incident tetrahedra for each face
            std::vector<std::vector<std::size_t>> face_tets(nf);
            for (std::size_t t = 0; t < nt; ++t) {
                auto tet = primal_.tetrahedron_at(t);
                std::vector<vertex_index> tri0 = { tet[0], tet[1], tet[2] }; std::sort(tri0.begin(), tri0.end());
                std::vector<vertex_index> tri1 = { tet[0], tet[1], tet[3] }; std::sort(tri1.begin(), tri1.end());
                std::vector<vertex_index> tri2 = { tet[0], tet[2], tet[3] }; std::sort(tri2.begin(), tri2.end());
                std::vector<vertex_index> tri3 = { tet[1], tet[2], tet[3] }; std::sort(tri3.begin(), tri3.end());
                for (auto& tri : { tri0, tri1, tri2, tri3 }) {
                    auto fidx = primal_.find_simplex(2, tri);
                    if (fidx != -1) face_tets[static_cast<std::size_t>(fidx)].push_back(t);
                }
            }

            // Dual volumes for vertices (3‑cells) = sum(tetrahedron volume / 4)
            for (std::size_t v = 0; v < nv; ++v) {
                scalar_type vol = 0;
                for (std::size_t t : vertex_tets[v]) vol += tet_vol[t] / 4_r;
                dual_volumes_[3][v] = vol;
                primal_to_dual_[0][v] = v;
                dual_to_primal_[3][v] = v;
            }

            // Dual volumes for edges (2‑cells) – polygon area from incident tetrahedra
            for (std::size_t e = 0; e < ne; ++e) {
                scalar_type area = 0;
                for (std::size_t t : edge_tets[e]) {
                    // Find the two faces of tet t that contain edge e
                    std::vector<point_type> pts;
                    pts.push_back(tet_center[t]);
                    for (std::size_t f = 0; f < nf; ++f) {
                        auto tri = primal_.triangle_at(f);
                        bool has0 = false, has1 = false;
                        for (int i = 0; i < 3; ++i) {
                            if (tri[i] == primal_.edge_at(e)[0]) has0 = true;
                            if (tri[i] == primal_.edge_at(e)[1]) has1 = true;
                        }
                        if (has0 && has1) pts.push_back(face_center[f]);
                    }
                    pts.push_back(edge_mid[e]);
                    // Triangulate: tetrahedron barycentre, first face centre, edge midpoint
                    // and tetrahedron barycentre, second face centre, edge midpoint
                    if (pts.size() >= 4) {
                        area += triangle_area(pts[0], pts[1], pts[3]);
                        area += triangle_area(pts[0], pts[2], pts[3]);
                    }
                    else if (pts.size() == 3) {
                        area += triangle_area(pts[0], pts[1], pts[2]);
                    }
                }
                dual_volumes_[2][e] = area;
                primal_to_dual_[1][e] = e;
                dual_to_primal_[2][e] = e;
            }

            // Dual volumes for faces (1‑cells) – segment between barycentres of incident tetrahedra
            for (std::size_t f = 0; f < nf; ++f) {
                const auto& tets = face_tets[f];
                if (tets.size() == 2) {
                    dual_volumes_[1][f] = metric_(tet_center[tets[0]], tet_center[tets[1]]);
                }
                else if (tets.size() == 1) {
                    dual_volumes_[1][f] = metric_(tet_center[tets[0]], face_center[f]);
                }
                else {
                    dual_volumes_[1][f] = 0;
                }
                primal_to_dual_[2][f] = f;
                dual_to_primal_[1][f] = f;
            }

            // Dual 0‑cells for tetrahedra (measure = 1)
            for (std::size_t t = 0; t < nt; ++t) {
                dual_volumes_[0][t] = 1;
                primal_to_dual_[3][t] = t;
                dual_to_primal_[0][t] = t;
            }
        }

        /**
         * @brief Compute the area of a triangle given its vertices.
         * Uses Heron's formula with the given metric.
         */
        scalar_type triangle_area(const point_type& a, const point_type& b, const point_type& c) const {
            scalar_type ab = metric_(a, b);
            scalar_type bc = metric_(b, c);
            scalar_type ca = metric_(c, a);
            scalar_type s = (ab + bc + ca) / 2_r;
            using delta::sqrt;
            return sqrt(s * (s - ab) * (s - bc) * (s - ca));
        }

        /**
         * @brief Compute the volume of a tetrahedron given its vertices.
         * Uses the scalar triple product formula.
         */
        scalar_type tetrahedron_volume(const point_type& a, const point_type& b,
            const point_type& c, const point_type& d) const {
            Eigen::Matrix<scalar_type, 3, 1> ab = (b - a).data();
            Eigen::Matrix<scalar_type, 3, 1> ac = (c - a).data();
            Eigen::Matrix<scalar_type, 3, 1> ad = (d - a).data();
            using delta::abs;
            return abs(ab.cross(ac).dot(ad)) / 6_r;
        }
    };

} // namespace delta::geometry

#endif // DELTA_GEOMETRY_DUAL_COMPLEX_H