// include/delta/geometry/hat_basis.h
// include/delta/geometry/hat_basis.h
// ============================================================================
// HAT BASIS FUNCTIONS – PIECEWISE LINEAR LAGRANGE BASIS ON SIMPLICIAL MESHES
// ============================================================================
//
// 1.  MATHEMATICAL DEFINITION
// ---------------------------
// For a simplicial complex (triangles in 2D, tetrahedra in 3D) the hat (or
// “hat”) function φ_v associated with vertex v is the unique continuous,
// piecewise linear function such that:
//      φ_v(v) = 1   and   φ_v(other vertices) = 0.
// On any top‑dimensional simplex, φ_v coincides with the barycentric
// coordinate λ_v.  The set {φ_v} forms a partition of unity:
//      Σ_{v} φ_v(p) = 1   for every point p in the mesh.
//
//
// 2.  WHY ORIENTATION MATTERS – THE TRAP WITH ABSOLUTE AREA
// ----------------------------------------------------------
// Many naive implementations compute triangle areas using `abs(cross) / 2`.
// This destroys the sign of the area, which is essential for two reasons:
//
//   2.1  Gradients of hat functions depend on the sign.
//        For a triangle with vertices v0, v1, v2 in counter‑clockwise order,
//        the (oriented) area is positive.  The gradient of λ0 is:
//            ∇λ0 = ( (v2 - v1)⊥ ) / (2 * area)               (1)
//        where (dx,dy)⊥ = (dy, -dx).  If you replace area by |area|,
//        the sign of ∇λ0 flips when the orientation is clockwise.
//        The resulting gradient would point in the opposite direction,
//        breaking the identity ∇λ0 + ∇λ1 + ∇λ2 = 0 and ruining any
//        differential operator that relies on consistent orientation.
//
//   2.2  Inside / outside tests via barycentric coordinates.
//        Barycentric coordinates are defined as ratios of oriented areas:
//            λ_i = oriented_area(p, v_j, v_k) / oriented_area(v0, v1, v2).
//        With unsigned areas a point inside the triangle yields positive
//        coordinates, but s point outside gives at least one negative
//        coordinate.  If you use absolute values, the test becomes
//        meaningless – you can no longer locate a point reliably.
//
//   => In both cases **oriented (signed) area** is mandatory.
//      Using `abs` produces silent, hard‑to‑debug errors (wrong sign of
//      gradient, incorrect simplex location).
//
//
// 3.  THE CORRECT GRADIENT FORMULAS (2D)
// ---------------------------------------
// Let area = 0.5 * ((v1 - v0) × (v2 - v0))   (2D cross product = scalar).
// Then:
//      ∇λ0 = ( (v2 - v1)⊥ ) / (2 * area),   where (x,y)⊥ = (y, -x)
//      ∇λ1 = ( (v0 - v2)⊥ ) / (2 * area)
//      ∇λ2 = ( (v1 - v0)⊥ ) / (2 * area)
//
// These vectors satisfy ∇λ0 + ∇λ1 + ∇λ2 = 0 and ∇λi · (vj - vi) = δij
// (Kronecker delta) along edges.
//
// For the triangle (0,0)-(1,0)-(0,1) oriented counter‑clockwise, we have:
//   area = +0.5, ∇λ0 = (-1,-1), ∇λ1 = (1,0), ∇λ2 = (0,1).
// If you used the opposite rotation ( -⊥ ), you would obtain (1,1) for ∇λ0,
// which is the gradient of 1-x-y but with a sign error – the function would
// still be barycentric, but the sign would depend on orientation.
//
// In the current implementation we use the rotation (dx,dy) → (-dy, dx).
// This is the +90° rotation, which together with the signed area gives the
// correct gradient for any orientation.
//
//
// 4.  THE CORRECT GRADIENT FORMULAS (3D)
// ---------------------------------------
// For tetrahedron with vertices v0,v1,v2,v3 and signed volume
//      vol = ( (v1-v0) × (v2-v0) )·(v3-v0) / 6,
// the gradient of λ0 is:
//      ∇λ0 = - ( (v2-v1) × (v3-v1) ) / (6 vol)
// and cyclically for the other vertices.  Again, the sign is crucial.
//
//
// 5.  BARYCENTRIC COORDINATES VIA ORIENTED AREAS / VOLUMES
// ---------------------------------------------------------
// For a point p and a triangle (v0,v1,v2):
//      λ0 = orient(p, v1, v2) / orient(v0, v1, v2)
//      λ1 = orient(v0, p, v2) / orient(v0, v1, v2)
//      λ2 = orient(v0, v1, p) / orient(v0, v1, v2)
// with orient(a,b,c) = (b - a) × (c - a)  (scalar, signed).
//
// For a tetrahedron (v0,v1,v2,v3):
//      λi = signed_vol(p, ... ) / signed_vol(v0,v1,v2,v3),
// where the numerator is the signed volume of the sub‑tetrahedron formed
// by replacing vi with p.
//
// Because we use oriented quantities, p is inside the simplex iff all λi
// are in [0,1] (up to a small tolerance).  This is exactly what locate_point()
// does.
//
//
// 6.  WHY EPSILON MUST BE USER‑CONTROLLABLE
// ------------------------------------------
// The tolerance eps controls how far outside the [0,1] range a barycentric
// coordinate can be and still be considered “inside”.  Hard‑coding eps = 1e-6
// is wrong because:
//   * The global default epsilon of the library may be much smaller (1e-30).
//   * Different applications need different tolerances.
//   * On extremely coarse meshes or with large rational numbers 1e-6 may be
//     too big; on very fine meshes it may be too small.
//
// Therefore the constructor accepts an explicit eps parameter that defaults
// to delta::default_eps().  This follows the library’s design principle:
// all numerical tolerances are centralised and adjustable.
//
//
// 7.  COMMON PITFALLS AND HOW TO AVOID THEM
// ------------------------------------------
// ✘ Using `abs(area)` => gradients have wrong sign, location unreliable.
// ✘ Using `(dy, -dx)` rotation instead of `(-dy, dx)` ⇒ gradients swapped
//    for triangles of opposite orientation.
// ✘ Hard‑coding tolerance or using a global static variable ⇒ inflexible,
//    breaks multi‑threading and custom precision requirements.
// ✘ Forgetting to normalise gradients by `2*area` (2D) or `6*vol` (3D).
// ✘ Assuming vertices are always stored in counter‑clockwise order.
//    Our formulas work for any orientation because the signed area changes
//    sign accordingly, keeping the resulting λi correct.
//
//
// 8.  VERIFICATION – HOW TO TEST THAT YOUR IMPLEMENTATION IS CORRECT
// --------------------------------------------------------------------
// The test suite checks:
//   • Interpolation of a linear function (exact up to rational arithmetic).
//   • Evaluation at vertices (1 for own vertex, 0 for others).
//   • Gradients on a known triangle: ∇λ0 = (-1,-1), ∇λ1 = (1,0), ∇λ2 = (0,1).
//   • locate_point() correctly identifies the containing simplex.
//   • Barycentric coordinates sum to 1 (within eps).
//
// If any of these tests fail, the most likely cause is an orientation / sign
// mistake or the improper use of absolute area.
//
// ============================================================================
#ifndef DELTA_GEOMETRY_HAT_BASIS_H
#define DELTA_GEOMETRY_HAT_BASIS_H

#include <vector>
#include <optional>
#include "delta/geometry/simplicial_complex.h"
#include "delta/geometry/constructive_core.h"
#include "delta/rational/context.h"

namespace delta::geometry {

    template<typename Complex>
    class HatBasis {
    public:
        using point_type = typename Complex::point_type;
        using scalar_type = typename Complex::scalar_type;
        using vertex_index = typename Complex::vertex_index;
        using vector_type = Vector<scalar_type, point_type::RowsAtCompileTime>;

        static constexpr int Dim = point_type::RowsAtCompileTime;
        static_assert(Dim == 2 || Dim == 3, "HatBasis only for 2D or 3D");

        explicit HatBasis(const Complex& mesh,
            const scalar_type& eps = delta::default_eps())
            : mesh_(mesh), eps_(eps) {
            precompute_gradients();
        }

        scalar_type evaluate(vertex_index v, const point_type& p) const {
            auto loc = locate_point(p);
            if (!loc) return scalar_type(0);
            const auto& [simp_key, bary] = *loc;
            int dim = simp_key.first;
            std::size_t idx = simp_key.second;
            const auto& vertices = mesh_.get_simplex(dim, idx);
            for (std::size_t i = 0; i < vertices.size(); ++i) {
                if (vertices[i] == v) return bary[i];
            }
            return scalar_type(0);
        }

        point_type gradient(vertex_index v, const point_type& p) const {
            auto loc = locate_point(p);
            if (!loc) return point_type::Zero();
            const auto& [simp_key, bary] = *loc;
            int dim = simp_key.first;
            std::size_t idx = simp_key.second;
            const auto& vertices = mesh_.get_simplex(dim, idx);
            for (std::size_t i = 0; i < vertices.size(); ++i) {
                if (vertices[i] == v) {
                    if constexpr (Dim == 2) {
                        return grad2d_[idx].col(i);
                    }
                    else {
                        return grad3d_[idx].col(i);
                    }
                }
            }
            return point_type::Zero();
        }

        scalar_type interpolate(const point_type& p, const std::vector<scalar_type>& vertex_values) const {
            auto loc = locate_point(p);
            if (!loc) return scalar_type(0);
            const auto& [simp_key, bary] = *loc;
            int dim = simp_key.first;
            std::size_t idx = simp_key.second;
            const auto& vertices = mesh_.get_simplex(dim, idx);
            scalar_type result = 0;
            for (std::size_t i = 0; i < vertices.size(); ++i) {
                result += bary[i] * vertex_values[vertices[i]];
            }
            return result;
        }

        std::optional<std::pair<std::pair<int, std::size_t>, std::vector<scalar_type>>>
            locate_point(const point_type& p) const {
            const int top_dim = Dim;
            for (std::size_t idx = 0; idx < mesh_.num_simplices(top_dim); ++idx) {
                const auto& vertices = mesh_.get_simplex(top_dim, idx);
                std::vector<scalar_type> bary = barycentric_coordinates(p, vertices);
                bool inside = true;
                for (scalar_type coord : bary) {
                    if (coord < -eps_ || coord > 1 + eps_) {
                        inside = false;
                        break;
                    }
                }
                if (inside) {
                    for (auto& coord : bary) {
                        if (coord < 0 && coord > -eps_) coord = 0;
                        if (coord > 1 && coord < 1 + eps_) coord = 1;
                    }
                    return std::make_pair(std::make_pair(top_dim, idx), std::move(bary));
                }
            }
            return std::nullopt;
        }

    private:
        const Complex& mesh_;
        scalar_type eps_;

        std::vector<Eigen::Matrix<scalar_type, Dim, Dim + 1>> grad2d_;
        std::vector<Eigen::Matrix<scalar_type, Dim, Dim + 1>> grad3d_;

        void precompute_gradients() {
            if constexpr (Dim == 2) {
                grad2d_.resize(mesh_.num_triangles());
                for (std::size_t t = 0; t < mesh_.num_triangles(); ++t) {
                    auto tri = mesh_.triangle_at(t);
                    point_type v0 = mesh_.vertex(tri[0]);
                    point_type v1 = mesh_.vertex(tri[1]);
                    point_type v2 = mesh_.vertex(tri[2]);

                    vector_type e1 = v1 - v0;
                    vector_type e2 = v2 - v0;
                    scalar_type cross_val = e1.x() * e2.y() - e1.y() * e2.x();
                    scalar_type area = cross_val / 2;
                    if (area == 0) continue;

                    vector_type v2v1 = v2 - v1;
                    point_type grad0;
                    grad0 << -v2v1.y(), v2v1.x();      // поворот на +90°
                    grad0 /= (2 * area);

                    vector_type v0v2 = v0 - v2;
                    point_type grad1;
                    grad1 << -v0v2.y(), v0v2.x();
                    grad1 /= (2 * area);

                    vector_type v1v0 = v1 - v0;
                    point_type grad2;
                    grad2 << -v1v0.y(), v1v0.x();
                    grad2 /= (2 * area);

                    grad2d_[t].resize(Dim, 3);
                    grad2d_[t].col(0) = grad0;
                    grad2d_[t].col(1) = grad1;
                    grad2d_[t].col(2) = grad2;
                }
            }
            else if constexpr (Dim == 3) {
                grad3d_.resize(mesh_.num_tetrahedra());
                for (std::size_t tet = 0; tet < mesh_.num_tetrahedra(); ++tet) {
                    auto t = mesh_.tetrahedron_at(tet);
                    point_type v0 = mesh_.vertex(t[0]);
                    point_type v1 = mesh_.vertex(t[1]);
                    point_type v2 = mesh_.vertex(t[2]);
                    point_type v3 = mesh_.vertex(t[3]);

                    vector_type v10 = v1 - v0;
                    vector_type v20 = v2 - v0;
                    vector_type v30 = v3 - v0;
                    scalar_type vol = v10.dot(v20.cross(v30)) / 6;
                    if (vol == 0) continue;

                    vector_type v21 = v2 - v1;
                    vector_type v31 = v3 - v1;

                    vector_type grad0_vec = -v21.cross(v31) / (6 * vol);
                    vector_type grad1_vec = v20.cross(v30) / (6 * vol);
                    vector_type grad2_vec = v10.cross(v30) / (6 * vol);
                    vector_type grad3_vec = -v10.cross(v20) / (6 * vol);

                    grad3d_[tet].resize(Dim, 4);
                    grad3d_[tet].col(0) = grad0_vec.data();
                    grad3d_[tet].col(1) = grad1_vec.data();
                    grad3d_[tet].col(2) = grad2_vec.data();
                    grad3d_[tet].col(3) = grad3_vec.data();
                }
            }
        }

        std::vector<scalar_type> barycentric_coordinates(const point_type& p, const std::vector<vertex_index>& vertices) const {
            if constexpr (Dim == 2) {
                const point_type& v0 = mesh_.vertex(vertices[0]);
                const point_type& v1 = mesh_.vertex(vertices[1]);
                const point_type& v2 = mesh_.vertex(vertices[2]);

                auto orient = [](const point_type& a, const point_type& b, const point_type& c) -> scalar_type {
                    return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
                    };
                scalar_type area_total = orient(v0, v1, v2);
                if (area_total == 0) return { 0, 0, 0 };

                scalar_type area0 = orient(p, v1, v2);
                scalar_type area1 = orient(v0, p, v2);
                scalar_type area2 = orient(v0, v1, p);
                return { area0 / area_total, area1 / area_total, area2 / area_total };
            }
            else { // Dim == 3
                const point_type& v0 = mesh_.vertex(vertices[0]);
                const point_type& v1 = mesh_.vertex(vertices[1]);
                const point_type& v2 = mesh_.vertex(vertices[2]);
                const point_type& v3 = mesh_.vertex(vertices[3]);

                auto signed_vol = [](const point_type& a, const point_type& b,
                    const point_type& c, const point_type& d) -> scalar_type {
                        vector_type ab = b - a;
                        vector_type ac = c - a;
                        vector_type ad = d - a;
                        return ab.dot(ac.cross(ad)) / 6;
                    };
                scalar_type vol_total = signed_vol(v0, v1, v2, v3);
                if (vol_total == 0) return { 0, 0, 0, 0 };

                scalar_type vol0 = signed_vol(p, v1, v2, v3);
                scalar_type vol1 = signed_vol(v0, p, v2, v3);
                scalar_type vol2 = signed_vol(v0, v1, p, v3);
                scalar_type vol3 = signed_vol(v0, v1, v2, p);
                return { vol0 / vol_total, vol1 / vol_total, vol2 / vol_total, vol3 / vol_total };
            }
        }
    };

} // namespace delta::geometry

#endif // DELTA_GEOMETRY_HAT_BASIS_H