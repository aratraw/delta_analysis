// include/delta/numerical/solvers/advection_solver.h
#pragma once

#include "delta/core/path_concept.h"
#include "delta/geometry/simplicial_complex.h"
#include "delta/numerical/boundary_conditions.h"
#include "delta/numerical/concepts.h"           // содержит FiniteVolumeGrid2D
#include "delta/core/operational_function.h"
#include <vector>
#include <functional>
#include <optional>
#include <unordered_map>

namespace delta::numerical {

    // -----------------------------------------------------------------------------
    // Helper to build center->index map for triangles
    // -----------------------------------------------------------------------------
    template<typename Complex>
    std::unordered_map<typename Complex::point_type, std::size_t>
        build_center_to_index_map(const Complex& mesh) {
        using Point = typename Complex::point_type;
        std::unordered_map<Point, std::size_t> map;
        map.reserve(mesh.num_triangles());
        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto tri = mesh.triangle_at(t);
            Point a = mesh.vertex(tri[0]);
            Point b = mesh.vertex(tri[1]);
            Point c = mesh.vertex(tri[2]);
            Point center = (a + b + c) / typename Point::Scalar{ 3 };
            map[center] = t;
        }
        return map;
    }

    // -----------------------------------------------------------------------------
    // 2D Finite Volume upwind solver using Δ‑analysis concepts
    // -----------------------------------------------------------------------------
    template<typename Path, typename Value, typename VelocityFunc, typename Metric, typename BC>
        requires FiniteVolumeGrid2D<typename Path::GridType, Metric, typename Path::GridType::scalar_type>
    OperationalFunction<typename Path::GridType::point_type, Value, typename Path::GridType>
        solve_advection_upwind_2d(
            const Path& path,
            const OperationalFunction<typename Path::GridType::point_type, Value, typename Path::GridType>& u0,
            const VelocityFunc& velocity,                // function point_type -> point_type
            double dt,
            std::size_t num_steps,
            const Metric& metric,
            const BC& boundary_conds)
    {
        using Complex = typename Path::GridType;     // SimplicialComplex<2, ...>
        using Point = typename Complex::point_type;
        using Scalar = typename Point::Scalar;
        using Addr = Point;

        static_assert(Point::RowsAtCompileTime == 2, "2D solver requires 2D points");
        static_assert(std::is_constructible_v<Value, Scalar>, "Value must be constructible from Scalar");

        const Complex& mesh = path.current_grid();
        std::size_t n_tri = mesh.num_triangles();

        // Build center -> index map for fast lookup (used at the end)
        auto center_to_idx = build_center_to_index_map(mesh);

        // Precompute geometric data per triangle using metric
        std::vector<Scalar> cell_volume(n_tri);
        std::vector<Point> cell_center(n_tri);
        for (std::size_t t = 0; t < n_tri; ++t) {
            cell_volume[t] = mesh.cell_volume(t, metric);
            cell_center[t] = mesh.cell_center(t);
        }

        // Extract initial values into a vector indexed by triangle
        std::vector<Value> u_vec(n_tri);
        for (std::size_t t = 0; t < n_tri; ++t) {
            u_vec[t] = u0(cell_center[t]);
        }

        // Time stepping loop
        for (std::size_t step = 0; step < num_steps; ++step) {
            std::vector<Value> flux_accum(n_tri, Value{ 0 });

            // Loop over all edges
            for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
                // Geometric data from mesh using metric
                Scalar edge_len = mesh.edge_length(e, metric);
                if (edge_len == 0) continue;

                Point normal = mesh.edge_normal(e, metric); // oriented area vector (length = edge_len)
                Point edge_center = mesh.edge_center(e);
                Point vel = velocity(edge_center);
                Scalar vn = vel.dot(normal); // signed flux density

                // Get neighboring cells
                auto [left, right_opt] = mesh.edge_neighbors(e);

                if (right_opt.has_value()) {
                    // Internal edge
                    std::size_t right = *right_opt;
                    Value u_upwind = (vn >= 0) ? u_vec[left] : u_vec[right];
                    Value flux = Value(vn) * u_upwind; // already includes edge_len because normal has length
                    flux_accum[left] -= flux;
                    flux_accum[right] += flux;
                }
                else {
                    // Boundary edge: use boundary conditions (upwind)
                    Value flux = boundary_conds.boundary_flux(left, e, u_vec[left], vn);
                    flux_accum[left] -= flux; // sign consistent with internal edges
                }
            }

            // Update solution
            std::vector<Value> u_new_vec(n_tri);
            for (std::size_t t = 0; t < n_tri; ++t) {
                u_new_vec[t] = u_vec[t] + Value(dt) * flux_accum[t] / Value(cell_volume[t]);
            }
            u_vec = std::move(u_new_vec);
        }

        // Construct final OperationalFunction using center->index map
        auto center_to_idx_copy = center_to_idx;
        auto u_vec_copy = u_vec;
        OperationalFunction<Point, Value, Complex> u_final(
            mesh,
            [center_to_idx_copy, u_vec_copy](const Point& addr) -> Value {
                auto it = center_to_idx_copy.find(addr);
                if (it == center_to_idx_copy.end()) {
                    throw std::out_of_range("Address not found in final solution");
                }
                return u_vec_copy[it->second];
            }
        );

        return u_final;
    }

} // namespace delta::numerical