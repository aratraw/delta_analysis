// include/delta/numerical/solvers/poisson_solver.h
#pragma once

#include "delta/geometry/simplicial_complex.h"
#include "delta/numerical/fem_assemblers.h"
#include "delta/numerical/boundary_conditions.h"
#include "delta/core/operational_function.h"
#include <Eigen/Sparse>
#include <vector>
#include <unordered_map>

namespace delta::numerical {

    // -----------------------------------------------------------------------------
    // Concept for a finite element grid (reused from heat_solver)
    // -----------------------------------------------------------------------------
    template<typename G>
    concept FiniteElementGrid = requires(G g, std::size_t i) {
        typename G::point_type;
        { g.num_vertices() } -> std::convertible_to<std::size_t>;
        { g.vertex(i) } -> std::same_as<typename G::point_type>;
        { g.num_triangles() } -> std::convertible_to<std::size_t>;
        { g.triangle_at(i) } -> std::same_as<typename G::triangle_type>;
    };

    // -----------------------------------------------------------------------------
    // Helper: build vertex -> index map (copied from heat_solver)
    // -----------------------------------------------------------------------------
    template<typename Complex>
    std::unordered_map<typename Complex::point_type, std::size_t>
        build_vertex_to_index_map(const Complex& mesh) {
        using Point = typename Complex::point_type;
        std::unordered_map<Point, std::size_t> map;
        map.reserve(mesh.num_vertices());
        for (std::size_t i = 0; i < mesh.num_vertices(); ++i) {
            map[mesh.vertex(i)] = i;
        }
        return map;
    }

    // -----------------------------------------------------------------------------
    // Poisson solver for simplicial meshes (2D/3D) using FEM (Δ‑style)
    // Solves -Δu = f  with Dirichlet BC.
    // Returns solution as OperationalFunction on vertices.
    // -----------------------------------------------------------------------------
    template<typename Path, typename Value, typename Metric, typename BC>
        requires FiniteElementGrid<typename Path::GridType>
    OperationalFunction<typename Path::GridType::point_type, Value, typename Path::GridType>
        solve_poisson(
            const Path& path,
            const OperationalFunction<typename Path::GridType::point_type, Value, typename Path::GridType>& rhs,
            const BC& bc,
            const Metric& metric,
            double time = 0.0)   // time for time-dependent BC (though Poisson is static)
    {
        using Complex = typename Path::GridType;
        using Point = typename Complex::point_type;
        using Index = int;
        const Complex& mesh = path.current_grid();
        std::size_t n = mesh.num_vertices();

        // Precompute vertex->index map for final function
        auto vertex_to_idx = build_vertex_to_index_map(mesh);

        // Assemble stiffness matrix K and mass matrix M
        auto K = assemble_stiffness_matrix(mesh);   // cotangent Laplacian
        auto M = assemble_mass_matrix(mesh);

        // Right-hand side vector: b = M * f (where f is the rhs function)
        Eigen::Matrix<Value, Eigen::Dynamic, 1> b(n);
        b.setZero();
        for (std::size_t i = 0; i < n; ++i) {
            Value f_i = rhs(mesh.vertex(i));
            for (typename Eigen::SparseMatrix<Value>::InnerIterator it(M, i); it; ++it) {
                b(it.row()) += it.value() * f_i;
            }
        }

        // System matrix A = K (since we solve K u = M f)
        Eigen::SparseMatrix<Value> A = K;

        // Apply boundary conditions (Dirichlet only)
        // We modify A and b in place, but we need a copy of A to avoid destroying original.
        // Since we won't reuse A, we can modify it directly.
        for (std::size_t i = 0; i < n; ++i) {
            BCType type;
            BCValue<Value> bc_val;
            if (!bc.get(i, type, bc_val)) continue;
            if (type == BCType::Dirichlet) {
                Value g = bc_val(time, i);
                // Zero out row i
                for (int k = 0; k < A.outerSize(); ++k) {
                    for (typename Eigen::SparseMatrix<Value>::InnerIterator it(A, k); it; ++it) {
                        if (it.row() == static_cast<Index>(i)) {
                            it.valueRef() = 0;
                        }
                    }
                }
                A.coeffRef(static_cast<Index>(i), static_cast<Index>(i)) = 1.0;
                b(i) = g;
            }
            // Neumann/Robin not implemented (would require modification)
        }

        // Solve linear system
        Eigen::SparseLU<Eigen::SparseMatrix<Value>> solver;
        solver.compute(A);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Poisson solver: matrix factorization failed");
        }
        Eigen::Matrix<Value, Eigen::Dynamic, 1> x = solver.solve(b);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Poisson solver: solution failed");
        }

        // Extract solution vector
        std::vector<Value> u_vec(n);
        for (std::size_t i = 0; i < n; ++i) {
            u_vec[i] = x(static_cast<Index>(i));
        }

        // Build final OperationalFunction
        auto vertex_to_idx_copy = vertex_to_idx;
        auto u_vec_copy = u_vec;
        OperationalFunction<Point, Value, Complex> u_final(
            mesh,
            [vertex_to_idx_copy, u_vec_copy](const Point& addr) -> Value {
                auto it = vertex_to_idx_copy.find(addr);
                if (it == vertex_to_idx_copy.end()) {
                    throw std::out_of_range("Address not found in Poisson solution");
                }
                return u_vec_copy[it->second];
            }
        );

        return u_final;
    }

} // namespace delta::numerical