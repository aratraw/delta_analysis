// include/delta/numerical/solvers/heat_solver.h
#pragma once

#include "delta/geometry/simplicial_complex.h"
#include "delta/numerical/fem_assemblers.h"
#include "delta/numerical/boundary_conditions.h"
#include "delta/numerical/concepts.h"
#include "delta/core/operational_function.h"
#include <Eigen/Sparse>
#include <functional>
#include <vector>
#include <unordered_map>

namespace delta::numerical {

    // -----------------------------------------------------------------------------
    // Helper: build vertex -> index map for fast lookup
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
    // Apply Dirichlet boundary conditions to matrix and RHS
    // -----------------------------------------------------------------------------
    template<typename Value, typename BC>
    void apply_dirichlet_to_system(Eigen::SparseMatrix<Value>& A,
        Eigen::Matrix<Value, Eigen::Dynamic, 1>& b,
        std::size_t n,
        const BC& bc,
        double t,
        const std::vector<std::size_t>& dof_map = {}) {
        using Index = int;
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t global_i = dof_map.empty() ? i : dof_map[i];
            BCType type;
            BCValue<Value> bc_val;
            if (!bc.get(global_i, type, bc_val)) continue;
            if (type == BCType::Dirichlet) {
                Value g = bc_val(t, global_i);
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
            // Neumann, Robin, Periodic not implemented here
        }
    }

    // -----------------------------------------------------------------------------
    // Explicit Euler for heat equation (Δ‑style)
    // -----------------------------------------------------------------------------
    template<typename Path, typename Value, typename SourceFunc, typename Metric, typename BC>
        requires FiniteElementGrid<typename Path::GridType>&&
    IsMetric<Metric, typename Path::GridType::point_type, typename Path::GridType::scalar_type>
        OperationalFunction<typename Path::GridType::point_type, Value, typename Path::GridType>
        solve_heat_explicit(
            const Path& path,
            const OperationalFunction<typename Path::GridType::point_type, Value, typename Path::GridType>& u0,
            const SourceFunc& source,        // source(t, vertex_index) -> Value
            Value alpha,
            double dt,
            std::size_t num_steps,
            const Metric& metric,
            const BC& bc)
    {
        using Complex = typename Path::GridType;
        using Point = typename Complex::point_type;
        const Complex& mesh = path.current_grid();
        std::size_t n = mesh.num_vertices();

        // Precompute vertex->index map for the final function
        auto vertex_to_idx = build_vertex_to_index_map(mesh);

        // Assemble mass and stiffness matrices (once) using metric
        auto M = assemble_mass_matrix(mesh, metric);
        auto K = assemble_stiffness_matrix(mesh, metric);
        auto lumpedM = lumped_mass_matrix(M); // diagonal lumped mass

        // Extract initial values into vector indexed by vertex
        std::vector<Value> u_vec(n);
        for (std::size_t i = 0; i < n; ++i) {
            u_vec[i] = u0(mesh.vertex(i));
        }

        // Time stepping loop
        for (std::size_t step = 0; step < num_steps; ++step) {
            double t = step * dt;
            double t_next = (step + 1) * dt;

            // RHS = M * source(t)
            Eigen::Matrix<Value, Eigen::Dynamic, 1> rhs(n);
            rhs.setZero();
            for (std::size_t i = 0; i < n; ++i) {
                Value si = source(t, i);
                for (typename Eigen::SparseMatrix<Value>::InnerIterator it(M, i); it; ++it) {
                    rhs(it.row()) += it.value() * si;
                }
            }

            // Compute -K * u
            Eigen::Matrix<Value, Eigen::Dynamic, 1> u_eig(n);
            for (std::size_t i = 0; i < n; ++i) u_eig(i) = u_vec[i];
            Eigen::Matrix<Value, Eigen::Dynamic, 1> Ku = K * u_eig;

            // Explicit update
            std::vector<Value> u_new_vec(n);
            for (std::size_t i = 0; i < n; ++i) {
                u_new_vec[i] = u_vec[i] + Value(dt) * (-alpha * Ku(i) / lumpedM(i) + rhs(i) / lumpedM(i));
            }

            // Apply Dirichlet BC (overwrite)
            for (std::size_t i = 0; i < n; ++i) {
                BCType type;
                BCValue<Value> bc_val;
                if (bc.get(i, type, bc_val) && type == BCType::Dirichlet) {
                    u_new_vec[i] = bc_val(t_next, i);
                }
            }

            u_vec = std::move(u_new_vec);
        }

        // Build final OperationalFunction (address = point, lookup via map)
        auto vertex_to_idx_copy = vertex_to_idx;
        auto u_vec_copy = u_vec;
        OperationalFunction<Point, Value, Complex> u_final(
            mesh,
            [vertex_to_idx_copy, u_vec_copy](const Point& addr) -> Value {
                auto it = vertex_to_idx_copy.find(addr);
                if (it == vertex_to_idx_copy.end()) {
                    throw std::out_of_range("Address not found in final solution");
                }
                return u_vec_copy[it->second];
            }
        );

        return u_final;
    }

    // -----------------------------------------------------------------------------
    // Implicit Euler for heat equation (Δ‑style)
    // -----------------------------------------------------------------------------
    template<typename Path, typename Value, typename SourceFunc, typename Metric, typename BC>
        requires FiniteElementGrid<typename Path::GridType>&&
    IsMetric<Metric, typename Path::GridType::point_type, typename Path::GridType::scalar_type>
        OperationalFunction<typename Path::GridType::point_type, Value, typename Path::GridType>
        solve_heat_implicit(
            const Path& path,
            const OperationalFunction<typename Path::GridType::point_type, Value, typename Path::GridType>& u0,
            const SourceFunc& source,
            Value alpha,
            double dt,
            std::size_t num_steps,
            const Metric& metric,
            const BC& bc)
    {
        using Complex = typename Path::GridType;
        using Point = typename Complex::point_type;
        using Index = int;
        const Complex& mesh = path.current_grid();
        std::size_t n = mesh.num_vertices();

        auto vertex_to_idx = build_vertex_to_index_map(mesh);

        // Precompute matrices (constant for this grid) using metric
        auto K = assemble_stiffness_matrix(mesh, metric);
        auto M = assemble_mass_matrix(mesh, metric);
        // System matrix A = M + αΔt K
        Eigen::SparseMatrix<Value> A = M;
        if (alpha != 0 && dt != 0) {
            A += alpha * dt * K;
        }

        // Extract initial values
        std::vector<Value> u_vec(n);
        for (std::size_t i = 0; i < n; ++i) {
            u_vec[i] = u0(mesh.vertex(i));
        }

        for (std::size_t step = 0; step < num_steps; ++step) {
            double t = step * dt;
            double t_next = (step + 1) * dt;

            // RHS = M * u^n + Δt * M * source(t+1)
            Eigen::Matrix<Value, Eigen::Dynamic, 1> rhs(n);
            rhs.setZero();
            // M * u^n
            for (std::size_t i = 0; i < n; ++i) {
                for (typename Eigen::SparseMatrix<Value>::InnerIterator it(M, i); it; ++it) {
                    rhs(it.row()) += it.value() * u_vec[it.col()];
                }
            }
            // Δt * M * source(t+1)
            for (std::size_t i = 0; i < n; ++i) {
                Value si = source(t_next, i);
                for (typename Eigen::SparseMatrix<Value>::InnerIterator it(M, i); it; ++it) {
                    rhs(it.row()) += dt * it.value() * si;
                }
            }

            // Apply Dirichlet BC (modifies a copy of A)
            Eigen::SparseMatrix<Value> A_mod = A;
            apply_dirichlet_to_system(A_mod, rhs, n, bc, t_next);

            // Solve
            Eigen::SparseLU<Eigen::SparseMatrix<Value>> solver;
            solver.compute(A_mod);
            if (solver.info() != Eigen::Success)
                throw std::runtime_error("Implicit heat solver: factorization failed");

            Eigen::Matrix<Value, Eigen::Dynamic, 1> x = solver.solve(rhs);
            if (solver.info() != Eigen::Success)
                throw std::runtime_error("Implicit heat solver: solution failed");

            u_vec.assign(x.data(), x.data() + n);
        }

        // Build final OperationalFunction
        auto vertex_to_idx_copy = vertex_to_idx;
        auto u_vec_copy = u_vec;
        OperationalFunction<Point, Value, Complex> u_final(
            mesh,
            [vertex_to_idx_copy, u_vec_copy](const Point& addr) -> Value {
                auto it = vertex_to_idx_copy.find(addr);
                if (it == vertex_to_idx_copy.end()) {
                    throw std::out_of_range("Address not found in final solution");
                }
                return u_vec_copy[it->second];
            }
        );

        return u_final;
    }

    // -----------------------------------------------------------------------------
    // Crank-Nicolson for heat equation (Δ‑style)
    // -----------------------------------------------------------------------------
    template<typename Path, typename Value, typename SourceFunc, typename Metric, typename BC>
        requires FiniteElementGrid<typename Path::GridType>&&
    IsMetric<Metric, typename Path::GridType::point_type, typename Path::GridType::scalar_type>
        OperationalFunction<typename Path::GridType::point_type, Value, typename Path::GridType>
        solve_heat_crank_nicolson(
            const Path& path,
            const OperationalFunction<typename Path::GridType::point_type, Value, typename Path::GridType>& u0,
            const SourceFunc& source,
            Value alpha,
            double dt,
            std::size_t num_steps,
            const Metric& metric,
            const BC& bc)
    {
        using Complex = typename Path::GridType;
        using Point = typename Complex::point_type;
        using Index = int;
        const Complex& mesh = path.current_grid();
        std::size_t n = mesh.num_vertices();

        auto vertex_to_idx = build_vertex_to_index_map(mesh);

        auto K = assemble_stiffness_matrix(mesh, metric);
        auto M = assemble_mass_matrix(mesh, metric);

        // Left matrix: L = M + (αΔt/2) K
        Eigen::SparseMatrix<Value> L = M;
        if (alpha != 0 && dt != 0) {
            L += (alpha * dt / 2) * K;
        }
        // Right matrix: R = M - (αΔt/2) K
        Eigen::SparseMatrix<Value> R = M;
        if (alpha != 0 && dt != 0) {
            R -= (alpha * dt / 2) * K;
        }

        // Extract initial values
        std::vector<Value> u_vec(n);
        for (std::size_t i = 0; i < n; ++i) {
            u_vec[i] = u0(mesh.vertex(i));
        }

        for (std::size_t step = 0; step < num_steps; ++step) {
            double t = step * dt;
            double t_next = (step + 1) * dt;

            // RHS = R * u^n + (Δt/2) M (source(t) + source(t+1))
            Eigen::Matrix<Value, Eigen::Dynamic, 1> rhs(n);
            rhs.setZero();
            // R * u^n
            for (std::size_t i = 0; i < n; ++i) {
                for (typename Eigen::SparseMatrix<Value>::InnerIterator it(R, i); it; ++it) {
                    rhs(it.row()) += it.value() * u_vec[it.col()];
                }
            }
            // (Δt/2) M (source(t) + source(t+1))
            for (std::size_t i = 0; i < n; ++i) {
                Value si_t = source(t, i);
                Value si_tp1 = source(t_next, i);
                Value avg = (si_t + si_tp1) / Value(2);
                for (typename Eigen::SparseMatrix<Value>::InnerIterator it(M, i); it; ++it) {
                    rhs(it.row()) += dt * it.value() * avg;
                }
            }

            // Apply Dirichlet BC
            Eigen::SparseMatrix<Value> L_mod = L;
            apply_dirichlet_to_system(L_mod, rhs, n, bc, t_next);

            // Solve
            Eigen::SparseLU<Eigen::SparseMatrix<Value>> solver;
            solver.compute(L_mod);
            if (solver.info() != Eigen::Success)
                throw std::runtime_error("Crank-Nicolson heat solver: factorization failed");

            Eigen::Matrix<Value, Eigen::Dynamic, 1> x = solver.solve(rhs);
            if (solver.info() != Eigen::Success)
                throw std::runtime_error("Crank-Nicolson heat solver: solution failed");

            u_vec.assign(x.data(), x.data() + n);
        }

        // Build final OperationalFunction
        auto vertex_to_idx_copy = vertex_to_idx;
        auto u_vec_copy = u_vec;
        OperationalFunction<Point, Value, Complex> u_final(
            mesh,
            [vertex_to_idx_copy, u_vec_copy](const Point& addr) -> Value {
                auto it = vertex_to_idx_copy.find(addr);
                if (it == vertex_to_idx_copy.end()) {
                    throw std::out_of_range("Address not found in final solution");
                }
                return u_vec_copy[it->second];
            }
        );

        return u_final;
    }

} // namespace delta::numerical