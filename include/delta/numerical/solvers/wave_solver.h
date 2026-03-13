// include/delta/numerical/solvers/wave_solver.h
#pragma once

#include "delta/geometry/simplicial_complex.h"
#include "delta/numerical/fem_assemblers.h"
#include "delta/numerical/boundary_conditions.h"
#include "delta/numerical/concepts.h"
#include "delta/core/operational_function.h"
#include <Eigen/Sparse>
#include <vector>
#include <unordered_map>
#include <cmath>

namespace delta::numerical {

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
    // Explicit leapfrog scheme for wave equation on simplicial meshes (FEM)
    // Solves: M * d²u/dt² = -c² K u + M f
    // -----------------------------------------------------------------------------
    template<typename Path, typename Value, typename SourceFunc, typename Metric, typename BC>
        requires FiniteElementGrid<typename Path::GridType>&&
    IsMetric<Metric, typename Path::GridType::point_type, typename Path::GridType::scalar_type>
        OperationalFunction<typename Path::GridType::point_type, Value, typename Path::GridType>
        solve_wave_explicit(
            const Path& path,
            const OperationalFunction<typename Path::GridType::point_type, Value, typename Path::GridType>& u0,
            const OperationalFunction<typename Path::GridType::point_type, Value, typename Path::GridType>& v0,
            const SourceFunc& source,        // source(t, vertex_index) -> Value
            Value c,                         // wave speed
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

        // Precompute vertex->index map for final function
        auto vertex_to_idx = build_vertex_to_index_map(mesh);

        // Assemble matrices using metric
        auto K = assemble_stiffness_matrix(mesh, metric);   // stiffness
        auto M = assemble_mass_matrix(mesh, metric);
        auto lumpedM = lumped_mass_matrix(M);       // diagonal lumped mass

        // Extract initial displacement and velocity
        std::vector<Value> u_prev(n);  // u^{n-1}
        std::vector<Value> u_curr(n);  // u^n
        for (std::size_t i = 0; i < n; ++i) {
            Point p = mesh.vertex(i);
            u_prev[i] = u0(p);
            u_curr[i] = u0(p) + Value(dt) * v0(p);   // u_1 = u_0 + dt * v_0  (first step)
        }

        // Time stepping (start from step = 1, since step 0 is initial)
        for (std::size_t step = 1; step < num_steps; ++step) {
            double t = step * dt;
            double t_next = (step + 1) * dt;

            // RHS = M * f(t)  (source term at current time)
            Eigen::Matrix<Value, Eigen::Dynamic, 1> rhs(n);
            rhs.setZero();
            for (std::size_t i = 0; i < n; ++i) {
                Value fi = source(t, i);
                for (typename Eigen::SparseMatrix<Value>::InnerIterator it(M, i); it; ++it) {
                    rhs(it.row()) += it.value() * fi;
                }
            }

            // Compute -c² K * u_curr
            Eigen::Matrix<Value, Eigen::Dynamic, 1> u_curr_eig(n);
            for (std::size_t i = 0; i < n; ++i) u_curr_eig(i) = u_curr[i];
            Eigen::Matrix<Value, Eigen::Dynamic, 1> Ku = K * u_curr_eig;

            // Leapfrog update: u_{n+1} = 2 u_n - u_{n-1} + dt² ( -c² M^{-1} K u_n + M^{-1} rhs )
            std::vector<Value> u_next(n);
            for (std::size_t i = 0; i < n; ++i) {
                u_next[i] = Value(2) * u_curr[i] - u_prev[i]
                    + Value(dt * dt) * (-c * c * Ku(i) / lumpedM(i) + rhs(i) / lumpedM(i));
            }

            // Apply Dirichlet BC (overwrite). Neumann BC are not supported in explicit scheme.
            for (std::size_t i = 0; i < n; ++i) {
                BCType type;
                BCValue<Value> bc_val;
                if (bc.get_vertex_condition(i, type, bc_val) && type == BCType::Dirichlet) {
                    u_next[i] = bc_val(t_next, i);
                }
            }

            // Shift for next step
            u_prev = std::move(u_curr);
            u_curr = std::move(u_next);
        }

        // u_curr now contains solution at final time (u_num_steps)
        auto vertex_to_idx_copy = vertex_to_idx;
        auto u_final_vec = u_curr;  // copy
        OperationalFunction<Point, Value, Complex> u_final(
            mesh,
            [vertex_to_idx_copy, u_final_vec](const Point& addr) -> Value {
                auto it = vertex_to_idx_copy.find(addr);
                if (it == vertex_to_idx_copy.end()) {
                    throw std::out_of_range("Address not found in wave solution");
                }
                return u_final_vec[it->second];
            }
        );

        return u_final;
    }

    // -----------------------------------------------------------------------------
    // Newmark scheme for wave equation on simplicial meshes (implicit)
    // Parameters: beta, gamma (defaults to average acceleration: beta=0.25, gamma=0.5)
    // -----------------------------------------------------------------------------
    template<typename Path, typename Value, typename SourceFunc, typename Metric, typename BC>
        requires FiniteElementGrid<typename Path::GridType>&&
    IsMetric<Metric, typename Path::GridType::point_type, typename Path::GridType::scalar_type>
        OperationalFunction<typename Path::GridType::point_type, Value, typename Path::GridType>
        solve_wave_newmark(
            const Path& path,
            const OperationalFunction<typename Path::GridType::point_type, Value, typename Path::GridType>& u0,
            const OperationalFunction<typename Path::GridType::point_type, Value, typename Path::GridType>& v0,
            const SourceFunc& source,
            Value c,
            double dt,
            std::size_t num_steps,
            const Metric& metric,
            const BC& bc,
            double beta = 0.25,
            double gamma = 0.5)
    {
        using Complex = typename Path::GridType;
        using Point = typename Complex::point_type;
        using Index = int;
        const Complex& mesh = path.current_grid();
        std::size_t n = mesh.num_vertices();

        auto vertex_to_idx = build_vertex_to_index_map(mesh);

        // Assemble matrices using metric
        auto K = assemble_stiffness_matrix(mesh, metric);
        auto M = assemble_mass_matrix(mesh, metric);
        auto lumpedM = lumped_mass_matrix(M);

        // Effective stiffness for Newmark: K_eff = M + beta * dt² * c² * K
        Eigen::SparseMatrix<Value> K_eff = M;
        if (beta != 0 && dt != 0) {
            K_eff += Value(beta * dt * dt) * c * c * K;
        }

        // Extract initial displacement, velocity, and compute initial acceleration
        std::vector<Value> u(n), v(n), a(n);
        for (std::size_t i = 0; i < n; ++i) {
            Point p = mesh.vertex(i);
            u[i] = u0(p);
            v[i] = v0(p);
        }

        // Compute initial acceleration: M a0 = -c² K u0 + M f(0)
        Eigen::Matrix<Value, Eigen::Dynamic, 1> rhs0(n);
        rhs0.setZero();
        // -c² K u0
        Eigen::Matrix<Value, Eigen::Dynamic, 1> u_eig(n);
        for (std::size_t i = 0; i < n; ++i) u_eig(i) = u[i];
        Eigen::Matrix<Value, Eigen::Dynamic, 1> Ku = K * u_eig;
        rhs0 = -c * c * Ku;
        // + M f(0)
        for (std::size_t i = 0; i < n; ++i) {
            Value fi = source(0.0, i);
            for (typename Eigen::SparseMatrix<Value>::InnerIterator it(M, i); it; ++it) {
                rhs0(it.row()) += it.value() * fi;
            }
        }
        // Apply boundary conditions to initial acceleration system
        // We solve M a0 = rhs0. Use lumpedM as placeholder (Neumann not used here)
        Eigen::SparseMatrix<Value> M_mod = M;
        apply_boundary_conditions(M_mod, rhs0, lumpedM, n, bc, 0.0);
        // Solve M a0 = rhs0
        Eigen::SparseLU<Eigen::SparseMatrix<Value>> solver_init;
        solver_init.compute(M_mod);
        if (solver_init.info() != Eigen::Success)
            throw std::runtime_error("Newmark wave solver: initial acceleration factorization failed");
        Eigen::Matrix<Value, Eigen::Dynamic, 1> a0 = solver_init.solve(rhs0);
        if (solver_init.info() != Eigen::Success)
            throw std::runtime_error("Newmark wave solver: initial acceleration solve failed");
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = a0(i);
        }

        // Time stepping
        for (std::size_t step = 0; step < num_steps; ++step) {
            double t = step * dt;
            double t_next = (step + 1) * dt;

            // Predictors
            std::vector<Value> u_pred(n), v_pred(n);
            for (std::size_t i = 0; i < n; ++i) {
                u_pred[i] = u[i] + dt * v[i] + Value((0.5 - beta) * dt * dt) * a[i];
                v_pred[i] = v[i] + Value((1.0 - gamma) * dt) * a[i];
            }

            // Compute effective RHS: rhs = M f(t+1) - c² K u_pred
            Eigen::Matrix<Value, Eigen::Dynamic, 1> rhs(n);
            rhs.setZero();
            // M f(t+1)
            for (std::size_t i = 0; i < n; ++i) {
                Value fi = source(t_next, i);
                for (typename Eigen::SparseMatrix<Value>::InnerIterator it(M, i); it; ++it) {
                    rhs(it.row()) += it.value() * fi;
                }
            }
            // - c² K u_pred
            Eigen::Matrix<Value, Eigen::Dynamic, 1> u_pred_eig(n);
            for (std::size_t i = 0; i < n; ++i) u_pred_eig(i) = u_pred[i];
            Eigen::Matrix<Value, Eigen::Dynamic, 1> Ku_pred = K * u_pred_eig;
            rhs -= c * c * Ku_pred;

            // Apply boundary conditions (Dirichlet and Neumann) to the system for acceleration
            Eigen::SparseMatrix<Value> K_eff_mod = K_eff;
            apply_boundary_conditions(K_eff_mod, rhs, lumpedM, n, bc, t_next);

            // Solve for a_{n+1}: K_eff * a_{n+1} = rhs
            Eigen::SparseLU<Eigen::SparseMatrix<Value>> solver;
            solver.compute(K_eff_mod);
            if (solver.info() != Eigen::Success)
                throw std::runtime_error("Newmark wave solver: factorization failed");
            Eigen::Matrix<Value, Eigen::Dynamic, 1> a_next = solver.solve(rhs);
            if (solver.info() != Eigen::Success)
                throw std::runtime_error("Newmark wave solver: solution failed");

            // Update displacement and velocity
            std::vector<Value> u_next(n), v_next(n);
            for (std::size_t i = 0; i < n; ++i) {
                a[i] = a_next(i);   // store for next step
                u_next[i] = u_pred[i] + Value(beta * dt * dt) * a[i];
                v_next[i] = v_pred[i] + Value(gamma * dt) * a[i];
            }

            // Overwrite Dirichlet values for displacement and velocity (as before)
            for (std::size_t i = 0; i < n; ++i) {
                BCType type;
                BCValue<Value> bc_val;
                if (bc.get_vertex_condition(i, type, bc_val) && type == BCType::Dirichlet) {
                    u_next[i] = bc_val(t_next, i);
                    v_next[i] = Value(0);   // velocity zero at fixed nodes (could also be given)
                    a[i] = Value(0);         // acceleration zero
                }
            }

            u = std::move(u_next);
            v = std::move(v_next);
        }

        // u now contains final displacement
        auto vertex_to_idx_copy = vertex_to_idx;
        auto u_final_vec = u;
        OperationalFunction<Point, Value, Complex> u_final(
            mesh,
            [vertex_to_idx_copy, u_final_vec](const Point& addr) -> Value {
                auto it = vertex_to_idx_copy.find(addr);
                if (it == vertex_to_idx_copy.end()) {
                    throw std::out_of_range("Address not found in wave solution");
                }
                return u_final_vec[it->second];
            }
        );

        return u_final;
    }

} // namespace delta::numerical