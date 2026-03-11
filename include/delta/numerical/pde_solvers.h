// include/delta/numerical/pde_solvers.h
#pragma once

#include "delta/core/grid_concept.h"
#include "delta/core/regulative_idea.h"
#include "delta/core/operational_function.h"
#include "discrete_operators.h"
#include <Eigen/Sparse>
#include <vector>
#include <functional>

namespace delta::numerical {

    // -------------------------------------------------------------------------
    // Poisson solver on a general grid (using discrete Laplacian)
    // -------------------------------------------------------------------------

    /**
     * @brief Solve the Poisson equation -Δu = f on a simplicial mesh with Dirichlet BC.
     *
     * @tparam Complex  Type satisfying SimplicialComplex.
     * @tparam Metric   Address metric.
     * @param mesh      The mesh.
     * @param rhs       Right-hand side values at vertices (size = mesh.size()).
     * @param dirichlet_mask Boolean mask for Dirichlet vertices (true = fixed).
     * @param dirichlet_values Fixed values at Dirichlet vertices.
     * @param metric    Metric used for Laplacian.
     * @return std::vector<typename Complex::value_type> Solution at vertices.
     */
    template<typename Complex, typename Metric>
    std::vector<typename Complex::value_type>
        solve_poisson(const Complex& mesh,
            const std::vector<typename Complex::value_type>& rhs,
            const std::vector<bool>& dirichlet_mask,
            const std::vector<typename Complex::value_type>& dirichlet_values,
            const Metric& metric) {
        using Value = typename Complex::value_type;
        std::size_t n = mesh.size();

        // Build cotangent Laplacian matrix
        std::vector<Eigen::Triplet<Value>> triplets;
        std::vector<Value> vertex_areas(n, Value{ 0 });

        // Compute vertex areas (Voronoi)
        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto [v0, v1, v2] = mesh.triangle(t);
            auto p0 = mesh.vertex(v0);
            auto p1 = mesh.vertex(v1);
            auto p2 = mesh.vertex(v2);
            Value area = triangle_area(p0, p1, p2, metric);
            vertex_areas[v0] += area / Value{ 3 };
            vertex_areas[v1] += area / Value{ 3 };
            vertex_areas[v2] += area / Value{ 3 };
        }

        // Compute cotangent weights
        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto [v0, v1, v2] = mesh.triangle(t);
            auto p0 = mesh.vertex(v0);
            auto p1 = mesh.vertex(v1);
            auto p2 = mesh.vertex(v2);

            auto cot = [&](const auto& a, const auto& b, const auto& c) -> Value {
                auto ab = b - a;
                auto ac = c - a;
                Value dot = ab.dot(ac);
                Value cross = abs(ab.x() * ac.y() - ab.y() * ac.x());
                if (cross == Value{ 0 }) return Value{ 0 };
                return dot / cross;
                };

            Value cot0 = cot(p1, p2, p0);
            Value cot1 = cot(p2, p0, p1);
            Value cot2 = cot(p0, p1, p2);

            auto add_edge = [&](std::size_t i, std::size_t j, Value w) {
                if (dirichlet_mask[i] && dirichlet_mask[j]) return; // both fixed, no equation
                if (!dirichlet_mask[i] && !dirichlet_mask[j]) {
                    triplets.emplace_back(i, j, -w / (Value{ 2 } * vertex_areas[i]));
                    triplets.emplace_back(j, i, -w / (Value{ 2 } * vertex_areas[j]));
                    triplets.emplace_back(i, i, w / (Value{ 2 } * vertex_areas[i]));
                    triplets.emplace_back(j, j, w / (Value{ 2 } * vertex_areas[j]));
                }
                else if (!dirichlet_mask[i]) {
                    // j is fixed: move to RHS
                    triplets.emplace_back(i, i, w / (Value{ 2 } * vertex_areas[i]));
                    // RHS contribution will be added later
                }
                else {
                    // i is fixed, j free: similar
                    triplets.emplace_back(j, j, w / (Value{ 2 } * vertex_areas[j]));
                }
                };

            add_edge(v0, v1, cot2);
            add_edge(v1, v2, cot0);
            add_edge(v2, v0, cot1);
        }

        Eigen::SparseMatrix<Value> A(n, n);
        A.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::Matrix<Value, Eigen::Dynamic, 1> b(n);
        for (std::size_t i = 0; i < n; ++i) {
            b(i) = rhs[i];
            if (dirichlet_mask[i]) {
                A.row(i).setZero();
                A.coeffRef(i, i) = 1.0;
                b(i) = dirichlet_values[i];
            }
        }

        Eigen::SparseLU<Eigen::SparseMatrix<Value>> solver;
        solver.compute(A);
        Eigen::Matrix<Value, Eigen::Dynamic, 1> x = solver.solve(b);

        std::vector<Value> result(n);
        for (std::size_t i = 0; i < n; ++i) result[i] = x(i);
        return result;
    }

    // -------------------------------------------------------------------------
    // Heat equation solver (simple explicit Euler)
    // -------------------------------------------------------------------------

    /**
     * @brief Solve the heat equation ∂u/∂t = α Δu + f using explicit Euler.
     *
     * @tparam Complex  SimplicialComplex.
     * @tparam Metric   Metric.
     * @param mesh      The mesh.
     * @param u0        Initial condition at vertices.
     * @param f         Source term (function of time and vertex index, or constant).
     * @param alpha     Thermal diffusivity.
     * @param dt        Time step.
     * @param num_steps Number of steps.
     * @param metric    Metric.
     * @return std::vector<std::vector<Value>> Solution at each time step.
     */
    template<typename Complex, typename Metric>
    std::vector<std::vector<typename Complex::value_type>>
        solve_heat_explicit(const Complex& mesh,
            const std::vector<typename Complex::value_type>& u0,
            const std::function<typename Complex::value_type(double, std::size_t)>& f,
            typename Complex::value_type alpha,
            double dt,
            std::size_t num_steps,
            const Metric& metric) {
        using Value = typename Complex::value_type;
        std::size_t n = mesh.size();

        // Precompute Laplacian matrix (cotangent)
        // We'll use a simpler approach: compute L as sparse matrix
        // For explicit Euler, we need u_new = u_old + dt * (α L u_old + f)

        // Build L (cotangent Laplacian) - reuse code from discrete_laplacian_cotangent but as matrix
        std::vector<Eigen::Triplet<Value>> triplets;
        std::vector<Value> vertex_areas(n, Value{ 0 });

        // Compute areas and cotangents (similar to above)
        // ... (same as in Poisson)
        // For brevity, we'll just call discrete_laplacian_cotangent which returns vector, not matrix.
        // But we need matrix for repeated application. Let's build matrix explicitly.

        // Actually, for explicit Euler we can compute L*u by calling discrete_laplacian_cotangent each step.
        // That might be simpler. We'll do that.

        std::vector<Value> u = u0;
        std::vector<std::vector<Value>> result;
        result.push_back(u);

        for (std::size_t step = 0; step < num_steps; ++step) {
            double t = step * dt;
            auto Lu = discrete_laplacian_cotangent(mesh, u, metric);
            std::vector<Value> u_new(n);
            for (std::size_t i = 0; i < n; ++i) {
                Value fi = f(t, i);
                u_new[i] = u[i] + dt * (alpha * Lu[i] + fi);
            }
            u = u_new;
            result.push_back(u);
        }
        return result;
    }

} // namespace delta::numerical 