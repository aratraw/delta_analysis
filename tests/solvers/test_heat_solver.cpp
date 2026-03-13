// tests/solvers/test_heat_solver.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "delta/numerical/solvers/heat_solver.h"
#include "delta/geometry/simplicial_complex.h"
#include "delta/core/operational_function.h"
#include "../test_fixtures.h"

namespace delta::testing {

    using namespace delta::numerical;
    using namespace delta::geometry;
    using namespace delta::core;

    // -----------------------------------------------------------------------------
    // Helper to create a regular triangulation of the unit square [0,1] x [0,1]
    // (same as in Poisson test)
    // -----------------------------------------------------------------------------
    SimplicialComplex<2, double> create_unit_square_mesh(std::size_t n) {
        using Complex = SimplicialComplex<2, double>;
        using Point = Complex::point_type;
        Complex mesh;

        double h = 1.0 / static_cast<double>(n);

        std::vector<std::vector<std::size_t>> indices(n + 1, std::vector<std::size_t>(n + 1));
        for (std::size_t j = 0; j <= n; ++j) {
            for (std::size_t i = 0; i <= n; ++i) {
                Point p(i * h, j * h);
                indices[i][j] = mesh.add_vertex(p);
            }
        }

        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                std::size_t v00 = indices[i][j];
                std::size_t v10 = indices[i + 1][j];
                std::size_t v01 = indices[i][j + 1];
                std::size_t v11 = indices[i + 1][j + 1];

                mesh.add_triangle(v00, v10, v11);
                mesh.add_triangle(v00, v11, v01);
            }
        }
        return mesh;
    }

    // -----------------------------------------------------------------------------
    // Test fixture for heat solver convergence tests
    // -----------------------------------------------------------------------------
    class HeatSolverTest : public ::testing::Test {
    protected:
        using Scalar = double;
        using Complex = SimplicialComplex<2, Scalar>;
        using Point = Complex::point_type;
        using Path = SimplicialDeltaPath<2, Scalar>;
        EuclideanMetric metric;

        // Exact solution for test problem: u(x,y,t) = sin(πx) sin(πy) exp(-2π² t)
        static Scalar u_exact(const Point& p, Scalar t) {
            return std::sin(M_PI * p.x()) * std::sin(M_PI * p.y()) * std::exp(-2.0 * M_PI * M_PI * t);
        }

        // Source term f = ∂u/∂t - αΔu = 0 for this exact solution (α=1)
        static Scalar source_zero(const Point&, Scalar) { return 0.0; }

        // Initial condition
        static Scalar u0(const Point& p) {
            return std::sin(M_PI * p.x()) * std::sin(M_PI * p.y());
        }

        // Boundary conditions: Dirichlet u=0 on all boundaries
        BoundaryConditions<Scalar> zero_dirichlet_bc(const Complex& mesh) {
            BoundaryConditions<Scalar> bc;
            for (std::size_t i = 0; i < mesh.num_vertices(); ++i) {
                Point p = mesh.vertex(i);
                if (p.x() == 0.0 || p.x() == 1.0 || p.y() == 0.0 || p.y() == 1.0) {
                    bc.set(i, BCType::Dirichlet, 0.0);
                }
            }
            return bc;
        }

        // Neumann zero (insulated) boundary conditions for all boundaries
        BoundaryConditions<Scalar> zero_neumann_bc(const Complex& mesh) {
            BoundaryConditions<Scalar> bc;
            for (std::size_t i = 0; i < mesh.num_vertices(); ++i) {
                Point p = mesh.vertex(i);
                if (p.x() == 0.0 || p.x() == 1.0 || p.y() == 0.0 || p.y() == 1.0) {
                    bc.set(i, BCType::Neumann, 0.0);
                }
            }
            return bc;
        }

        // Compute L2 error between numerical and exact solutions at time t
        Scalar l2_error(const Complex& mesh,
            const OperationalFunction<Point, Scalar, Complex>& u_num,
            Scalar t) {
            Scalar error_sq = 0.0;
            Scalar total_area = 0.0;
            auto dual_areas = compute_vertex_dual_areas(mesh, metric);
            for (std::size_t i = 0; i < mesh.num_vertices(); ++i) {
                Point p = mesh.vertex(i);
                Scalar diff = u_num(p) - u_exact(p, t);
                Scalar weight = dual_areas[i];
                error_sq += diff * diff * weight;
                total_area += weight;
            }
            return std::sqrt(error_sq / total_area);
        }

        // Estimate convergence order
        Scalar convergence_order(Scalar error1, Scalar error2, Scalar h1, Scalar h2) {
            return std::log(error1 / error2) / std::log(h1 / h2);
        }
    };

    // -----------------------------------------------------------------------------
    // Test explicit Euler scheme convergence
    // -----------------------------------------------------------------------------
    TEST_F(HeatSolverTest, ExplicitEulerConvergence) {
        std::vector<std::size_t> grid_sizes = { 4, 8, 16 };
        std::vector<Scalar> errors;
        std::vector<Scalar> h_vals;

        Scalar T_final = 0.1;           // final time
        Scalar dt = 0.001;              // fixed time step (must satisfy CFL for explicit)

        for (std::size_t n : grid_sizes) {
            Complex mesh = create_unit_square_mesh(n);
            Scalar h = 1.0 / n;

            // Initial condition
            OperationalFunction<Point, Scalar, Complex> u0_field(mesh, u0);

            // Boundary conditions
            auto bc = zero_dirichlet_bc(mesh);

            // Dummy path (not used except to provide grid)
            Path path(mesh);

            // Solve
            Scalar alpha = 1.0;
            std::size_t num_steps = static_cast<std::size_t>(T_final / dt);
            auto u_num = solve_heat_explicit(path, u0_field, source_zero, alpha,
                dt, num_steps, metric, bc);

            Scalar error = l2_error(mesh, u_num, T_final);
            errors.push_back(error);
            h_vals.push_back(h);
        }

        // Expect first-order convergence (explicit Euler O(dt + h²), but with fixed dt, should see h²?)
        // Actually with fixed dt, as h decreases, error dominated by spatial discretization -> O(h²).
        // We'll check order around 1.5-2.0.
        for (std::size_t i = 0; i + 1 < errors.size(); ++i) {
            Scalar order = convergence_order(errors[i], errors[i + 1], h_vals[i], h_vals[i + 1]);
            EXPECT_GT(order, 1.5);
            EXPECT_LT(order, 2.5);
        }
    }

    // -----------------------------------------------------------------------------
    // Test implicit Euler scheme convergence
    // -----------------------------------------------------------------------------
    TEST_F(HeatSolverTest, ImplicitEulerConvergence) {
        std::vector<std::size_t> grid_sizes = { 4, 8, 16 };
        std::vector<Scalar> errors;
        std::vector<Scalar> h_vals;

        Scalar T_final = 0.1;
        Scalar dt = 0.01;   // can be larger for implicit

        for (std::size_t n : grid_sizes) {
            Complex mesh = create_unit_square_mesh(n);
            Scalar h = 1.0 / n;

            OperationalFunction<Point, Scalar, Complex> u0_field(mesh, u0);
            auto bc = zero_dirichlet_bc(mesh);
            Path path(mesh);

            Scalar alpha = 1.0;
            std::size_t num_steps = static_cast<std::size_t>(T_final / dt);
            auto u_num = solve_heat_implicit(path, u0_field, source_zero, alpha,
                dt, num_steps, metric, bc);

            Scalar error = l2_error(mesh, u_num, T_final);
            errors.push_back(error);
            h_vals.push_back(h);
        }

        for (std::size_t i = 0; i + 1 < errors.size(); ++i) {
            Scalar order = convergence_order(errors[i], errors[i + 1], h_vals[i], h_vals[i + 1]);
            EXPECT_GT(order, 1.5);
            EXPECT_LT(order, 2.5);
        }
    }

    // -----------------------------------------------------------------------------
    // Test Crank-Nicolson scheme convergence
    // -----------------------------------------------------------------------------
    TEST_F(HeatSolverTest, CrankNicolsonConvergence) {
        std::vector<std::size_t> grid_sizes = { 4, 8, 16 };
        std::vector<Scalar> errors;
        std::vector<Scalar> h_vals;

        Scalar T_final = 0.1;
        Scalar dt = 0.01;

        for (std::size_t n : grid_sizes) {
            Complex mesh = create_unit_square_mesh(n);
            Scalar h = 1.0 / n;

            OperationalFunction<Point, Scalar, Complex> u0_field(mesh, u0);
            auto bc = zero_dirichlet_bc(mesh);
            Path path(mesh);

            Scalar alpha = 1.0;
            std::size_t num_steps = static_cast<std::size_t>(T_final / dt);
            auto u_num = solve_heat_crank_nicolson(path, u0_field, source_zero, alpha,
                dt, num_steps, metric, bc);

            Scalar error = l2_error(mesh, u_num, T_final);
            errors.push_back(error);
            h_vals.push_back(h);
        }

        // Crank-Nicolson is second order in time and space, so with fixed dt we expect O(h²)
        for (std::size_t i = 0; i + 1 < errors.size(); ++i) {
            Scalar order = convergence_order(errors[i], errors[i + 1], h_vals[i], h_vals[i + 1]);
            EXPECT_GT(order, 1.8);
            EXPECT_LT(order, 2.2);
        }
    }

    // -----------------------------------------------------------------------------
    // Test conservation of average temperature for insulated boundaries (Neumann=0)
    // -----------------------------------------------------------------------------
    TEST_F(HeatSolverTest, InsulatedBoundariesConservesAverage) {
        std::size_t n = 16;
        Complex mesh = create_unit_square_mesh(n);
        Path path(mesh);

        // Initial condition: constant + perturbation that doesn't affect average? Actually constant itself.
        // Choose u0 = 2.0 (constant) -> average should stay 2.0
        Scalar constant_value = 2.0;
        OperationalFunction<Point, Scalar, Complex> u0_field(mesh,
            [constant_value](const Point&) { return constant_value; });

        // Zero Neumann on all boundaries
        auto bc = zero_neumann_bc(mesh);

        // Zero source
        auto source = [](const Point&, Scalar) { return 0.0; };

        Scalar alpha = 1.0;
        Scalar dt = 0.01;
        Scalar T_final = 1.0;
        std::size_t num_steps = static_cast<std::size_t>(T_final / dt);

        // Use implicit scheme (any scheme should conserve average)
        auto u_final = solve_heat_implicit(path, u0_field, source, alpha,
            dt, num_steps, metric, bc);

        // Compute average temperature at final time
        Scalar total = 0.0;
        Scalar total_area = 0.0;
        auto dual_areas = compute_vertex_dual_areas(mesh, metric);
        for (std::size_t i = 0; i < mesh.num_vertices(); ++i) {
            Point p = mesh.vertex(i);
            total += u_final(p) * dual_areas[i];
            total_area += dual_areas[i];
        }
        Scalar avg_final = total / total_area;

        EXPECT_NEAR(avg_final, constant_value, 1e-12);
    }

    // -----------------------------------------------------------------------------
    // Test with non-zero source term and known exact solution (optional)
    // -----------------------------------------------------------------------------
    TEST_F(HeatSolverTest, WithSourceTerm) {
        // Use exact solution u = e^{-t} sin(πx) sin(πy)
        // Then ∂u/∂t = -u, Δu = -2π² u, so equation ∂u/∂t - Δu = -u + 2π² u = (2π² -1) u
        // So source f = (2π² -1) u.
        std::size_t n = 16;
        Complex mesh = create_unit_square_mesh(n);
        Path path(mesh);

        auto u0 = [](const Point& p) { return std::sin(M_PI * p.x()) * std::sin(M_PI * p.y()); };
        OperationalFunction<Point, Scalar, Complex> u0_field(mesh, u0);

        auto source = [](const Point& p, Scalar t) {
            Scalar u = std::sin(M_PI * p.x()) * std::sin(M_PI * p.y()) * std::exp(-t);
            return (2.0 * M_PI * M_PI - 1.0) * u;
            };

        auto bc = zero_dirichlet_bc(mesh);

        Scalar alpha = 1.0;
        Scalar dt = 0.01;
        Scalar T_final = 0.1;
        std::size_t num_steps = static_cast<std::size_t>(T_final / dt);

        auto u_num = solve_heat_crank_nicolson(path, u0_field, source, alpha,
            dt, num_steps, metric, bc);

        Scalar error = l2_error(mesh, u_num, T_final);
        EXPECT_LT(error, 0.01);
    }

} // namespace delta::testing