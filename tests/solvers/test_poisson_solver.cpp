// tests/solvers/test_poisson_solver.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "delta/numerical/solvers/poisson_solver.h"
#include "delta/geometry/simplicial_complex.h"
#include "delta/core/operational_function.h"
#include "../test_fixtures.h"

namespace delta::testing {

    using namespace delta::numerical;
    using namespace delta::geometry;
    using namespace delta::core;

    // -----------------------------------------------------------------------------
    // Helper to create a regular triangulation of the unit square [0,1] x [0,1]
    // with n x n grid cells (each split into two triangles).
    // Returns a SimplicialComplex<2, double>.
    // -----------------------------------------------------------------------------
    SimplicialComplex<2, double> create_unit_square_mesh(std::size_t n) {
        using Complex = SimplicialComplex<2, double>;
        using Point = Complex::point_type;
        Complex mesh;

        double h = 1.0 / static_cast<double>(n);

        // Add vertices
        std::vector<std::vector<std::size_t>> indices(n + 1, std::vector<std::size_t>(n + 1));
        for (std::size_t j = 0; j <= n; ++j) {
            for (std::size_t i = 0; i <= n; ++i) {
                Point p(i * h, j * h);
                indices[i][j] = mesh.add_vertex(p);
            }
        }

        // Add triangles
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                std::size_t v00 = indices[i][j];
                std::size_t v10 = indices[i + 1][j];
                std::size_t v01 = indices[i][j + 1];
                std::size_t v11 = indices[i + 1][j + 1];

                // First triangle (lower-left)
                mesh.add_triangle(v00, v10, v11);
                // Second triangle (upper-right) - using (v00, v11, v01) to avoid diagonal crossing?
                // Standard splitting: (i,j)-(i+1,j)-(i+1,j+1) and (i,j)-(i+1,j+1)-(i,j+1)
                mesh.add_triangle(v00, v11, v01);
            }
        }

        return mesh;
    }

    // -----------------------------------------------------------------------------
    // Test fixture for Poisson solver convergence tests
    // -----------------------------------------------------------------------------
    class PoissonSolverTest : public ::testing::Test {
    protected:
        using Scalar = double;
        using Complex = SimplicialComplex<2, Scalar>;
        using Point = Complex::point_type;
        using Path = SimplicialDeltaPath<2, Scalar>;  // not actually used for refinement, just for interface
        EuclideanMetric metric;

        // Boundary conditions: Dirichlet u = 0 on all boundary vertices
        BoundaryConditions<Scalar> zero_dirichlet_bc(const Complex& mesh) {
            BoundaryConditions<Scalar> bc;
            // Identify boundary vertices: those with x=0 or x=1 or y=0 or y=1
            for (std::size_t i = 0; i < mesh.num_vertices(); ++i) {
                Point p = mesh.vertex(i);
                if (p.x() == 0.0 || p.x() == 1.0 || p.y() == 0.0 || p.y() == 1.0) {
                    bc.set(i, BCType::Dirichlet, 0.0);
                }
            }
            return bc;
        }

        // Compute L2 error between numerical and exact solutions
        Scalar l2_error(const Complex& mesh,
            const OperationalFunction<Point, Scalar, Complex>& u_num,
            const std::function<Scalar(const Point&)>& u_exact) {
            Scalar error_sq = 0.0;
            Scalar total_area = 0.0;
            for (std::size_t i = 0; i < mesh.num_vertices(); ++i) {
                Point p = mesh.vertex(i);
                Scalar diff = u_num(p) - u_exact(p);
                // We need a weight for each vertex. Use dual area (Voronoi cell).
                // Compute dual area via vertex_dual_areas from discrete_operators.
                // But that function is in numerical, so we can use it.
                auto dual_areas = compute_vertex_dual_areas(mesh, metric);
                Scalar weight = dual_areas[i];
                error_sq += diff * diff * weight;
                total_area += weight;
            }
            return std::sqrt(error_sq / total_area);
        }

        // Estimate convergence order from two errors and mesh sizes
        Scalar convergence_order(Scalar error1, Scalar error2, Scalar h1, Scalar h2) {
            return std::log(error1 / error2) / std::log(h1 / h2);
        }
    };

    // -----------------------------------------------------------------------------
    // Test convergence for Poisson problem -Δu = f with u = sin(πx) sin(πy)
    // -----------------------------------------------------------------------------
    TEST_F(PoissonSolverTest, ConvergenceSineSolution) {
        std::vector<std::size_t> grid_sizes = { 4, 8, 16, 32 };  // number of cells per dimension
        std::vector<Scalar> errors;
        std::vector<Scalar> h_vals;

        for (std::size_t n : grid_sizes) {
            // Create mesh
            Complex mesh = create_unit_square_mesh(n);
            Scalar h = 1.0 / static_cast<Scalar>(n);  // approximate element size

            // Exact solution and right-hand side
            auto u_exact = [](const Point& p) {
                return std::sin(M_PI * p.x()) * std::sin(M_PI * p.y());
                };
            auto f = [](const Point& p) {
                // -Δu = 2π² sin(πx) sin(πy)
                return 2.0 * M_PI * M_PI * std::sin(M_PI * p.x()) * std::sin(M_PI * p.y());
                };

            // Create right-hand side field
            OperationalFunction<Point, Scalar, Complex> rhs(mesh, f);

            // Boundary conditions (Dirichlet u=0 on boundary)
            auto bc = zero_dirichlet_bc(mesh);

            // Create a dummy path (not used in solver except to provide current_grid)
            // The solver expects a Path object; we can use SimplicialDeltaPath with the mesh.
            Path path(mesh);

            // Solve
            auto u_num = solve_poisson(path, rhs, bc, metric);

            // Compute error
            Scalar error = l2_error(mesh, u_num, u_exact);
            errors.push_back(error);
            h_vals.push_back(h);

            // Optional: log
            // std::cout << "n=" << n << ", h=" << h << ", error=" << error << std::endl;
        }

        // Check convergence order between successive grids
        for (std::size_t i = 0; i + 1 < errors.size(); ++i) {
            Scalar order = convergence_order(errors[i], errors[i + 1], h_vals[i], h_vals[i + 1]);
            EXPECT_NEAR(order, 2.0, 0.2) << "Convergence order between levels " << i << " and " << i + 1;
        }

        // Also check that the error decreases
        for (std::size_t i = 0; i + 1 < errors.size(); ++i) {
            EXPECT_LT(errors[i + 1], errors[i]) << "Error should decrease with refinement";
        }
    }

    // -----------------------------------------------------------------------------
    // Test with constant right-hand side and zero Dirichlet BC
    // Solution should be positive inside the domain.
    // -----------------------------------------------------------------------------
    TEST_F(PoissonSolverTest, ConstantRHS) {
        std::size_t n = 16;  // moderate grid
        Complex mesh = create_unit_square_mesh(n);
        auto bc = zero_dirichlet_bc(mesh);

        // f = 1
        auto f = [](const Point&) { return 1.0; };
        OperationalFunction<Point, Scalar, Complex> rhs(mesh, f);

        Path path(mesh);
        auto u_num = solve_poisson(path, rhs, bc, metric);

        // Check that solution is positive at the center (0.5, 0.5)
        Point center(0.5, 0.5);
        Scalar u_center = u_num(center);
        EXPECT_GT(u_center, 0.0);

        // Also check that solution is symmetric (optional)
        Point p1(0.25, 0.5);
        Point p2(0.75, 0.5);
        EXPECT_NEAR(u_num(p1), u_num(p2), 1e-10);

        // Maximum should be at the center
        Scalar max_val = u_center;
        for (std::size_t i = 0; i < mesh.num_vertices(); ++i) {
            Point p = mesh.vertex(i);
            Scalar val = u_num(p);
            EXPECT_LE(val, max_val + 1e-12);
        }
    }

} // namespace delta::testing