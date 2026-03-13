// test/solvers/test_advection_solver.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include "delta/numerical/solvers/advection_solver.h"
#include "delta/geometry/simplicial_complex.h"
#include "delta/core/operational_function.h"
#include "../test_fixtures.h"

namespace delta::testing {

    using namespace delta::numerical;
    using namespace delta::geometry;
    using namespace delta::core;

    // -----------------------------------------------------------------------------
    // Helper to create a regular triangulation of the unit square [0,1] x [0,1]
    // with n x n grid cells (each split into two triangles). Returns mesh and also
    // computes centers of triangles for convenience.
    // -----------------------------------------------------------------------------
    std::pair<SimplicialComplex<2, double>, std::vector<Eigen::Vector2d>>
        create_square_mesh_with_centers(std::size_t n) {
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

        // Add triangles and compute centers
        std::vector<Point> centers;
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < n; ++i) {
                std::size_t v00 = indices[i][j];
                std::size_t v10 = indices[i + 1][j];
                std::size_t v01 = indices[i][j + 1];
                std::size_t v11 = indices[i + 1][j + 1];

                // Triangle 1: (v00, v10, v11)
                mesh.add_triangle(v00, v10, v11);
                Point c1 = (mesh.vertex(v00) + mesh.vertex(v10) + mesh.vertex(v11)) / 3.0;
                centers.push_back(c1);

                // Triangle 2: (v00, v11, v01)
                mesh.add_triangle(v00, v11, v01);
                Point c2 = (mesh.vertex(v00) + mesh.vertex(v11) + mesh.vertex(v01)) / 3.0;
                centers.push_back(c2);
            }
        }
        return { std::move(mesh), centers };
    }

    // -----------------------------------------------------------------------------
    // Test fixture for advection solver
    // -----------------------------------------------------------------------------
    class AdvectionSolverTest : public ::testing::Test {
    protected:
        using Scalar = double;
        using Complex = SimplicialComplex<2, Scalar>;
        using Point = Complex::point_type;
        using Path = SimplicialDeltaPath<2, Scalar>; // dummy path, not refined

        EuclideanMetric metric;

        // Gaussian initial condition
        static Scalar gaussian(const Point& p, Scalar sigma = 0.1) {
            Scalar dx = p.x() - 0.5;
            Scalar dy = p.y() - 0.5;
            return std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
        }

        // Constant velocity field
        static Point constant_velocity(const Point&, Scalar vx, Scalar vy) {
            return Point(vx, vy);
        }

        // Compute total mass (integral of u over the domain)
        Scalar total_mass(const Complex& mesh,
            const OperationalFunction<Point, Scalar, Complex>& u) {
            Scalar mass = 0.0;
            for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
                Point center = mesh.cell_center(t);
                Scalar area = mesh.cell_volume(t, metric);
                mass += u(center) * area;
            }
            return mass;
        }

        // Compute center of mass
        Point center_of_mass(const Complex& mesh,
            const OperationalFunction<Point, Scalar, Complex>& u) {
            Scalar mass = 0.0;
            Point weighted_sum(0.0, 0.0);
            for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
                Point center = mesh.cell_center(t);
                Scalar area = mesh.cell_volume(t, metric);
                Scalar val = u(center);
                mass += val * area;
                weighted_sum += val * area * center;
            }
            if (mass == 0.0) return Point(0.0, 0.0);
            return weighted_sum / mass;
        }
    };

    // -----------------------------------------------------------------------------
    // Test: advection of Gaussian pulse with constant velocity
    // Checks mass conservation and center of mass motion
    // -----------------------------------------------------------------------------
    TEST_F(AdvectionSolverTest, GaussianPulseAdvection) {
        // Create a reasonably fine mesh (40x40 cells -> 3200 triangles)
        std::size_t n = 40;
        auto [mesh, centers] = create_square_mesh_with_centers(n);
        Path path(mesh); // dummy path

        // Parameters
        Scalar sigma = 0.1;
        Scalar vx = 0.2;
        Scalar vy = 0.1;
        Scalar dt = 0.005;
        Scalar T = 0.2; // total time
        std::size_t num_steps = static_cast<std::size_t>(T / dt);

        // Velocity function
        auto velocity = [vx, vy](const Point&) { return Point(vx, vy); };

        // Initial condition
        OperationalFunction<Point, Scalar, Complex> u0(mesh,
            [sigma](const Point& p) { return gaussian(p, sigma); });

        // Boundary conditions: empty (all edges open/outflow)
        BoundaryConditions<Scalar> bc;

        // Solve
        auto u_final = solve_advection_upwind_2d(path, u0, velocity, dt, num_steps,
            metric, bc);

        // Compute mass before and after
        Scalar mass0 = total_mass(mesh, u0);
        Scalar mass1 = total_mass(mesh, u_final);
        EXPECT_NEAR(mass1, mass0, 1e-3); // expect mass conservation (within discretization error)

        // Compute center of mass before and after
        Point cm0 = center_of_mass(mesh, u0);
        Point cm1 = center_of_mass(mesh, u_final);
        Point expected_cm = cm0 + Point(vx * T, vy * T);

        EXPECT_NEAR(cm1.x(), expected_cm.x(), 0.02);
        EXPECT_NEAR(cm1.y(), expected_cm.y(), 0.02);
    }

    // -----------------------------------------------------------------------------
    // Test: zero velocity -> solution should remain unchanged
    // -----------------------------------------------------------------------------
    TEST_F(AdvectionSolverTest, ZeroVelocity) {
        std::size_t n = 20;
        auto [mesh, centers] = create_square_mesh_with_centers(n);
        Path path(mesh);

        Scalar sigma = 0.15;
        auto velocity = [](const Point&) { return Point(0.0, 0.0); };

        OperationalFunction<Point, Scalar, Complex> u0(mesh,
            [sigma](const Point& p) { return gaussian(p, sigma); });

        BoundaryConditions<Scalar> bc;

        Scalar dt = 0.01;
        Scalar T = 0.1;
        std::size_t num_steps = static_cast<std::size_t>(T / dt);

        auto u_final = solve_advection_upwind_2d(path, u0, velocity, dt, num_steps,
            metric, bc);

        // Compare values at cell centers
        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            Point center = mesh.cell_center(t);
            EXPECT_NEAR(u_final(center), u0(center), 1e-12);
        }
    }

    // -----------------------------------------------------------------------------
    // Test: constant field should remain constant
    // -----------------------------------------------------------------------------
    TEST_F(AdvectionSolverTest, ConstantField) {
        std::size_t n = 20;
        auto [mesh, centers] = create_square_mesh_with_centers(n);
        Path path(mesh);

        Scalar constant_value = 3.5;
        auto velocity = [](const Point&) { return Point(0.3, 0.2); };

        OperationalFunction<Point, Scalar, Complex> u0(mesh,
            [constant_value](const Point&) { return constant_value; });

        BoundaryConditions<Scalar> bc;

        Scalar dt = 0.01;
        Scalar T = 0.1;
        std::size_t num_steps = static_cast<std::size_t>(T / dt);

        auto u_final = solve_advection_upwind_2d(path, u0, velocity, dt, num_steps,
            metric, bc);

        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            Point center = mesh.cell_center(t);
            EXPECT_NEAR(u_final(center), constant_value, 1e-12);
        }
    }

} // namespace delta::testing