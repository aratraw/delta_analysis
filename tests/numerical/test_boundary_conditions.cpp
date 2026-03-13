// tests/numerical/test_boundary_conditions.cpp
#include <gtest/gtest.h>
#include "delta/numerical/boundary_conditions.h"
#include "delta/geometry/simplicial_complex.h"
#include "../test_fixtures.h"

namespace delta::testing {

    using namespace delta::numerical;
    using namespace delta::geometry;

    // -----------------------------------------------------------------------------
    // Test fixture for BoundaryConditions tests with a simple 2D mesh
    // -----------------------------------------------------------------------------
    class BoundaryConditionsTest : public SimplicialComplexFixture<2, double> {
    protected:
        using Scalar = double;
        using BC = BoundaryConditions<Scalar>;

        // We'll use the square2D mesh from the fixture
        BC bc;

        void SetUp() override {
            SimplicialComplexFixture::SetUp();
            // square2D has vertices 0,1,2,3 and edges:
            // e0: (0,1) bottom
            // e1: (1,2) right
            // e2: (2,3) top
            // e3: (3,0) left
            // e4: (0,2) diagonal (internal)
            bc = BC{};
        }
    };

    // -----------------------------------------------------------------------------
    // Basic set/get for vertices
    // -----------------------------------------------------------------------------
    TEST_F(BoundaryConditionsTest, SetAndGetVertex) {
        bc.set(1, BCType::Dirichlet, 3.14);
        bc.set(2, BCType::Neumann, 2.5);

        BCType type;
        BCValue<Scalar> value;

        EXPECT_TRUE(bc.get_vertex_condition(1, type, value));
        EXPECT_EQ(type, BCType::Dirichlet);
        EXPECT_DOUBLE_EQ(value(0.0, 1), 3.14);

        EXPECT_TRUE(bc.get_vertex_condition(2, type, value));
        EXPECT_EQ(type, BCType::Neumann);
        EXPECT_DOUBLE_EQ(value(0.0, 2), 2.5);

        EXPECT_FALSE(bc.get_vertex_condition(3, type, value));
    }

    // -----------------------------------------------------------------------------
    // Set/get for edges
    // -----------------------------------------------------------------------------
    TEST_F(BoundaryConditionsTest, SetAndGetEdge) {
        bc.set_edge_condition(0, BCType::Dirichlet, 0.0);
        bc.set_edge_condition(1, BCType::Neumann, 1.0);

        BCType type;
        BCValue<Scalar> value;

        EXPECT_TRUE(bc.get_edge_condition(0, type, value));
        EXPECT_EQ(type, BCType::Dirichlet);
        EXPECT_DOUBLE_EQ(value(0.0, 0), 0.0);

        EXPECT_TRUE(bc.get_edge_condition(1, type, value));
        EXPECT_EQ(type, BCType::Neumann);
        EXPECT_DOUBLE_EQ(value(0.0, 1), 1.0);

        EXPECT_FALSE(bc.get_edge_condition(2, type, value));
    }

    // -----------------------------------------------------------------------------
    // Set multiple vertices at once
    // -----------------------------------------------------------------------------
    TEST_F(BoundaryConditionsTest, SetMultipleVertices) {
        std::vector<std::size_t> indices = { 0, 2, 4 };
        bc.set(indices, BCType::Dirichlet, 42.0);

        BCType type;
        BCValue<Scalar> value;
        EXPECT_TRUE(bc.get_vertex_condition(0, type, value));
        EXPECT_EQ(type, BCType::Dirichlet);
        EXPECT_DOUBLE_EQ(value(0.0, 0), 42.0);

        EXPECT_TRUE(bc.get_vertex_condition(2, type, value));
        EXPECT_DOUBLE_EQ(value(0.0, 2), 42.0);

        EXPECT_FALSE(bc.get_vertex_condition(1, type, value));
    }

    // -----------------------------------------------------------------------------
    // Periodic pairs
    // -----------------------------------------------------------------------------
    TEST_F(BoundaryConditionsTest, PeriodicPairs) {
        bc.set_periodic_pair(0, 10);
        bc.set_periodic_pair(1, 11);

        const auto& pairs = bc.periodic_pairs();
        ASSERT_EQ(pairs.size(), 2u);
        EXPECT_EQ(pairs[0].first, 0u);
        EXPECT_EQ(pairs[0].second, 10u);
        EXPECT_EQ(pairs[1].first, 1u);
        EXPECT_EQ(pairs[1].second, 11u);
    }

    // -----------------------------------------------------------------------------
    // Boundary flux for Dirichlet condition
    // -----------------------------------------------------------------------------
    TEST_F(BoundaryConditionsTest, BoundaryFluxDirichlet) {
        // Set Dirichlet condition on bottom edge (e0) with value 5.0
        bc.set_edge_condition(0, BCType::Dirichlet, 5.0);

        // Case 1: vn > 0 (flow out of domain) -> use cell value
        double u_cell = 2.0;
        double vn = 1.5;
        double flux = bc.boundary_flux(0, 0, u_cell, vn);
        EXPECT_DOUBLE_EQ(flux, vn * u_cell);

        // Case 2: vn < 0 (flow into domain) -> use boundary value
        vn = -1.5;
        flux = bc.boundary_flux(0, 0, u_cell, vn);
        EXPECT_DOUBLE_EQ(flux, vn * 5.0);
    }

    // -----------------------------------------------------------------------------
    // Boundary flux for Neumann condition
    // -----------------------------------------------------------------------------
    TEST_F(BoundaryConditionsTest, BoundaryFluxNeumann) {
        // Set Neumann condition on right edge (e1) with flux value 2.0
        bc.set_edge_condition(1, BCType::Neumann, 2.0);

        double u_cell = 3.0;
        double vn = 1.5; // vn is ignored for Neumann
        double flux = bc.boundary_flux(1, 1, u_cell, vn);
        EXPECT_DOUBLE_EQ(flux, 2.0);

        // Check with negative vn also ignored
        flux = bc.boundary_flux(1, 1, u_cell, -1.5);
        EXPECT_DOUBLE_EQ(flux, 2.0);
    }

    // -----------------------------------------------------------------------------
    // Boundary flux when no condition is set (outflow / open boundary)
    // -----------------------------------------------------------------------------
    TEST_F(BoundaryConditionsTest, BoundaryFluxOutflow) {
        // No condition set on top edge (e2)
        double u_cell = 3.5;
        double vn = 1.2;
        double flux = bc.boundary_flux(2, 2, u_cell, vn);
        // Default outflow: flux = vn * u_cell
        EXPECT_DOUBLE_EQ(flux, vn * u_cell);

        // With negative vn, still outflow (depends on implementation; current returns vn*u_cell)
        flux = bc.boundary_flux(2, 2, u_cell, -0.8);
        EXPECT_DOUBLE_EQ(flux, -0.8 * u_cell);
    }

    // -----------------------------------------------------------------------------
    // Time-dependent boundary conditions
    // -----------------------------------------------------------------------------
    TEST_F(BoundaryConditionsTest, TimeDependentBC) {
        // Create a time-dependent function: f(t, idx) = t * idx
        auto func = [](double t, std::size_t idx) { return t * static_cast<double>(idx); };
        bc.set(3, BCType::Dirichlet, BCValue<double>(func));

        BCType type;
        BCValue<double> value;
        ASSERT_TRUE(bc.get_vertex_condition(3, type, value));

        EXPECT_DOUBLE_EQ(value(2.5, 3), 2.5 * 3.0);
        EXPECT_DOUBLE_EQ(value(0.0, 3), 0.0);
        EXPECT_DOUBLE_EQ(value(1.0, 3), 3.0);
    }

} // namespace delta::testing