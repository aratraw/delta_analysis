// tests/numerical/test_simplicial_complex.cpp
#include <gtest/gtest.h>
#include "delta/geometry/simplicial_complex.h"
#include "delta/geometry/discrete_forms.h"

// -----------------------------------------------------------------------------
// Tests for SimplicialComplex and basic DiscreteForm functionality
// -----------------------------------------------------------------------------

TEST(SimplicialComplexTest, CreateTriangle) {
    delta::geometry::SimplicialComplex<2, double> complex;
    auto v0 = complex.add_vertex({ 0.0, 0.0 });
    auto v1 = complex.add_vertex({ 1.0, 0.0 });
    auto v2 = complex.add_vertex({ 0.0, 1.0 });
    complex.add_triangle(v0, v1, v2);

    EXPECT_EQ(complex.num_vertices(), 3);
    EXPECT_EQ(complex.num_edges(), 3);
    EXPECT_EQ(complex.num_triangles(), 1);

    EXPECT_GE(complex.find_simplex(1, { v0, v1 }), 0);
    EXPECT_EQ(complex.find_simplex(1, { v0, v1 }), complex.find_simplex(1, { v1, v0 }));
}

TEST(DiscreteFormsTest, ExteriorDerivativeZeroForm) {
    delta::geometry::SimplicialComplex<2, double> complex;
    auto v0 = complex.add_vertex({ 0.0, 0.0 });
    auto v1 = complex.add_vertex({ 1.0, 0.0 });
    auto v2 = complex.add_vertex({ 0.0, 1.0 });
    complex.add_triangle(v0, v1, v2);

    // Create a 0‑form (scalar field on vertices)
    delta::geometry::DiscreteForm<0, double, decltype(complex)> f(complex);
    f.at(v0) = 0.0;
    f.at(v1) = 1.0;
    f.at(v2) = 0.0;

    // Compute exterior derivative (1‑form on edges)
    auto df = f.d();

    // Retrieve edge indices (edges are stored with sorted vertices)
    auto e01 = complex.find_simplex(1, { v0, v1 });
    auto e12 = complex.find_simplex(1, { v1, v2 });
    auto e20 = complex.find_simplex(1, { v2, v0 });

    // For edge (i,j) with i<j, df = f(j) - f(i)
    EXPECT_DOUBLE_EQ(df.at(e01), 1.0);          // f(1)-f(0)
    EXPECT_DOUBLE_EQ(df.at(e12), -1.0);         // f(2)-f(1) = 0-1
    EXPECT_DOUBLE_EQ(df.at(e20), 0.0);          // f(2)-f(0) = 0-0
}