// tests/geometry/dual_complex_test.cpp
#include <gtest/gtest.h>
#include <vector>
#include <array>
#include "delta/geometry/simplicial_complex.h"
#include "delta/geometry/dual_complex.h"
#include "delta/core/regulative_idea.h"
#include "../test_fixtures_geometry_numerical.h"

namespace delta::testing {

    class DualComplexTest : public GeometryNumericalTest {
    protected:
        EuclideanMetric metric;
    };

    TEST_F(DualComplexTest, Dual2D_UnitSquare) {
        Complex<2> mesh;
        make_unit_square_triangulation(mesh);
        DualComplex dual(mesh, metric);

        EXPECT_EQ(dual.num_cells(2), mesh.num_vertices());
        EXPECT_EQ(dual.num_cells(1), mesh.num_edges());
        EXPECT_EQ(dual.num_cells(0), mesh.num_triangles());

        Scalar total_area = 0;
        for (std::size_t i = 0; i < dual.num_cells(2); ++i) {
            total_area += dual.dual_volume(2, i);
            EXPECT_GT(dual.dual_volume(2, i), 0);
        }
        EXPECT_RATIONAL_NEAR(total_area, 1_r, delta::default_eps());

        for (std::size_t v = 0; v < mesh.num_vertices(); ++v) {
            std::size_t d = dual.primal_to_dual(0, v);
            EXPECT_EQ(dual.dual_to_primal(2, d), v);
        }
        for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
            std::size_t d = dual.primal_to_dual(1, e);
            EXPECT_EQ(dual.dual_to_primal(1, d), e);
        }
        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            std::size_t d = dual.primal_to_dual(2, t);
            EXPECT_EQ(dual.dual_to_primal(0, d), t);
        }
    }

    TEST_F(DualComplexTest, Dual3D_SingleTetrahedron) {
        Complex<3> mesh;
        auto v0 = mesh.add_vertex({ 0,0,0 });
        auto v1 = mesh.add_vertex({ 1,0,0 });
        auto v2 = mesh.add_vertex({ 0,1,0 });
        auto v3 = mesh.add_vertex({ 0,0,1 });
        mesh.add_edge(v0, v1); mesh.add_edge(v0, v2); mesh.add_edge(v0, v3);
        mesh.add_edge(v1, v2); mesh.add_edge(v1, v3); mesh.add_edge(v2, v3);
        mesh.add_triangle(v0, v1, v2); mesh.add_triangle(v0, v1, v3);
        mesh.add_triangle(v0, v2, v3); mesh.add_triangle(v1, v2, v3);
        mesh.add_tetrahedron(v0, v1, v2, v3);

        DualComplex dual(mesh, metric);

        EXPECT_EQ(dual.num_cells(3), mesh.num_vertices());
        EXPECT_EQ(dual.num_cells(2), mesh.num_edges());
        EXPECT_EQ(dual.num_cells(1), mesh.num_triangles());
        EXPECT_EQ(dual.num_cells(0), mesh.num_tetrahedra());

        Scalar total_vol = 0;
        for (std::size_t i = 0; i < dual.num_cells(3); ++i) {
            total_vol += dual.dual_volume(3, i);
            EXPECT_GT(dual.dual_volume(3, i), 0);
        }
        Scalar expected = 1_r / 6_r;
        EXPECT_RATIONAL_NEAR(total_vol, expected, delta::default_eps());

        for (std::size_t v = 0; v < mesh.num_vertices(); ++v) {
            std::size_t d = dual.primal_to_dual(0, v);
            EXPECT_EQ(dual.dual_to_primal(3, d), v);
        }
        for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
            std::size_t d = dual.primal_to_dual(1, e);
            EXPECT_EQ(dual.dual_to_primal(2, d), e);
        }
        for (std::size_t f = 0; f < mesh.num_triangles(); ++f) {
            std::size_t d = dual.primal_to_dual(2, f);
            EXPECT_EQ(dual.dual_to_primal(1, d), f);
        }
        for (std::size_t t = 0; t < mesh.num_tetrahedra(); ++t) {
            std::size_t d = dual.primal_to_dual(3, t);
            EXPECT_EQ(dual.dual_to_primal(0, d), t);
        }
    }

    TEST_F(DualComplexTest, Dual3D_UnitCube) {
        using VertexIdx = typename Complex<3>::vertex_index;

        Complex<3> mesh;
        std::vector<Point<3>> pts = {
            {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
            {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
        };
        std::vector<VertexIdx> idx(8);
        for (int i = 0; i < 8; ++i) idx[i] = mesh.add_vertex(pts[i]);

        for (int i = 0; i < 8; ++i)
            for (int j = i + 1; j < 8; ++j)
                if ((pts[i] - pts[j]).data().norm() == 1_r)
                    mesh.add_edge(idx[i], idx[j]);

        // Правильное разбиение куба на 6 тетраэдров
 // Все они содержат главную диагональ от вершины 0 до вершины 6. 
    // ЕСЛИ КОМУ-ТО ПОНАДОБИТСЯ КОРРЕКТНОЕ РАЗБИЕНИЕ КУБА НА ТЕТРАЭДРЫ - ЭТО ОНО.
        using Tet = std::array<VertexIdx, 4>;
        std::vector<Tet> tets = {
            {idx[0], idx[1], idx[2], idx[6]},
            {idx[0], idx[1], idx[5], idx[6]},
            {idx[0], idx[4], idx[5], idx[6]},
            {idx[0], idx[4], idx[7], idx[6]},
            {idx[0], idx[3], idx[7], idx[6]},
            {idx[0], idx[3], idx[2], idx[6]}
        };
        for (const auto& tet : tets)
            mesh.add_tetrahedron(tet[0], tet[1], tet[2], tet[3]);

        DualComplex dual(mesh, metric);

        Scalar total = 0;
        for (std::size_t i = 0; i < dual.num_cells(3); ++i)
            total += dual.dual_volume(3, i);

        EXPECT_RATIONAL_NEAR(total, 1_r, delta::default_eps());
    }
    TEST_F(DualComplexTest, Dual2D_AdditionalChecks) {
        Complex<2> mesh;
        make_unit_square_triangulation(mesh);
        DualComplex dual(mesh, metric);

        auto tri0 = mesh.triangle_at(0);
        auto tri1 = mesh.triangle_at(1);
        Point<2> c0 = (mesh.vertex(tri0[0]) + mesh.vertex(tri0[1]) + mesh.vertex(tri0[2])) / 3_r;
        Point<2> c1 = (mesh.vertex(tri1[0]) + mesh.vertex(tri1[1]) + mesh.vertex(tri1[2])) / 3_r;

        std::ptrdiff_t diag = mesh.find_simplex(1, { 0, 2 });
        ASSERT_NE(diag, -1);
        EXPECT_RATIONAL_NEAR(dual.dual_volume(1, static_cast<std::size_t>(diag)),
            metric(c0, c1), delta::default_eps());

        std::ptrdiff_t e01 = mesh.find_simplex(1, { 0, 1 });
        ASSERT_NE(e01, -1);
        Point<2> mid01 = (mesh.vertex(0) + mesh.vertex(1)) / 2_r;
        EXPECT_RATIONAL_NEAR(dual.dual_volume(1, static_cast<std::size_t>(e01)),
            metric(c0, mid01), delta::default_eps());

        // Правильные двойственные площади вершин для квадрата:
        // вершина 0 (угол) – два прилегающих треугольника → 1/3 + 0 = 1/3?
        // На самом деле в нашей триангуляции вершина 0 входит в оба треугольника:
        // треугольник 0 (v0,v1,v2) и треугольник 1 (v0,v2,v3). Доля от каждого = площадь_треугольника / 3.
        // Площадь каждого треугольника = 0.5, 0.5/3 = 1/6. Сумма = 1/3.
        // Вершина 1 входит только в треугольник 0 → 1/6.
        EXPECT_RATIONAL_NEAR(dual.dual_volume(2, 0), 1_r / 3_r, delta::default_eps());
        EXPECT_RATIONAL_NEAR(dual.dual_volume(2, 1), 1_r / 6_r, delta::default_eps());
    }

    TEST_F(DualComplexTest, Dual3D_TetrahedronDetailed) {
        Complex<3> mesh;
        auto v0 = mesh.add_vertex({ 0,0,0 });
        auto v1 = mesh.add_vertex({ 1,0,0 });
        auto v2 = mesh.add_vertex({ 0,1,0 });
        auto v3 = mesh.add_vertex({ 0,0,1 });
        mesh.add_edge(v0, v1); mesh.add_edge(v0, v2); mesh.add_edge(v0, v3);
        mesh.add_edge(v1, v2); mesh.add_edge(v1, v3); mesh.add_edge(v2, v3);
        mesh.add_triangle(v0, v1, v2);
        mesh.add_triangle(v0, v1, v3);
        mesh.add_triangle(v0, v2, v3);
        mesh.add_triangle(v1, v2, v3);
        mesh.add_tetrahedron(v0, v1, v2, v3);

        DualComplex dual(mesh, metric);

        Point<3> tet_center = (mesh.vertex(v0) + mesh.vertex(v1) + mesh.vertex(v2) + mesh.vertex(v3)) / 4_r;

        auto check_face = [&](typename Complex<3>::vertex_index a,
            typename Complex<3>::vertex_index b,
            typename Complex<3>::vertex_index c) {
                auto fidx = mesh.find_simplex(2, { a, b, c });
                ASSERT_NE(fidx, -1);
                Point<3> fc = (mesh.vertex(a) + mesh.vertex(b) + mesh.vertex(c)) / 3_r;
                EXPECT_RATIONAL_NEAR(dual.dual_volume(1, static_cast<std::size_t>(fidx)),
                    metric(tet_center, fc), delta::default_eps());
            };
        check_face(v0, v1, v2);
        check_face(v0, v1, v3);
        check_face(v0, v2, v3);
        check_face(v1, v2, v3);

        auto area3d = [](const Point<3>& p, const Point<3>& q, const Point<3>& r) -> Scalar {
            auto v1 = (q - p).data();
            auto v2 = (r - p).data();
            return v1.cross(v2).norm() / 2_r;
            };

        auto check_edge = [&](typename Complex<3>::vertex_index a,
            typename Complex<3>::vertex_index b,
            typename Complex<3>::vertex_index f1,
            typename Complex<3>::vertex_index f2) {
                auto eidx = mesh.find_simplex(1, { a, b });
                ASSERT_NE(eidx, -1);
                Point<3> mid = (mesh.vertex(a) + mesh.vertex(b)) / 2_r;
                Point<3> fc1 = (mesh.vertex(a) + mesh.vertex(b) + mesh.vertex(f1)) / 3_r;
                Point<3> fc2 = (mesh.vertex(a) + mesh.vertex(b) + mesh.vertex(f2)) / 3_r;
                Scalar dual_area = area3d(tet_center, fc1, mid) + area3d(tet_center, mid, fc2);
                EXPECT_RATIONAL_NEAR(dual.dual_volume(2, static_cast<std::size_t>(eidx)),
                    dual_area, delta::default_eps());
            };
        check_edge(v0, v1, v2, v3);
        check_edge(v0, v2, v1, v3);
        check_edge(v0, v3, v1, v2);
        check_edge(v1, v2, v0, v3);
        check_edge(v1, v3, v0, v2);
        check_edge(v2, v3, v0, v1);
    }

} // namespace delta::testing