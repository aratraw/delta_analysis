//tests/test_fixtures_geometry_numerical.h
#pragma once

#include <Eigen/Sparse>
#include "test_fixtures.h"
#include "delta/geometry/simplicial_complex.h"
#include "delta/geometry/geometry_ops.h"


namespace delta::testing {

    class GeometryNumericalTest : public DeltaTest {
    protected:
        using Scalar = Rational;

        static constexpr int DIM_VERTEX = 0;
        static constexpr int DIM_EDGE = 1;
        static constexpr int DIM_TRIANGLE = 2;
        static constexpr int DIM_TETRAHEDRON = 3;

        Rational old_precision;

        void SetUp() override {
            old_precision = delta::default_eps(); // запоминаем текущее значение точности
        }

        void TearDown() override {
            delta::default_eps_value() = old_precision; // восстанавливаем после вызова теста с изменённым требованием точности
        }

        static void set_precision(const Rational& eps) {
            delta::default_eps_value() = eps;
        }


        template<int Dim>
        using Complex = delta::geometry::SimplicialComplex<Dim, Scalar>;

        template<int Dim>
        using Point = typename Complex<Dim>::point_type;

        template<int Dim>
        using VertexIndex = typename Complex<Dim>::vertex_index;

        template<int Dim>
        using Edge = typename Complex<Dim>::edge_type;

        template<int Dim>
        using Triangle = typename Complex<Dim>::triangle_type;

        template<int Dim>
        using Tetrahedron = typename Complex<Dim>::tetrahedron_type;

        // -------------------------------------------------------------------------
        // Proxy methods for SimplicialComplex
        // -------------------------------------------------------------------------
        template<int Dim>
        VertexIndex<Dim> add_vertex(Complex<Dim>& mesh, const Point<Dim>& p) {
            return mesh.add_vertex(p);
        }

        template<int Dim>
        bool add_edge(Complex<Dim>& mesh, VertexIndex<Dim> v0, VertexIndex<Dim> v1) {
            return mesh.add_edge(v0, v1);
        }

        template<int Dim>
        bool add_triangle(Complex<Dim>& mesh, VertexIndex<Dim> v0, VertexIndex<Dim> v1, VertexIndex<Dim> v2) {
            return mesh.add_triangle(v0, v1, v2);
        }

        template<int Dim>
        bool add_tetrahedron(Complex<Dim>& mesh, VertexIndex<Dim> v0, VertexIndex<Dim> v1,
            VertexIndex<Dim> v2, VertexIndex<Dim> v3) {
            return mesh.add_tetrahedron(v0, v1, v2, v3);
        }

        template<int Dim>
        std::size_t num_vertices(const Complex<Dim>& mesh) const {
            return mesh.num_vertices();
        }

        template<int Dim>
        std::size_t num_edges(const Complex<Dim>& mesh) const {
            return mesh.num_edges();
        }

        template<int Dim>
        std::size_t num_triangles(const Complex<Dim>& mesh) const {
            return mesh.num_triangles();
        }

        template<int Dim>
        std::size_t num_tetrahedra(const Complex<Dim>& mesh) const {
            return mesh.num_tetrahedra();
        }

        template<int Dim>
        const Point<Dim>& vertex(const Complex<Dim>& mesh, VertexIndex<Dim> i) const {
            return mesh.vertex(i);
        }

        template<int Dim>
        Edge<Dim> edge_at(const Complex<Dim>& mesh, std::size_t idx) const {
            return mesh.edge_at(idx);
        }

        template<int Dim>
        Triangle<Dim> triangle_at(const Complex<Dim>& mesh, std::size_t idx) const {
            return mesh.triangle_at(idx);
        }

        template<int Dim>
        Tetrahedron<Dim> tetrahedron_at(const Complex<Dim>& mesh, std::size_t idx) const {
            return mesh.tetrahedron_at(idx);
        }

        template<int Dim>
        std::ptrdiff_t find_simplex(const Complex<Dim>& mesh, int dim,
            const std::vector<VertexIndex<Dim>>& vertices) const {
            return mesh.find_simplex(dim, vertices);
        }

        template<int Dim>
        auto comparator(const Complex<Dim>& mesh) const {
            return mesh.comparator();
        }

        template<int Dim>
        auto begin(const Complex<Dim>& mesh) const {
            return mesh.begin();
        }

        template<int Dim>
        auto end(const Complex<Dim>& mesh) const {
            return mesh.end();
        }

        // -------------------------------------------------------------------------
        // Helper to create unit square triangulation (2D)
        // -------------------------------------------------------------------------
        template<int Dim = 2>
        void make_unit_square_triangulation(Complex<Dim>& mesh) {
            static_assert(Dim == 2, "Unit square triangulation is for 2D only");
            using Point = typename Complex<Dim>::point_type;

            auto v0 = add_vertex(mesh, Point(0_r, 0_r));
            auto v1 = add_vertex(mesh, Point(1_r, 0_r));
            auto v2 = add_vertex(mesh, Point(1_r, 1_r));
            auto v3 = add_vertex(mesh, Point(0_r, 1_r));

            add_edge(mesh, v0, v1);
            add_edge(mesh, v1, v2);
            add_edge(mesh, v2, v0);
            add_edge(mesh, v0, v2);
            add_edge(mesh, v2, v3);
            add_edge(mesh, v3, v0);

            add_triangle(mesh, v0, v1, v2);
            add_triangle(mesh, v0, v2, v3);
        }

        // -------------------------------------------------------------------------
        // Wrappers for geometry_ops functions
        // -------------------------------------------------------------------------
        template<typename Complex, typename Metric>
        static auto edge_length(const Complex& mesh, std::size_t edge_idx, const Metric& metric) {
            return delta::geometry::edge_length(mesh, edge_idx, metric);
        }

        template<typename Complex>
        static auto edge_center(const Complex& mesh, std::size_t edge_idx) {
            return delta::geometry::edge_center(mesh, edge_idx);
        }

        template<typename Complex>
        static auto triangle_center(const Complex& mesh, std::size_t tri_idx) {
            return delta::geometry::triangle_center(mesh, tri_idx);
        }

        template<typename Complex, typename Metric>
        static auto triangle_area(const Complex& mesh, std::size_t tri_idx, const Metric& metric) {
            return delta::geometry::triangle_area(mesh, tri_idx, metric);
        }

        template<typename Complex, typename Metric>
        static auto tetrahedron_volume(const Complex& mesh, std::size_t tet_idx, const Metric& metric) {
            return delta::geometry::tetrahedron_volume(mesh, tet_idx, metric);
        }

        template<int Dim, typename Complex, typename Metric>
        static auto cell_volume(const Complex& mesh, std::size_t cell_idx, const Metric& metric) {
            return delta::geometry::cell_volume<Dim>(mesh, cell_idx, metric);
        }

        template<typename Complex, typename Metric>
        static auto edge_normal_2d(const Complex& mesh, std::size_t edge_idx, const Metric& metric) {
            return delta::geometry::edge_normal_2d(mesh, edge_idx, metric);
        }

        template<typename Complex>
        static auto edge_neighbors_2d(const Complex& mesh, std::size_t edge_idx) {
            return delta::geometry::edge_neighbors_2d(mesh, edge_idx);
        }

        // -------------------------------------------------------------------------
        // Utilities for matrix/vector comparison
        // -------------------------------------------------------------------------
        template<typename Derived>
        static bool matrix_near(const Eigen::DenseBase<Derived>& A,
            const Eigen::DenseBase<Derived>& B,
            double eps = 1e-12) {
            return (A - B).norm() <= eps;
        }

        template<typename Scalar>
        static bool sparse_matrix_near(const Eigen::SparseMatrix<Scalar>& A,
            const Eigen::SparseMatrix<Scalar>& B,
            Scalar eps = delta::default_eps()) {
            if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
            Eigen::SparseMatrix<Scalar> diff = A - B;
            diff.prune(eps);
            return diff.nonZeros() == 0;
        }

        template<int Dim>
        static bool vector_near(const Point<Dim>& a, const Point<Dim>& b,
            Scalar eps = delta::default_eps()) {
            return (a - b).squaredNorm() <= eps * eps;
        }

        // Random point generator
        template<int Dim>
        Point<Dim> random_point() {
            static std::mt19937 rng(42);
            static std::uniform_real_distribution<double> dist(0.0, 1.0);
            Point<Dim> p;
            for (int i = 0; i < Dim; ++i)
                p(i) = Scalar(dist(rng));
            return p;
        }
    };

#define EXPECT_SPARSE_NEAR(A, B, eps) \
    EXPECT_PRED3((::delta::testing::GeometryNumericalTest::sparse_matrix_near<decltype(A)::Scalar>), A, B, eps)

} // namespace delta::testing