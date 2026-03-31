// tests/test_fixtures_geometry_numerical.h
#pragma once

#include <Eigen/Sparse>
#include <random>
#include <optional>
#include <iomanip>
#include <sstream>
#include "test_fixtures.h"
#include "delta/core/rational.h"
#include "delta/geometry/simplicial_complex.h"
#include "delta/geometry/constructive_core.h"
#include "delta/geometry/product_regulative.h"
#include "delta/geometry/matrix_field.h"
#include "delta/geometry/tensor_field.h"
#include "delta/numerical/discrete_operators.h"

namespace delta::testing {
    using namespace delta;
    using namespace delta::geometry;
    using namespace delta::numerical;
    /**
     * @brief Test fixture for Stage 0 geometry modules.
     *
     * Provides type aliases and proxy methods for:
     *   - SimplicialComplex
     *   - ConstructiveCore (Point, Vector, K)
     *   - ProductRegulativeIdea and ProductDeltaPath
     *
     * Also includes utilities for matrix/vector comparison and
     * random point generation.
     */
    class GeometryNumericalTest : public DeltaTest {
    protected:
        using Scalar = Rational;

        // Dimension constants (matching SimplicialComplex)
        static constexpr int DIM_VERTEX = 0;
        static constexpr int DIM_EDGE = 1;
        static constexpr int DIM_TRIANGLE = 2;
        static constexpr int DIM_TETRAHEDRON = 3;

        // -------------------------------------------------------------------------
        // SimplicialComplex related types and proxies
        // -------------------------------------------------------------------------
        template<int Dim>
        using Complex = delta::geometry::SimplicialComplex<Dim, Scalar>;

        template<int Dim>
        using Point = typename Complex<Dim>::point_type;  // Это Eigen::Matrix

        template<int Dim>
        using VertexIndex = typename Complex<Dim>::vertex_index;

        template<int Dim>
        using Edge = typename Complex<Dim>::edge_type;

        template<int Dim>
        using Triangle = typename Complex<Dim>::triangle_type;

        template<int Dim>
        using Tetrahedron = typename Complex<Dim>::tetrahedron_type;

        // Proxies for SimplicialComplex construction
        template<int Dim>
        VertexIndex<Dim> add_vertex(Complex<Dim>& mesh, const Point<Dim>& p) {
            return mesh.add_vertex(p);
        }

        template<int Dim>
        bool add_edge(Complex<Dim>& mesh, VertexIndex<Dim> v0, VertexIndex<Dim> v1) {
            return mesh.add_edge(v0, v1);
        }

        template<int Dim>
        bool add_triangle(Complex<Dim>& mesh,
            VertexIndex<Dim> v0,
            VertexIndex<Dim> v1,
            VertexIndex<Dim> v2) {
            return mesh.add_triangle(v0, v1, v2);
        }

        template<int Dim>
        std::enable_if_t<Dim >= 3, bool> add_tetrahedron(Complex<Dim>& mesh,
            VertexIndex<Dim> v0,
            VertexIndex<Dim> v1,
            VertexIndex<Dim> v2,
            VertexIndex<Dim> v3) {
            return mesh.add_tetrahedron(v0, v1, v2, v3);
        }

        // Accessors
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
        std::enable_if_t<Dim >= 3, Tetrahedron<Dim>> tetrahedron_at(const Complex<Dim>& mesh, std::size_t idx) const {
            return mesh.tetrahedron_at(idx);
        }

        template<int Dim>
        std::ptrdiff_t find_simplex(const Complex<Dim>& mesh,
            int dim,
            const std::vector<VertexIndex<Dim>>& vertices) const {
            return mesh.find_simplex(dim, vertices);
        }

        // Geometric queries (methods of SimplicialComplex)
        template<typename Complex, typename Metric>
        static auto edge_length(const Complex& mesh, std::size_t edge_idx, const Metric& metric) {
            return mesh.edge_length(edge_idx, metric);
        }

        template<typename Complex, typename Metric>
        static auto cell_volume(const Complex& mesh, std::size_t cell_idx, const Metric& metric) {
            return mesh.cell_volume(cell_idx, metric);
        }

        // 2D-specific geometric queries
        template<typename Complex>
        static auto edge_neighbors_2d(const Complex& mesh, std::size_t edge_idx) {
            return mesh.edge_neighbors_2d(edge_idx);
        }

        template<typename Complex, typename Metric>
        static auto edge_normal_2d(const Complex& mesh, std::size_t edge_idx, const Metric& metric) {
            return mesh.edge_normal_2d(edge_idx, metric);
        }

        // Incidence and subdivision
        template<typename Complex>
        static auto incident_faces(const Complex& mesh,
            int top_dim,
            std::size_t idx,
            int low_dim) {
            return mesh.incident_faces(top_dim, idx, low_dim);
        }

        template<typename Complex>
        static auto barycentric_subdivide(const Complex& mesh) {
            return mesh.barycentric_subdivide();
        }

        // -------------------------------------------------------------------------
        // Constructive Core types and proxies - ВАРИАНТ А (Point = Eigen::Matrix)
        // -------------------------------------------------------------------------
        template<int Dim>
        using Vector = delta::geometry::Vector<Scalar, Dim>;

        // Check if a point belongs to the constructive core K - перегрузки для 2D и 3D
        static bool is_in_K(const Eigen::Matrix<Scalar, 2, 1>& p) {
            return delta::geometry::is_in_K(p);
        }
        static bool is_in_K(const Eigen::Matrix<Scalar, 3, 1>& p) {
            return delta::geometry::is_in_K(p);
        }

        // Operations on points and vectors - перегрузки для 2D и 3D
        static Vector<2> point_minus_point(const Eigen::Matrix<Scalar, 2, 1>& a,
            const Eigen::Matrix<Scalar, 2, 1>& b) {
            return delta::geometry::operator-(a, b);
        }
        static Vector<3> point_minus_point(const Eigen::Matrix<Scalar, 3, 1>& a,
            const Eigen::Matrix<Scalar, 3, 1>& b) {
            return delta::geometry::operator-(a, b);
        }

        static std::optional<Eigen::Matrix<Scalar, 2, 1>> point_plus_vector(
            const Eigen::Matrix<Scalar, 2, 1>& p,
            const Vector<2>& v) {
            return delta::geometry::operator+(p, v);
        }
        static std::optional<Eigen::Matrix<Scalar, 3, 1>> point_plus_vector(
            const Eigen::Matrix<Scalar, 3, 1>& p,
            const Vector<3>& v) {
            return delta::geometry::operator+(p, v);
        }

        static Vector<2> vector_plus_vector(const Vector<2>& u, const Vector<2>& v) {
            return u + v;
        }
        static Vector<3> vector_plus_vector(const Vector<3>& u, const Vector<3>& v) {
            return u + v;
        }

        static Vector<2> scalar_times_vector(const Scalar& s, const Vector<2>& v) {
            return s * v;
        }
        static Vector<3> scalar_times_vector(const Scalar& s, const Vector<3>& v) {
            return s * v;
        }

        // Finite base numbers (static methods only)
        template<int Base>
        static bool is_representable(const Scalar& x) {
            return delta::geometry::FiniteBaseNumbers<Base>::is_representable(x);
        }

        static bool is_in_universal_core(const Scalar& x) {
            return delta::geometry::is_in_universal_core(x);
        }

        // -------------------------------------------------------------------------
        // Product Regulative Idea proxies
        // -------------------------------------------------------------------------
        template<typename RI1, typename RI2>
        using ProductIdea = delta::geometry::ProductRegulativeIdea<RI1, RI2>;

        template<typename RI1, typename RI2, typename Addr>
        static bool product_betweenness(const ProductIdea<RI1, RI2>& idea,
            const Addr& x,
            const Addr& y,
            const Addr& z) {
            return idea.betweenness()(x, y, z);
        }

        template<typename RI1, typename RI2, typename Addr>
        static auto product_metric(const ProductIdea<RI1, RI2>& idea,
            const Addr& a,
            const Addr& b) {
            return idea.metric()(a, b);
        }

        // -------------------------------------------------------------------------
        // ProductDeltaPath proxies
        // -------------------------------------------------------------------------
        using Path1D = delta::DeltaPath<
            Rational,                                   // Addr
            Rational,                                   // Value
            Rational,                                   // Distance
            delta::LessBetweenness,                     // Betweenness
            delta::EuclideanMetric,                     // Metric
            delta::EuclideanValueMetric,                // ValueMetric
            delta::StaticStrategy<delta::MidpointOperator>,  // Strategy
            std::less<Rational>                         // Compare
        >;

        using Path2D = delta::geometry::ProductDeltaPath<Path1D, Path1D>;

        // Тип функции для 2D продукта: принимает массив адресов, возвращает массив значений
        using Path2DFunc = std::function<std::array<Rational, 2>(const std::array<Rational, 2>&)>;

        static void product_path_advance(Path2D& path, const Path2DFunc& func) {
            path.advance(func);
        }

        static auto product_path_current_grid(const Path2D& path) {
            return path.current_grid();
        }

        static std::size_t product_path_level(const Path2D& path) {
            return path.level();
        }

        template<typename Metric>
        static auto product_path_max_gap(const Path2D& path, const Metric& metric) {
            return path.max_gap(metric);
        }

        // -------------------------------------------------------------------------
        // Helper: unit square triangulation (2D)
        // -------------------------------------------------------------------------
        template<int Dim = 2>
        void make_unit_square_triangulation(Complex<Dim>& mesh) {
            static_assert(Dim == 2, "Unit square triangulation is for 2D only");
            using Pt = Point<Dim>;

            auto v0 = add_vertex(mesh, Pt(0_r, 0_r));
            auto v1 = add_vertex(mesh, Pt(1_r, 0_r));
            auto v2 = add_vertex(mesh, Pt(1_r, 1_r));
            auto v3 = add_vertex(mesh, Pt(0_r, 1_r));

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
        // Utilities for matrix/vector comparison (Eigen based)
        // -------------------------------------------------------------------------
        template<typename Derived>
        static bool matrix_near(const Eigen::DenseBase<Derived>& A,
            const Eigen::DenseBase<Derived>& B,
            const Scalar& eps = delta::default_eps()) {
            return (A - B).norm() <= eps;
        }

        template<typename Scalar>
        static bool sparse_matrix_near(const Eigen::SparseMatrix<Scalar>& A,
            const Eigen::SparseMatrix<Scalar>& B,
            const Scalar& eps = delta::default_eps()) {
            if (A.rows() != B.rows() || A.cols() != B.cols())
                return false;
            Eigen::SparseMatrix<Scalar> diff = A - B;
            diff.prune(eps);
            return diff.nonZeros() == 0;
        }

        template<int Dim>
        static bool vector_near(const Eigen::Matrix<Scalar, Dim, 1>& a,
            const Eigen::Matrix<Scalar, Dim, 1>& b,
            const Scalar& eps = delta::default_eps()) {
            return (a - b).squaredNorm() <= eps * eps;
        }

        // -------------------------------------------------------------------------
        // Random point generator - безопасное преобразование double -> Rational
        // -------------------------------------------------------------------------
        template<int Dim>
        Eigen::Matrix<Scalar, Dim, 1> random_point() {
            static std::mt19937 rng(42);
            static std::uniform_real_distribution<double> dist(0.0, 1.0);
            Eigen::Matrix<Scalar, Dim, 1> p;
            for (int i = 0; i < Dim; ++i) {
                double d = dist(rng);
                std::stringstream ss;
                ss << std::setprecision(std::numeric_limits<double>::max_digits10) << d;
                p(i) = Scalar(ss.str());
            }
            return p;
        }

        // -------------------------------------------------------------------------
        // Stage 1 Fixture Updates: Tensor Fields, Matrix Fields,
        // Discrete Operators, Integrals, Dual Complex.
        // -------------------------------------------------------------------------

        // Сравнение двух Eigen-матриц одинакового размера
        template<typename Derived>
        static bool matrix_near(const Eigen::MatrixBase<Derived>& A,
            const Eigen::MatrixBase<Derived>& B,
            const Scalar& eps = delta::default_eps()) {
            return (A - B).norm() <= eps;
        }

        // Генерация случайной матрицы заданного размера со значениями в [0,1]
        template<int Rows, int Cols>
        static Eigen::Matrix<Scalar, Rows, Cols> random_matrix() {
            static std::mt19937 rng(42);
            static std::uniform_real_distribution<double> dist(0.0, 1.0);
            Eigen::Matrix<Scalar, Rows, Cols> m;
            for (int i = 0; i < Rows; ++i) {
                for (int j = 0; j < Cols; ++j) {
                    double d = dist(rng);
                    std::stringstream ss;
                    ss << std::setprecision(std::numeric_limits<double>::max_digits10) << d;
                    m(i, j) = Scalar(ss.str());
                }
            }
            return m;
        }

        // Генерация случайного скаляра (ранг 0)
        static Scalar random_scalar() {
            return random_matrix<1, 1>()(0, 0);
        }

        // Компаратор для точек (лексикографический)
        template<int Dim>
        struct PointLess {
            bool operator()(const Point<Dim>& a, const Point<Dim>& b) const {
                for (int i = 0; i < Dim; ++i) {
                    if (a[i] < b[i]) return true;
                    if (b[i] < a[i]) return false;
                }
                return false; // равны
            }
        };

        // -------------------------------------------------------------------------
        // Precision management (inherit from DeltaTest, but we add convenience)
        // -------------------------------------------------------------------------
        void SetUp() override {
            old_precision_ = delta::default_eps();
        }

        void TearDown() override {
            delta::set_default_eps(old_precision_);
        }

        static void set_precision(const Rational& eps) {
            delta::set_default_eps(eps);
        }

    private:
        Rational old_precision_;
    };

    // Convenience macro for sparse matrix comparison
#define EXPECT_SPARSE_NEAR(A, B, eps) \
    EXPECT_PRED3((::delta::testing::GeometryNumericalTest::sparse_matrix_near<decltype(A)::Scalar>), A, B, eps)

} // namespace delta::testing