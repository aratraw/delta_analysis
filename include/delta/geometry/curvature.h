// include/delta/geometry/curvature.h
#pragma once

#include <numbers>
#include <cmath>
#include "connection.h"
#include "simplicial_complex.h"

namespace delta::geometry {

    // -----------------------------------------------------------------------------
    // Базовые функции
    // -----------------------------------------------------------------------------

    /**
     * @brief Вычисление голономии вокруг грани (треугольника) симплициального комплекса.
     */
    template<typename Complex, typename Connection>
    auto holonomy_around_face(const Complex& mesh,
        std::size_t face_index,
        const Connection& conn) {
        using matrix_type = typename Connection::matrix_type;
        auto tri = mesh.triangle_at(face_index);
        std::vector<typename Connection::edge_type> edges = {
            {tri[0], tri[1]},
            {tri[1], tri[2]},
            {tri[2], tri[0]}
        };
        return conn.holonomy(edges);
    }

    /**
     * @brief Приближение тензора кривизны из голономии.
     */
    template<typename Matrix>
    Matrix curvature_from_holonomy(const Matrix& holonomy,
        const typename Matrix::Scalar& area) {
        if (area == typename Matrix::Scalar{ 0 }) {
            return Matrix::Zero();
        }
        return (holonomy - Matrix::Identity()) / area;
    }

    // -----------------------------------------------------------------------------
    // Вспомогательная функция для площади треугольника (использует метрику)
    // -----------------------------------------------------------------------------
    namespace detail {
        template<typename Point, typename Metric>
        auto triangle_area(const Point& a, const Point& b, const Point& c, const Metric& metric) {
            using Scalar = decltype(metric(a, b));
            Scalar ab = metric(a, b);
            Scalar bc = metric(b, c);
            Scalar ca = metric(c, a);
            Scalar s = (ab + bc + ca) / Scalar{ 2 };
            using std::sqrt;
            return sqrt(s * (s - ab) * (s - bc) * (s - ca));
        }
    } // namespace detail

    // -----------------------------------------------------------------------------
    // Специализированные функции для 3D
    // -----------------------------------------------------------------------------

    /**
     * @brief Вычисление тензора Риччи в вершине для 3D симплициального комплекса.
     */
    template<typename Complex, typename Connection, typename Metric>
    Eigen::Matrix<typename Complex::scalar_type, 3, 3>
        vertex_ricci_curvature_3d(const Complex& mesh,
            const Connection& conn,
            std::size_t vertex_index,
            const Metric& metric) {
        using Scalar = typename Complex::scalar_type;
        using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
        Matrix3 ricci = Matrix3::Zero();

        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto tri = mesh.triangle_at(t);
            int pos = -1;
            for (int i = 0; i < 3; ++i) {
                if (tri[i] == vertex_index) {
                    pos = i;
                    break;
                }
            }
            if (pos == -1) continue;

            auto hol = holonomy_around_face(mesh, t, conn);
            auto a = mesh.vertex(tri[0]);
            auto b = mesh.vertex(tri[1]);
            auto c = mesh.vertex(tri[2]);
            Scalar area = detail::triangle_area(a, b, c, metric);
            Matrix3 curv = curvature_from_holonomy(hol, area);
            ricci += curv / Scalar{ 3 };
        }

        return (ricci + ricci.transpose()) / Scalar{ 2 };
    }

    /**
     * @brief Скалярная кривизна в вершине для 3D.
     */
    template<typename Complex, typename Connection, typename Metric>
    typename Complex::scalar_type
        vertex_scalar_curvature_3d(const Complex& mesh,
            const Connection& conn,
            std::size_t vertex_index,
            const Metric& metric) {
        auto ricci = vertex_ricci_curvature_3d(mesh, conn, vertex_index, metric);
        return ricci.trace();
    }

    // -----------------------------------------------------------------------------
    // Специализированные функции для 4D
    // -----------------------------------------------------------------------------

    template<typename Complex, typename Connection, typename Metric>
    Eigen::Matrix<typename Complex::scalar_type, 4, 4>
        vertex_ricci_curvature_4d(const Complex& mesh,
            const Connection& conn,
            std::size_t vertex_index,
            const Metric& metric) {
        using Scalar = typename Complex::scalar_type;
        using Matrix4 = Eigen::Matrix<Scalar, 4, 4>;
        Matrix4 ricci = Matrix4::Zero();

        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto tri = mesh.triangle_at(t);
            int pos = -1;
            for (int i = 0; i < 3; ++i) {
                if (tri[i] == vertex_index) {
                    pos = i;
                    break;
                }
            }
            if (pos == -1) continue;

            auto hol = holonomy_around_face(mesh, t, conn);
            auto a = mesh.vertex(tri[0]);
            auto b = mesh.vertex(tri[1]);
            auto c = mesh.vertex(tri[2]);
            Scalar area = detail::triangle_area(a, b, c, metric);
            Matrix4 curv = curvature_from_holonomy(hol, area);
            ricci += curv / Scalar{ 3 };
        }

        return (ricci + ricci.transpose()) / Scalar{ 2 };
    }

    template<typename Complex, typename Connection, typename Metric>
    typename Complex::scalar_type
        vertex_scalar_curvature_4d(const Complex& mesh,
            const Connection& conn,
            std::size_t vertex_index,
            const Metric& metric) {
        auto ricci = vertex_ricci_curvature_4d(mesh, conn, vertex_index, metric);
        return ricci.trace();
    }

    // -----------------------------------------------------------------------------
    // Существующая функция vertex_curvature_deficit (2D) обновлена для использования метрики
    // -----------------------------------------------------------------------------
    template<typename Complex, typename Metric>
    typename Complex::scalar_type
        vertex_curvature_deficit(const Complex& mesh,
            std::size_t vertex_index,
            const Metric& metric) {
        using Scalar = typename Complex::scalar_type;
        Scalar sum_angles = Scalar{ 0 };

        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto tri = mesh.triangle_at(t);
            int pos = -1;
            for (int i = 0; i < 3; ++i) {
                if (tri[i] == vertex_index) {
                    pos = i;
                    break;
                }
            }
            if (pos == -1) continue;

            auto p0 = mesh.vertex(tri[0]);
            auto p1 = mesh.vertex(tri[1]);
            auto p2 = mesh.vertex(tri[2]);

            Scalar a, b, c; // длины сторон, образующих угол при вершине vertex_index
            if (pos == 0) {
                a = metric(p1, p0); // сторона vertex-p1
                b = metric(p2, p0); // сторона vertex-p2
                c = metric(p2, p1); // противоположная сторона
            }
            else if (pos == 1) {
                a = metric(p0, p1);
                b = metric(p2, p1);
                c = metric(p2, p0);
            }
            else { // pos == 2
                a = metric(p0, p2);
                b = metric(p1, p2);
                c = metric(p1, p0);
            }

            if (a == Scalar{ 0 } || b == Scalar{ 0 }) continue;

            // Теорема косинусов: cos(angle) = (a^2 + b^2 - c^2) / (2ab)
            Scalar cos_angle = (a * a + b * b - c * c) / (Scalar{ 2 } * a * b);
            if (cos_angle > Scalar{ 1 }) cos_angle = Scalar{ 1 };
            if (cos_angle < Scalar{ -1 }) cos_angle = Scalar{ -1 };
            Scalar angle = Scalar(std::acos(cos_angle.template convert_to<double>()));
            sum_angles += angle;
        }

        Scalar deficit = Scalar{ 2 } * Scalar(std::numbers::pi) - sum_angles;
        return deficit;
    }

} // namespace delta::geometry