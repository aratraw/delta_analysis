// include/delta/geometry/curvature.h
#pragma once

#include <numbers>
#include <cmath>
#include "connection.h"
#include "simplicial_complex.h"

namespace delta::geometry {

    // -----------------------------------------------------------------------------
    // Базовые функции (уже есть, оставляем)
    // -----------------------------------------------------------------------------

    /**
     * @brief Вычисление голономии вокруг грани (треугольника) симплициального комплекса.
     * @tparam Complex Тип, удовлетворяющий SimplicialComplex (должен иметь triangle_at).
     * @tparam Connection Тип связности.
     * @param mesh Сетка.
     * @param face_index Индекс грани (треугольника).
     * @param conn Связность.
     * @return Матрица голономии (произведение транспортов вдоль рёбер грани).
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
     * @param holonomy Матрица голономии.
     * @param area Площадь грани (должна быть > 0).
     * @return Матрица кривизны (элемент алгебры Ли).
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
    // Специализированные функции для 3D
    // -----------------------------------------------------------------------------

    /**
     * @brief Вычисление тензора Риччи в вершине для 3D симплициального комплекса.
     *
     * Тензор Риччи в вершине аппроксимируется как сумма по всем граням (треугольникам),
     * инцидентным вершине, от вклада кривизны грани, спроецированного на касательное
     * пространство. Возвращается симметричная матрица 3x3.
     *
     * @tparam Complex 3D симплициальный комплекс.
     * @tparam Connection Тип связности.
     * @tparam Metric Метрика (для вычисления площадей и нормалей).
     * @param mesh Сетка.
     * @param conn Связность.
     * @param vertex_index Индекс вершины.
     * @param metric Метрика.
     * @return Eigen::Matrix<typename Complex::scalar_type, 3, 3> тензор Риччи в вершине.
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

        // Собираем все треугольники, инцидентные вершине
        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto tri = mesh.triangle_at(t);
            // Проверяем, содержит ли треугольник данную вершину
            int pos = -1;
            for (int i = 0; i < 3; ++i) {
                if (tri[i] == vertex_index) {
                    pos = i;
                    break;
                }
            }
            if (pos == -1) continue;

            // Получаем голономию вокруг треугольника
            auto hol = holonomy_around_face(mesh, t, conn);
            // Вычисляем площадь треугольника
            auto a = mesh.vertex(tri[0]);
            auto b = mesh.vertex(tri[1]);
            auto c = mesh.vertex(tri[2]);
            Scalar area = triangle_area(a, b, c, metric);

            // Кривизна грани как элемент so(3)
            Matrix3 curv = curvature_from_holonomy(hol, area);

            // Для вклада в тензор Риччи в вершине нужно спроецировать кривизну
            // на касательную плоскость, но в дискретном случае часто используют
            // просто сумму кривизн граней с весами (например, 1/3 площади).
            // Более точный подход: учесть нормали.
            // Упростим: добавим кривизну, делённую на 3 (по числу вершин в треугольнике).
            ricci += curv / Scalar{ 3 };
        }

        // Симметризуем результат (тензор Риччи должен быть симметричным)
        return (ricci + ricci.transpose()) / Scalar{ 2 };
    }

    /**
     * @brief Скалярная кривизна в вершине для 3D (след тензора Риччи).
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

    /**
     * @brief Вычисление тензора Риччи в вершине для 4D симплициального комплекса.
     *
     * Аналогично 3D, но возвращает матрицу 4x4.
     */
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
            Scalar area = triangle_area(a, b, c, metric);
            Matrix4 curv = curvature_from_holonomy(hol, area);
            ricci += curv / Scalar{ 3 }; // усреднение по трём вершинам
        }

        return (ricci + ricci.transpose()) / Scalar{ 2 };
    }

    /**
     * @brief Скалярная кривизна в вершине для 4D.
     */
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
    // Вспомогательная функция для площади треугольника (обобщённая метрика)
    // -----------------------------------------------------------------------------
    namespace detail {
        template<typename Point, typename Metric>
        auto triangle_area(const Point& a, const Point& b, const Point& c, const Metric& metric) {
            // Используем формулу Герона
            auto ab = metric(a, b);
            auto bc = metric(b, c);
            auto ca = metric(c, a);
            auto s = (ab + bc + ca) / 2;
            using std::sqrt;
            return sqrt(s * (s - ab) * (s - bc) * (s - ca));
        }
    }

    template<typename Point, typename Metric>
    auto triangle_area(const Point& a, const Point& b, const Point& c, const Metric& metric) {
        return detail::triangle_area(a, b, c, metric);
    }

    // -----------------------------------------------------------------------------
    // Существующая функция vertex_curvature_deficit (2D) остаётся без изменений
    // -----------------------------------------------------------------------------
    template<typename Complex, typename Metric>
    typename Complex::value_type
        vertex_curvature_deficit(const Complex& mesh,
            std::size_t vertex_index,
            const Metric& metric) {
        using Coord = typename Complex::value_type;
        Coord sum_angles = Coord{ 0 };
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

            Coord a, b, c;
            if (pos == 0) {
                a = metric(p1, p0);
                b = metric(p2, p0);
                c = metric(p2, p1);
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
            if (a == Coord{ 0 } || b == Coord{ 0 }) continue;
            Coord cos_angle = (a * a + b * b - c * c) / (Coord{ 2 } * a * b);
            if (cos_angle > Coord{ 1 }) cos_angle = Coord{ 1 };
            if (cos_angle < Coord{ -1 }) cos_angle = Coord{ -1 };
            Coord angle(std::acos(cos_angle.template convert_to<double>()));
            sum_angles += angle;
        }
        Coord deficit = Coord{ 2 } * Coord(std::numbers::pi) - sum_angles;
        return deficit;
    }

} // namespace delta::geometry