//include/delta/geometry/geometry_ops.h
#pragma once

#include <optional>
#include <cmath>
#include "simplicial_complex.h"
#include "delta/core/regulative_idea.h"

namespace delta::geometry {

    // -----------------------------------------------------------------------------
    // Длина ребра
    // -----------------------------------------------------------------------------
    template<typename Complex, typename Metric>
        requires delta::Metric<Metric, typename Complex::point_type>
    auto edge_length(const Complex& mesh, std::size_t edge_idx, const Metric& metric) {
        auto [v0, v1] = mesh.edge_at(edge_idx);
        return metric(mesh.vertex(v0), mesh.vertex(v1));
    }

    // -----------------------------------------------------------------------------
    // Центр ребра (требует LinearAddress)
    // -----------------------------------------------------------------------------
    template<typename Complex>
        requires delta::LinearAddress<typename Complex::point_type>
    auto edge_center(const Complex& mesh, std::size_t edge_idx) {
        auto [v0, v1] = mesh.edge_at(edge_idx);
        return (mesh.vertex(v0) + mesh.vertex(v1)) / typename Complex::scalar_type{ 2 };
    }

    // -----------------------------------------------------------------------------
    // Центр треугольника
    // -----------------------------------------------------------------------------
    template<typename Complex>
        requires delta::LinearAddress<typename Complex::point_type>
    auto triangle_center(const Complex& mesh, std::size_t tri_idx) {
        auto tri = mesh.triangle_at(tri_idx);
        return (mesh.vertex(tri[0]) + mesh.vertex(tri[1]) + mesh.vertex(tri[2])) / typename Complex::scalar_type{ 3 };
    }

    // -----------------------------------------------------------------------------
    // Площадь треугольника по формуле Герона
    // -----------------------------------------------------------------------------
    template<typename Complex, typename Metric>
        requires delta::Metric<Metric, typename Complex::point_type>
    auto triangle_area(const Complex& mesh, std::size_t tri_idx, const Metric& metric) {
        using Scalar = typename Complex::scalar_type;
        auto tri = mesh.triangle_at(tri_idx);
        Scalar a = metric(mesh.vertex(tri[0]), mesh.vertex(tri[1]));
        Scalar b = metric(mesh.vertex(tri[1]), mesh.vertex(tri[2]));
        Scalar c = metric(mesh.vertex(tri[2]), mesh.vertex(tri[0]));
        Scalar s = (a + b + c) / Scalar{ 2 };
        using delta::sqrt;
        // БЕЗ УКАЗАНИЯ ТРЕБУЕМОЙ ТОЧНОСТИ ОН БУДЕТ СЧИТАТЬ ДЛЯ MULTIPRECISION С ОХРЕНЕННОЙ ТОЧНОСТЬЮ НО ДОЛГО
        //ПЕРЕОПРЕДЕЛЯЙТЕ ТОЧНОСТЬ В .cpp или фикстурах - ближе к исполнению.
        return sqrt(s * (s - a) * (s - b) * (s - c));
    }

    // -----------------------------------------------------------------------------
    // Объём тетраэдра через определитель Кэли-Менгера
    // -----------------------------------------------------------------------------
    template<typename Complex, typename Metric>
        requires delta::Metric<Metric, typename Complex::point_type>
    auto tetrahedron_volume(const Complex& mesh, std::size_t tet_idx, const Metric& metric) {
        using Scalar = typename Complex::scalar_type;
        auto tet = mesh.tetrahedron_at(tet_idx);
        auto a = mesh.vertex(tet[0]), b = mesh.vertex(tet[1]),
            c = mesh.vertex(tet[2]), d = mesh.vertex(tet[3]);

        Scalar ab = metric(a, b);
        Scalar ac = metric(a, c);
        Scalar ad = metric(a, d);
        Scalar bc = metric(b, c);
        Scalar bd = metric(b, d);
        Scalar cd = metric(c, d);

        Eigen::Matrix<Scalar, 5, 5> M;
        M << 0, 1, 1, 1, 1,
            1, 0, ab* ab, ac* ac, ad* ad,
            1, ab* ab, 0, bc* bc, bd* bd,
            1, ac* ac, bc* bc, 0, cd* cd,
            1, ad* ad, bd* bd, cd* cd, 0;

        Scalar det = M.determinant();
        using delta::sqrt;
        // БЕЗ УКАЗАНИЯ ТРЕБУЕМОЙ ТОЧНОСТИ ОН БУДЕТ СЧИТАТЬ ДЛЯ MULTIPRECISION С ОХРЕНЕННОЙ ТОЧНОСТЬЮ НО ДОЛГО
        //ПЕРЕОПРЕДЕЛЯЙТЕ ТОЧНОСТЬ В .cpp или фикстурах - ближе к исполнению.
        return sqrt(det / 288);
    }

    // -----------------------------------------------------------------------------
    // Диспетчер объёма ячейки по размерности
    // -----------------------------------------------------------------------------
    template<int Dim, typename Complex, typename Metric>
        requires (Dim == 2 || Dim == 3) && delta::Metric<Metric, typename Complex::point_type>
    auto cell_volume(const Complex& mesh, std::size_t cell_idx, const Metric& metric) {
        if constexpr (Dim == 2) return triangle_area(mesh, cell_idx, metric);
        else return tetrahedron_volume(mesh, cell_idx, metric);
    }

    // -----------------------------------------------------------------------------
    // Нормаль к ребру в 2D (вектор длины, равной длине ребра, ортогонален ребру)
    // -----------------------------------------------------------------------------
    template<typename Complex, typename Metric>
        requires delta::Metric<Metric, typename Complex::point_type>
    auto edge_normal_2d(const Complex& mesh, std::size_t edge_idx, const Metric& metric) {
        static_assert(Complex::Dimension == 2, "edge_normal_2d only for 2D");
        using Point = typename Complex::point_type;
        using Scalar = typename Complex::scalar_type;

        auto [v0, v1] = mesh.edge_at(edge_idx);
        Point e = mesh.vertex(v1) - mesh.vertex(v0);
        // Поворот на -90°: (dx, dy) -> (dy, -dx)
        Point normal(e.y(), -e.x());

        Scalar euclidean_len = e.norm();
        if (euclidean_len > 0) {
            Scalar metric_len = metric(mesh.vertex(v0), mesh.vertex(v1));
            normal *= (metric_len / euclidean_len);
        }
        return normal;
    }

    // -----------------------------------------------------------------------------
    // Соседи ребра в 2D: левый и правый треугольники
    // -----------------------------------------------------------------------------
    template<typename Complex>
    std::pair<std::size_t, std::optional<std::size_t>>
        edge_neighbors_2d(const Complex& mesh, std::size_t edge_idx) {
        static_assert(Complex::Dimension == 2, "edge_neighbors_2d only for 2D");

        auto [v0, v1] = mesh.edge_at(edge_idx);
        std::optional<std::size_t> left, right;

        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto tri = mesh.triangle_at(t);
            for (int i = 0; i < 3; ++i) {
                if ((tri[i] == v0 && tri[(i + 1) % 3] == v1) || (tri[i] == v1 && tri[(i + 1) % 3] == v0)) {
                    bool positive = (tri[i] == v0 && tri[(i + 1) % 3] == v1);
                    if (positive) left = t;
                    else right = t;
                    break;
                }
            }
        }

        // Если левого нет, но правый есть – меняем местами
        if (!left && right) {
            left = right;
            right = std::nullopt;
        }

        return { left.value_or(0), right };
    }

} // namespace delta::geometry