// include/delta/numerical/discrete_operators.h
#pragma once

#include "delta/geometry/simplicial_complex.h"
#include "delta/core/regulative_idea.h"
#include "delta/numerical/concepts.h"
#include <vector>
#include <cmath>

namespace delta::numerical {

    // -------------------------------------------------------------------------
    // Вычисление двойственных площадей вершин (сумма 1/3 площадей прилегающих треугольников)
    // -------------------------------------------------------------------------
    /**
     * @brief Вычислить площади двойственных ячеек для всех вершин сетки (Voronoi).
     *
     * @tparam Complex Тип симплициального комплекса (2D).
     * @tparam Metric  Тип метрики.
     * @param mesh     Сетка.
     * @param metric   Метрика для вычисления площадей треугольников.
     * @return std::vector<typename Complex::scalar_type> Вектор площадей для каждой вершины.
     */
    template<typename Complex, typename Metric>
        requires FiniteElementGrid<Complex>&& IsMetric<Metric, typename Complex::point_type, typename Complex::scalar_type>
    std::vector<typename Complex::scalar_type> compute_vertex_dual_areas(
        const Complex& mesh,
        const Metric& metric)
    {
        using Scalar = typename Complex::scalar_type;
        std::size_t n_vertices = mesh.num_vertices();
        std::vector<Scalar> areas(n_vertices, Scalar{ 0 });

        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto tri = mesh.triangle_at(t);
            Scalar area = mesh.cell_volume(t, metric); // площадь треугольника через метрику
            Scalar third = area / Scalar{ 3 };
            areas[tri[0]] += third;
            areas[tri[1]] += third;
            areas[tri[2]] += third;
        }
        return areas;
    }

    // -------------------------------------------------------------------------
    // Дискретный градиент скалярного поля на вершинах (значения на рёбрах)
    // -------------------------------------------------------------------------
    /**
     * @brief Дискретный градиент скалярного поля на вершинах (возвращает значения на рёбрах).
     *
     * Для каждого ориентированного ребра (i,j) (как хранится в сетке, от меньшего индекса к большему)
     * вычисляет (f_j - f_i) / length(edge). Результирующий вектор имеет размер num_edges().
     *
     * @tparam Complex Тип симплициального комплекса.
     * @tparam Metric  Тип метрики.
     * @param mesh     Сетка.
     * @param vertex_values Скалярные значения на вершинах (std::vector или Eigen::VectorXd).
     * @param metric   Метрика для вычисления длин рёбер.
     * @return std::vector<Value> Значения градиента на рёбрах.
     */
    template<typename Complex, typename Metric, typename Container>
    auto discrete_gradient(const Complex& mesh,
        const Container& vertex_values,
        const Metric& metric)
    {
        using Value = typename Container::value_type;
        std::size_t n_edges = mesh.num_edges();
        std::vector<Value> gradient(n_edges);

        for (std::size_t e = 0; e < n_edges; ++e) {
            auto [v0, v1] = mesh.edge_at(e);
            Value len = mesh.edge_length(e, metric);
            if (len == Value{ 0 }) {
                gradient[e] = Value{ 0 };
            }
            else {
                gradient[e] = (vertex_values[v1] - vertex_values[v0]) / len;
            }
        }
        return gradient;
    }

    // -------------------------------------------------------------------------
    // "Сырая" дискретная дивергенция (без учёта площадей)
    // -------------------------------------------------------------------------
    /**
     * @brief Сырая дискретная дивергенция поля на рёбрах (1-формы) без масштабирования на площади.
     *
     * Для каждой вершины суммирует знаковые значения на рёбрах: вклад ребра (i,j) равен
     * +value для i, -value для j.
     *
     * @tparam Complex Тип симплициального комплекса.
     * @param mesh     Сетка.
     * @param edge_values Значения на рёбрах (std::vector или Eigen::VectorXd).
     * @return std::vector<Value> Сырая дивергенция в вершинах.
     */
    template<typename Complex, typename Container>
    std::vector<typename Container::value_type> discrete_divergence_raw(
        const Complex& mesh,
        const Container& edge_values)
    {
        using Value = typename Container::value_type;
        std::size_t n_vertices = mesh.num_vertices();
        std::vector<Value> div(n_vertices, Value{ 0 });

        for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
            auto [v0, v1] = mesh.edge_at(e);
            div[v0] += edge_values[e];
            div[v1] -= edge_values[e];
        }
        return div;
    }

    // -------------------------------------------------------------------------
    // Дискретная дивергенция с учётом площадей двойственных ячеек
    // -------------------------------------------------------------------------
    /**
     * @brief Дискретная дивергенция поля на рёбрах, нормированная на площади двойственных ячеек.
     *
     * Вычисляет (δ ω)(v) = (1 / A_v) * raw_divergence(v), где A_v — площадь двойственной ячейки (Voronoi).
     *
     * @tparam Complex Тип симплициального комплекса.
     * @tparam Metric  Тип метрики.
     * @param mesh     Сетка.
     * @param edge_values Значения на рёбрах.
     * @param metric   Метрика для вычисления площадей.
     * @return std::vector<Value> Дивергенция в вершинах.
     */
    template<typename Complex, typename Metric, typename Container>
    auto discrete_divergence(const Complex& mesh,
        const Container& edge_values,
        const Metric& metric)
    {
        using Value = typename Container::value_type;
        std::size_t n_vertices = mesh.num_vertices();

        auto vertex_areas = compute_vertex_dual_areas(mesh, metric);
        auto raw = discrete_divergence_raw(mesh, edge_values);

        std::vector<Value> result(n_vertices);
        for (std::size_t i = 0; i < n_vertices; ++i) {
            if (vertex_areas[i] != Value{ 0 }) {
                result[i] = raw[i] / vertex_areas[i];
            }
            else {
                result[i] = Value{ 0 };
            }
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Дискретный лапласиан (котангенсный) для скалярного поля на вершинах
    // -------------------------------------------------------------------------
    /**
     * @brief Дискретный лапласиан скалярного поля на вершинах (Δf = div grad f).
     *
     * Использует котангенсные веса (стандартный лапласиан DEC для 0-форм).
     *
     * @tparam Complex Тип симплициального комплекса (2D).
     * @tparam Metric  Тип метрики.
     * @param mesh     Сетка.
     * @param vertex_values Скалярные значения на вершинах.
     * @param metric   Метрика для вычисления длин сторон и площадей.
     * @return std::vector<Value> Значения лапласиана в вершинах.
     */
    template<typename Complex, typename Metric, typename Container>
    std::vector<typename Container::value_type> discrete_laplacian_cotangent(
        const Complex& mesh,
        const Container& vertex_values,
        const Metric& metric)
    {
        using Value = typename Container::value_type;
        using Scalar = typename Complex::scalar_type;
        std::size_t n_vertices = mesh.num_vertices();
        std::vector<Value> laplacian(n_vertices, Value{ 0 });

        // Предвычисляем площади вершин (нужны для нормировки)
        auto vertex_areas = compute_vertex_dual_areas(mesh, metric);

        // Проходим по всем треугольникам и накапливаем вклады в лапласиан
        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto tri = mesh.triangle_at(t);
            std::size_t v0 = tri[0], v1 = tri[1], v2 = tri[2];

            // Длины сторон через метрику (используем edge_length)
            std::size_t e01 = static_cast<std::size_t>(mesh.find_simplex(1, { v0, v1 }));
            std::size_t e12 = static_cast<std::size_t>(mesh.find_simplex(1, { v1, v2 }));
            std::size_t e20 = static_cast<std::size_t>(mesh.find_simplex(1, { v2, v0 }));

            Scalar len01 = mesh.edge_length(e01, metric);
            Scalar len12 = mesh.edge_length(e12, metric);
            Scalar len20 = mesh.edge_length(e20, metric);

            // Площадь треугольника
            Scalar area = mesh.cell_volume(t, metric);
            if (area <= Scalar{ 0 }) continue;

            // Котангенсы углов через длины сторон
            Scalar cot0 = (len12 * len12 + len20 * len20 - len01 * len01) / (Scalar{ 4 } * area);
            Scalar cot1 = (len20 * len20 + len01 * len01 - len12 * len12) / (Scalar{ 4 } * area);
            Scalar cot2 = (len01 * len01 + len12 * len12 - len20 * len20) / (Scalar{ 4 } * area);

            // Вклад в лапласиан для каждой вершины: 0.5 * cot(угла) * (u_i - u_j) для смежных вершин
            // Накопление без деления на площади (потом разделим)
            laplacian[v0] += cot1 * (vertex_values[v0] - vertex_values[v2])   // через угол при v1? Проверим индексы:
                + cot2 * (vertex_values[v0] - vertex_values[v1]); // обычно: для вершины v0, вклад от двух прилегающих рёбер: (v0-v1)*cot(угол при v2) + (v0-v2)*cot(угол при v1)
            laplacian[v1] += cot2 * (vertex_values[v1] - vertex_values[v0]) + cot0 * (vertex_values[v1] - vertex_values[v2]);
            laplacian[v2] += cot0 * (vertex_values[v2] - vertex_values[v1]) + cot1 * (vertex_values[v2] - vertex_values[v0]);
        }

        // Нормировка на площади вершин
        for (std::size_t i = 0; i < n_vertices; ++i) {
            if (vertex_areas[i] != Value{ 0 }) {
                laplacian[i] /= (Value{ 2 } * vertex_areas[i]); // коэффициент 1/2 из стандартной формулы
            }
        }
        return laplacian;
    }

} // namespace delta::numerical