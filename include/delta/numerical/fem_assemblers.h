// include/delta/numerical/fem_assemblers.h
#pragma once

#include "delta/geometry/simplicial_complex.h"
#include "delta/numerical/cotangent_laplacian.h"
#include "delta/numerical/concepts.h"
#include <Eigen/Sparse>
#include <vector>
#include <cmath>

namespace delta::numerical {

    /**
     * @brief Собрать массовую матрицу для линейных элементов на симплициальном комплексе (2D).
     *
     * Для треугольников в 2D используется формула:
     *   M_ii = area / 6
     *   M_ij = area / 12   для i ≠ j (смежные вершины в треугольнике)
     *
     * @tparam Complex Тип, удовлетворяющий FiniteElementGrid (точки должны иметь операторы + и *).
     * @tparam Metric  Тип метрики.
     * @param mesh Сетка.
     * @param metric Метрика для вычисления площадей треугольников.
     * @return Разрежённая матрица размера num_vertices() x num_vertices().
     */
    template<typename Complex, typename Metric>
        requires FiniteElementGrid<Complex>&& IsMetric<Metric, typename Complex::point_type, typename Complex::scalar_type>
    Eigen::SparseMatrix<typename Complex::scalar_type>
        assemble_mass_matrix(const Complex& mesh, const Metric& metric)
    {
        using Scalar = typename Complex::point_type::Scalar;
        using Index = int;
        std::size_t n = mesh.num_vertices();
        std::vector<Eigen::Triplet<Scalar>> triplets;

        // Для каждого треугольника вычисляем его площадь через метрику и вклады в масс-матрицу
        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto tri = mesh.triangle_at(t);
            // Площадь треугольника через метрику (используем метод cell_volume)
            Scalar area = mesh.cell_volume(t, metric);
            if (area <= Scalar{ 0 }) continue; // защита от вырожденных

            Scalar diag_contrib = area / Scalar{ 6 };
            Scalar off_contrib = area / Scalar{ 12 };

            for (int i = 0; i < 3; ++i) {
                Index vi = static_cast<Index>(tri[i]);
                // диагональ
                triplets.emplace_back(vi, vi, diag_contrib);

                for (int j = i + 1; j < 3; ++j) {
                    Index vj = static_cast<Index>(tri[j]);
                    triplets.emplace_back(vi, vj, off_contrib);
                    triplets.emplace_back(vj, vi, off_contrib);
                }
            }
        }

        Eigen::SparseMatrix<Scalar> M(static_cast<Index>(n), static_cast<Index>(n));
        M.setFromTriplets(triplets.begin(), triplets.end());
        return M;
    }

    /**
     * @brief Собрать матрицу жёсткости (stiffness matrix) для линейных элементов.
     *
     * Для треугольников в 2D это котангенсный лапласиан. Используем готовую функцию
     * build_cotangent_laplacian из numerical/cotangent_laplacian.h.
     *
     * @tparam Complex Тип, удовлетворяющий FiniteElementGrid.
     * @tparam Metric  Тип метрики.
     * @param mesh Сетка.
     * @param metric Метрика для вычисления длин и площадей.
     * @return Разрежённая матрица жёсткости.
     */
    template<typename Complex, typename Metric>
        requires FiniteElementGrid<Complex>&& IsMetric<Metric, typename Complex::point_type, typename Complex::scalar_type>
    Eigen::SparseMatrix<typename Complex::scalar_type>
        assemble_stiffness_matrix(const Complex& mesh, const Metric& metric)
    {
        return build_cotangent_laplacian(mesh, metric);
    }

    /**
     * @brief Собрать диагональную (сгущённую) массовую матрицу.
     *
     * Для каждого узла сумма элементов в строке исходной масс-матрицы.
     *
     * @param mass_matrix Полная массовая матрица.
     * @return Диагональная матрица (вектор диагональных элементов).
     */
    template<typename Scalar>
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
        lumped_mass_matrix(const Eigen::SparseMatrix<Scalar>& mass_matrix) {
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> lumped(mass_matrix.rows());
        lumped.setZero();
        for (int k = 0; k < mass_matrix.outerSize(); ++k) {
            for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(mass_matrix, k); it; ++it) {
                lumped(it.row()) += it.value();
            }
        }
        return lumped;
    }

} // namespace delta::numerical