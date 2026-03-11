#pragma once

#include "delta/geometry/simplicial_complex.h"
#include "delta/numerical/cotangent_laplacian.h"
#include <Eigen/Sparse>
#include <vector>

namespace delta::numerical {

    /**
     * @brief Собрать массовую матрицу для линейных элементов на симплициальном комплексе.
     *
     * Для треугольников в 2D используется формула:
     *   M_ii = area / 6
     *   M_ij = area / 12   для i ≠ j (соседние вершины в треугольнике)
     *
     * @tparam Complex Тип, удовлетворяющий SimplicialComplex (точки должны иметь оператор сложения и умножения на скаляр).
     * @param mesh Сетка.
     * @return Разрежённая матрица размера num_vertices() x num_vertices().
     */
    template<typename Complex>
    Eigen::SparseMatrix<typename Complex::point_type::Scalar>
        assemble_mass_matrix(const Complex& mesh) {
        using Scalar = typename Complex::point_type::Scalar;
        using Index = int;
        std::size_t n = mesh.num_vertices();
        std::vector<Eigen::Triplet<Scalar>> triplets;

        // Для каждого треугольника вычисляем его площадь и вклады в масс-матрицу
        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto tri = mesh.triangle_at(t);
            auto p0 = mesh.vertex(tri[0]);
            auto p1 = mesh.vertex(tri[1]);
            auto p2 = mesh.vertex(tri[2]);

            // Площадь треугольника (для евклидовой метрики)
            auto ab = p1 - p0;
            auto ac = p2 - p0;
            Scalar area = std::abs(ab.x() * ac.y() - ab.y() * ac.x()) / Scalar{ 2 };

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
     * Для треугольников в 2D это cotangent Laplacian. Используем готовую функцию
     * build_cotangent_laplacian из numerical/cotangent_laplacian.h.
     *
     * @tparam Complex Тип, удовлетворяющий SimplicialComplex (точки должны иметь координаты типа double или Rational).
     * @param mesh Сетка.
     * @return Разрежённая матрица жёсткости.
     */
    template<typename Complex>
    Eigen::SparseMatrix<typename Complex::point_type::Scalar>
        assemble_stiffness_matrix(const Complex& mesh) {
        return build_cotangent_laplacian(mesh);
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