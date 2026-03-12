// include/delta/geometry/hat_basis.h
#pragma once

#include "simplicial_complex.h"
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <stdexcept>

namespace delta::geometry {

    /**
     * @brief Кусочно-линейные базисные функции (hat functions) на симплициальном комплексе.
     *
     * Для каждой вершины v определена функция φ_v, равная 1 в этой вершине,
     * 0 во всех остальных вершинах, и линейная на каждом симплексе.
     *
     * @tparam Complex Тип симплициального комплекса (например, SimplicialComplex<Dim, Coord>).
     */
    template<typename Complex>
    class HatBasis {
    public:
        using point_type = typename Complex::point_type;
        using scalar_type = typename point_type::Scalar;
        using vertex_index = typename Complex::vertex_index;

        /**
         * @param mesh Ссылка на комплекс (время жизни должно превышать время жизни объекта HatBasis).
         */
        explicit HatBasis(const Complex& mesh) : mesh_(mesh) {
            // Предвычисляем градиенты для каждого симплекса
            precompute_gradients();
        }

        /**
         * @brief Значение базисной функции φ_v в точке p.
         * @param v Индекс вершины.
         * @param p Точка.
         * @return φ_v(p)
         */
        scalar_type evaluate(vertex_index v, const point_type& p) const {
            // Если p совпадает с вершиной – быстрое возвращение
            for (std::size_t i = 0; i < mesh_.num_vertices(); ++i) {
                if (p == mesh_.vertex(i)) {
                    return (i == v) ? scalar_type{ 1 } : scalar_type{ 0 };
                }
            }

            // Ищем симплекс, содержащий p
            auto simplex_info = locate_point(p);
            if (!simplex_info.has_value()) {
                return scalar_type{ 0 }; // точка вне комплекса
            }

            const auto& [dim, idx, bary_coords] = *simplex_info;
            // Барицентрические координаты относительно вершин симплекса
            const auto& simp = mesh_.get_simplex(dim, idx);
            // Находим позицию вершины v в этом симплексе
            int pos = -1;
            for (std::size_t i = 0; i < simp.size(); ++i) {
                if (simp[i] == v) {
                    pos = static_cast<int>(i);
                    break;
                }
            }
            if (pos == -1) return scalar_type{ 0 }; // вершина не в этом симплексе
            return bary_coords[pos];
        }

        /**
         * @brief Градиент базисной функции φ_v в точке p.
         * @param v Индекс вершины.
         * @param p Точка.
         * @return ∇φ_v(p) (постоянен на симплексе)
         */
        point_type gradient(vertex_index v, const point_type& p) const {
            // Ищем симплекс, содержащий p
            auto simplex_info = locate_point(p);
            if (!simplex_info.has_value()) {
                return point_type::Zero();
            }
            const auto& [dim, idx, bary_coords] = *simplex_info;
            // Градиент предвычислен для каждого симплекса и каждой вершины
            auto it = gradients_.find({ dim, idx });
            if (it == gradients_.end()) {
                return point_type::Zero();
            }
            const auto& grad_map = it->second;
            auto git = grad_map.find(v);
            if (git == grad_map.end()) {
                return point_type::Zero();
            }
            return git->second;
        }

    private:
        const Complex& mesh_;

        // Структура для хранения информации о найденном симплексе
        struct LocatedSimplex {
            int dim;
            std::size_t idx;
            std::vector<scalar_type> barycentric; // барицентрические координаты
        };

        // Кэш для градиентов: для каждого симплекса (dim, idx) хранится отображение вершина -> градиент
        mutable std::unordered_map<
            std::pair<int, std::size_t>,
            std::unordered_map<vertex_index, point_type>,
            boost::hash<std::pair<int, std::size_t>>
        > gradients_;

        // Предвычисление градиентов для всех симплексов
        void precompute_gradients() {
            // Определяем максимальную размерность
            int max_dim = 0;
            for (const auto& kv : mesh_.simplices_) {
                if (kv.first > max_dim) max_dim = kv.first;
            }
            for (int dim = 1; dim <= max_dim; ++dim) {
                std::size_t n = mesh_.num_simplices(dim);
                for (std::size_t idx = 0; idx < n; ++idx) {
                    const auto& simp = mesh_.get_simplex(dim, idx);
                    // Вычисляем градиенты для каждой вершины симплекса
                    auto grads = compute_gradients_on_simplex(simp, dim);
                    gradients_[{dim, idx}] = std::move(grads);
                }
            }
        }

        // Вычисление градиентов для всех вершин данного симплекса
        std::unordered_map<vertex_index, point_type>
            compute_gradients_on_simplex(const std::vector<vertex_index>& simp, int dim) const {
            std::unordered_map<vertex_index, point_type> result;
            if (dim == 0) return result;

            // Используем формулу для барицентрических координат на симплексе
            // Градиент λ_i = (кофактор) / (k! * объём)
            // Для линейного симплекса в евклидовом пространстве градиент можно вычислить через
            // метод наименьших квадратов или через обратную матрицу якобиана.
            // Упростим: для размерностей 1,2,3 используем известные формулы, для остальных – заглушка.
            if (dim == 1) {
                // Отрезок: grad λ_0 = -1/длина, grad λ_1 = 1/длина (направление от v0 к v1)
                point_type v0 = mesh_.vertex(simp[0]);
                point_type v1 = mesh_.vertex(simp[1]);
                point_type dir = v1 - v0;
                scalar_type len = dir.norm();
                if (len > 0) {
                    result[simp[0]] = -dir / len;
                    result[simp[1]] = dir / len;
                }
            }
            else if (dim == 2) {
                // Треугольник: используем формулу через поворот на 90°
                point_type v0 = mesh_.vertex(simp[0]);
                point_type v1 = mesh_.vertex(simp[1]);
                point_type v2 = mesh_.vertex(simp[2]);
                point_type e1 = v1 - v0;
                point_type e2 = v2 - v0;
                scalar_type area2 = std::abs(e1.x() * e2.y() - e1.y() * e2.x()); // удвоенная площадь
                if (area2 > 0) {
                    // Кофакторы (повёрнутые векторы)
                    point_type n0(e2.y(), -e2.x()); // перпендикуляр к e2
                    point_type n1(-e1.y(), e1.x()); // перпендикуляр к -e1 (для λ1)
                    point_type n2(e1.y() - e2.y(), e2.x() - e1.x()); // для λ2
                    // Градиенты = n_i / area2
                    result[simp[0]] = n0 / area2;
                    result[simp[1]] = n1 / area2;
                    result[simp[2]] = n2 / area2;
                }
            }
            else if (dim == 3) {
                // Тетраэдр: используем формулу через векторные произведения граней
                point_type v0 = mesh_.vertex(simp[0]);
                point_type v1 = mesh_.vertex(simp[1]);
                point_type v2 = mesh_.vertex(simp[2]);
                point_type v3 = mesh_.vertex(simp[3]);
                point_type e1 = v1 - v0;
                point_type e2 = v2 - v0;
                point_type e3 = v3 - v0;
                scalar_type vol6 = std::abs(e1.dot(e2.cross(e3))); // 6 * объём
                if (vol6 > 0) {
                    // Векторы площадей противоположных граней (направленные внутрь)
                    point_type A0 = (v2 - v1).cross(v3 - v1); // грань v1v2v3
                    point_type A1 = (v3 - v0).cross(v2 - v0); // грань v0v2v3
                    point_type A2 = (v1 - v0).cross(v3 - v0); // грань v0v1v3
                    point_type A3 = (v2 - v0).cross(v1 - v0); // грань v0v1v2
                    // Градиенты = A_i / (6V) с учётом знака (ориентация)
                    result[simp[0]] = A0 / vol6;
                    result[simp[1]] = A1 / vol6;
                    result[simp[2]] = A2 / vol6;
                    result[simp[3]] = A3 / vol6;
                }
            }
            else {
                // Для dim > 3 оставляем нулевые градиенты (можно добавить общую формулу через псевдообратную)
                for (auto vi : simp) {
                    result[vi] = point_type::Zero();
                }
            }
            return result;
        }

        // Поиск симплекса, содержащего точку p (простой перебор)
        std::optional<LocatedSimplex> locate_point(const point_type& p) const {
            // Перебираем все симплексы, начиная с максимальной размерности
            int max_dim = 0;
            for (const auto& kv : mesh_.simplices_) {
                if (kv.first > max_dim) max_dim = kv.first;
            }
            for (int dim = max_dim; dim >= 1; --dim) {
                std::size_t n = mesh_.num_simplices(dim);
                for (std::size_t idx = 0; idx < n; ++idx) {
                    const auto& simp = mesh_.get_simplex(dim, idx);
                    // Проверяем, принадлежит ли p симплексу через барицентрические координаты
                    auto bary = compute_barycentric(p, simp, dim);
                    if (bary.has_value()) {
                        return LocatedSimplex{ dim, idx, std::move(*bary) };
                    }
                }
            }
            return std::nullopt;
        }

        // Вычисление барицентрических координат точки p относительно симплекса (список индексов вершин)
        std::optional<std::vector<scalar_type>>
            compute_barycentric(const point_type& p, const std::vector<vertex_index>& simp, int dim) const {
            // Для размерностей 1,2,3 используем известные формулы, для остальных – заглушка
            std::vector<scalar_type> bary(simp.size(), scalar_type{ 0 });

            if (dim == 1) {
                // Отрезок: параметр t = (p - v0).dot(e) / |e|^2
                point_type v0 = mesh_.vertex(simp[0]);
                point_type v1 = mesh_.vertex(simp[1]);
                point_type e = v1 - v0;
                scalar_type len2 = e.squaredNorm();
                if (len2 == 0) return std::nullopt;
                scalar_type t = (p - v0).dot(e) / len2;
                if (t >= -1e-12 && t <= 1.0 + 1e-12) {
                    bary[0] = scalar_type{ 1 } - t;
                    bary[1] = t;
                    return bary;
                }
            }
            else if (dim == 2) {
                // Треугольник: через площади
                point_type v0 = mesh_.vertex(simp[0]);
                point_type v1 = mesh_.vertex(simp[1]);
                point_type v2 = mesh_.vertex(simp[2]);
                scalar_type area = std::abs((v1 - v0).cross(v2 - v0).norm()) / 2;
                if (area == 0) return std::nullopt;
                scalar_type a0 = std::abs((v1 - p).cross(v2 - p).norm()) / 2;
                scalar_type a1 = std::abs((v2 - p).cross(v0 - p).norm()) / 2;
                scalar_type a2 = std::abs((v0 - p).cross(v1 - p).norm()) / 2;
                // Проверка, что сумма площадей равна area (с допуском)
                if (std::abs(a0 + a1 + a2 - area) > 1e-12 * area) return std::nullopt;
                bary[0] = a0 / area;
                bary[1] = a1 / area;
                bary[2] = a2 / area;
                return bary;
            }
            else if (dim == 3) {
                // Тетраэдр: через объёмы
                point_type v0 = mesh_.vertex(simp[0]);
                point_type v1 = mesh_.vertex(simp[1]);
                point_type v2 = mesh_.vertex(simp[2]);
                point_type v3 = mesh_.vertex(simp[3]);
                scalar_type vol6 = std::abs((v1 - v0).dot((v2 - v0).cross(v3 - v0)));
                if (vol6 == 0) return std::nullopt;
                scalar_type v0_vol6 = std::abs((v1 - p).dot((v2 - p).cross(v3 - p)));
                scalar_type v1_vol6 = std::abs((v0 - p).dot((v2 - p).cross(v3 - p)));
                scalar_type v2_vol6 = std::abs((v0 - p).dot((v1 - p).cross(v3 - p)));
                scalar_type v3_vol6 = std::abs((v0 - p).dot((v1 - p).cross(v2 - p)));
                if (std::abs(v0_vol6 + v1_vol6 + v2_vol6 + v3_vol6 - vol6) > 1e-12 * vol6) return std::nullopt;
                bary[0] = v0_vol6 / vol6;
                bary[1] = v1_vol6 / vol6;
                bary[2] = v2_vol6 / vol6;
                bary[3] = v3_vol6 / vol6;
                return bary;
            }
            return std::nullopt;
        }
    };
    // Метод для проверки принадлежности точки симплексу и получения барицентрических координат
    template<typename OtherComplex>
    std::optional<std::vector<scalar_type>>
        locate_point_in_simplex(const point_type& p,
            const OtherComplex& mesh,
            int dim,
            std::size_t idx) const {
        const auto& simp = mesh.get_simplex(dim, idx);
        return compute_barycentric(p, simp, dim);
    }

} // namespace delta::geometry