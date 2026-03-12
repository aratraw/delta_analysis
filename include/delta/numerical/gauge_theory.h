// include/delta/numerical/gauge_theory.h
#pragma once

#include "delta/geometry/simplicial_complex.h"
#include "gauge_groups.h"
#include <vector>
#include <random>
#include <cmath>

namespace delta::numerical {

    // -----------------------------------------------------------------------------
    // GaugeField: калибровочное поле на симплициальном комплексе
    // -----------------------------------------------------------------------------
    template<typename Group, typename Complex>
    class GaugeField {
    public:
        using group_type = Group;
        using complex_type = Complex;
        using value_type = typename Group::value_type;
        using matrix_type = typename Group::matrix_type;
        using edge_index = std::size_t;

        explicit GaugeField(const Complex& mesh)
            : mesh_(mesh), links_(mesh.num_edges(), Group::identity()) {
        }

        // Доступ к линку по индексу ребра (ориентация от меньшего индекса к большему)
        Group& link(std::size_t e) { return links_.at(e); }
        const Group& link(std::size_t e) const { return links_.at(e); }

        // Установка линка для ориентированного ребра (from, to)
        void set_link(typename Complex::vertex_index from,
            typename Complex::vertex_index to,
            const Group& g) {
            std::ptrdiff_t eidx = mesh_.find_simplex(1, { from, to });
            if (eidx == -1)
                throw std::invalid_argument("Edge not found");
            links_[static_cast<std::size_t>(eidx)] = g;
        }

        // Получение линка для ориентированного ребра (если ориентация обратная, возвращаем обратный элемент)
        Group get_link(typename Complex::vertex_index from,
            typename Complex::vertex_index to) const {
            std::ptrdiff_t eidx = mesh_.find_simplex(1, { from, to });
            if (eidx != -1) {
                return links_[static_cast<std::size_t>(eidx)];
            }
            // Попробуем обратное ребро
            eidx = mesh_.find_simplex(1, { to, from });
            if (eidx != -1) {
                return links_[static_cast<std::size_t>(eidx)].inverse();
            }
            throw std::invalid_argument("Edge not found");
        }

        // Количество линков
        std::size_t size() const { return links_.size(); }

        // -------------------------------------------------------------------------
        // Действие Вильсона
        // S = β * Σ_{plaquettes} (1 - (1/N) Re Tr U_plaquette)
        // -------------------------------------------------------------------------
        double wilson_action(double beta) const {
            double sum = 0.0;
            const double invN = 1.0 / Group::dimension;
            for (std::size_t p = 0; p < mesh_.num_simplices(2); ++p) {
                auto tri = mesh_.triangle_at(p);
                // Ориентация треугольника: (v0,v1,v2) – предполагаем согласованную
                auto g01 = get_link(tri[0], tri[1]);
                auto g12 = get_link(tri[1], tri[2]);
                auto g20 = get_link(tri[2], tri[0]);
                Group plaquette = g01 * g12 * g20;
                sum += 1.0 - invN * plaquette.real_tr();
            }
            return beta * sum;
        }

        // -------------------------------------------------------------------------
        // Вариация действия по линку (сила) для U(1)
        // Для U(1): dS/dU_e = β Σ_{plaquettes containing e} sin(θ_plaquette) * (ориентация)
        // -------------------------------------------------------------------------
        template<typename = std::enable_if_t<std::is_same_v<Group, U1>>>
        double variation(std::size_t edge_idx) const {
            double sum = 0.0;
            // Находим все треугольники, содержащие данное ребро
            // Для этого нужна структура инцидентности в комплексе.
            // Пока реализуем через линейный поиск (для тестов).
            // В реальности следует предвычислить списки.
            const auto& mesh = mesh_;
            for (std::size_t p = 0; p < mesh.num_simplices(2); ++p) {
                auto tri = mesh.triangle_at(p);
                // Проверяем, есть ли наше ребро в треугольнике
                int orient = 0; // +1 если ребро совпадает по ориентации, -1 если обратное
                if ((tri[0] == edge.first && tri[1] == edge.second)) orient = 1;
                else if ((tri[1] == edge.first && tri[2] == edge.second)) orient = 1;
                else if ((tri[2] == edge.first && tri[0] == edge.second)) orient = 1;
                else if ((tri[0] == edge.second && tri[1] == edge.first)) orient = -1;
                else if ((tri[1] == edge.second && tri[2] == edge.first)) orient = -1;
                else if ((tri[2] == edge.second && tri[0] == edge.first)) orient = -1;
                if (orient == 0) continue;

                // Вычисляем произведение вокруг треугольника
                auto g01 = get_link(tri[0], tri[1]);
                auto g12 = get_link(tri[1], tri[2]);
                auto g20 = get_link(tri[2], tri[0]);
                U1 plaquette = g01 * g12 * g20;
                // Для U(1) произведение даёт фазу e^{iθ}
                double theta = plaquette.log(); // мнимая часть логарифма
                sum += orient * std::sin(theta);
            }
            return sum;
        }

        // Для SU(2) вариация сложнее – пока не реализуем
        template<typename = std::enable_if_t<std::is_same_v<Group, SU2>>>
        Eigen::Matrix2cd variation(std::size_t edge_idx) const {
            // Заглушка – вернём нулевую матрицу
            return Eigen::Matrix2cd::Zero();
        }

        // -------------------------------------------------------------------------
        // Рандомизация (для тестов)
        // -------------------------------------------------------------------------
        void randomize(double amplitude) {
            static std::mt19937 rng{ std::random_device{}() };
            if constexpr (std::is_same_v<Group, U1>) {
                std::uniform_real_distribution<double> dist(-amplitude, amplitude);
                for (auto& g : links_) {
                    g = U1(dist(rng));
                }
            }
            else if constexpr (std::is_same_v<Group, SU2>) {
                std::uniform_real_distribution<double> dist(-amplitude, amplitude);
                for (auto& g : links_) {
                    Eigen::Vector3d axis(dist(rng), dist(rng), dist(rng));
                    double theta = dist(rng);
                    g = SU2::exp(theta, axis);
                }
            }
            else if constexpr (std::is_same_v<Group, SU3>) {
                // Для SU(3) генерируем случайную антиэрмитову матрицу
                std::normal_distribution<double> ndist;
                for (auto& g : links_) {
                    Eigen::Matrix3cd alg = Eigen::Matrix3cd::Zero();
                    for (int i = 0; i < 3; ++i) {
                        for (int j = i + 1; j < 3; ++j) {
                            std::complex<double> z(ndist(rng), ndist(rng));
                            alg(i, j) = z;
                            alg(j, i) = -std::conj(z);
                        }
                    }
                    // делаем бесследовой
                    alg(0, 0) = std::complex<double>(ndist(rng), 0);
                    alg(1, 1) = std::complex<double>(ndist(rng), 0);
                    alg(2, 2) = -alg(0, 0) - alg(1, 1);
                    g = SU3::exp(alg * amplitude);
                }
            }
        }

    private:
        const Complex& mesh_;
        std::vector<Group> links_;
    };

    // -----------------------------------------------------------------------------
    // Преобразование калибровочного поля в связность (Connection)
    // -----------------------------------------------------------------------------
    template<typename Addr, typename Scalar, int Dim, typename Complex>
    Connection<Addr, Scalar, Dim>
        gauge_field_to_connection(const GaugeField<typename Connection<Addr, Scalar, Dim>::matrix_type, Complex>& gf) {
        Connection<Addr, Scalar, Dim> conn;
        // Предполагаем, что адреса – вершины, и матрицы из gf – это групповые элементы.
        // Для связности нужны матрицы, действующие на векторы. Если группа матричная, можно использовать их напрямую.
        for (std::size_t e = 0; e < gf.size(); ++e) {
            // Нужно получить вершины ребра из комплекса. Здесь Complex должен предоставлять метод edge_at.
            auto [v0, v1] = gf.mesh().edge_at(e);
            conn.set_transport(v0, v1, gf.link(e).matrix());
        }
        return conn;
    }

    // -----------------------------------------------------------------------------
    // Преобразование связности в калибровочное поле (логарифм, если нужна алгебра)
    // -----------------------------------------------------------------------------
    template<typename Addr, typename Scalar, int Dim, typename Complex>
    GaugeField<typename Connection<Addr, Scalar, Dim>::matrix_type, Complex>
        connection_to_gauge_field(const Connection<Addr, Scalar, Dim>& conn, const Complex& mesh) {
        using Group = typename Connection<Addr, Scalar, Dim>::matrix_type; // предполагаем, что матрица – групповой элемент
        GaugeField<Group, Complex> gf(mesh);
        for (const auto& [edge, mat] : conn) {
            // edge – пара (from,to)
            gf.set_link(edge.first, edge.second, Group(mat)); // просто копируем матрицу
        }
        return gf;
    }

} // namespace delta::numerical