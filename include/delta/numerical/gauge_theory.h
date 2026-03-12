// include/delta/numerical/gauge_theory.h
#pragma once

#include "delta/geometry/simplicial_complex.h"
#include "gauge_groups.h"
#include <vector>
#include <random>
#include <cmath>
#include <complex>
#include <optional>

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
        using vertex_index = typename Complex::vertex_index;

        explicit GaugeField(const Complex& mesh)
            : mesh_(mesh), links_(mesh.num_edges(), Group::identity()) {
        }

        // Доступ к линку по индексу ребра (ориентация от меньшего индекса к большему)
        Group& link(std::size_t e) { return links_.at(e); }
        const Group& link(std::size_t e) const { return links_.at(e); }

        // Установка линка для ориентированного ребра (from, to)
        void set_link(vertex_index from, vertex_index to, const Group& g) {
            std::ptrdiff_t eidx = mesh_.find_simplex(1, { from, to });
            if (eidx == -1)
                throw std::invalid_argument("Edge not found");
            links_[static_cast<std::size_t>(eidx)] = g;
        }

        // Получение линка для ориентированного ребра (если ориентация обратная, возвращаем обратный элемент)
        Group get_link(vertex_index from, vertex_index to) const {
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
                auto g01 = get_link(tri[0], tri[1]);
                auto g12 = get_link(tri[1], tri[2]);
                auto g20 = get_link(tri[2], tri[0]);
                Group plaquette = g01 * g12 * g20;
                sum += 1.0 - invN * plaquette.real_tr();
            }
            return beta * sum;
        }

        // -------------------------------------------------------------------------
        // Напряжённость поля как элемент алгебры Ли на треугольнике
        // Для унитарных групп: F = (U - U†)/2  (в дискретном виде, без деления на площадь)
        // -------------------------------------------------------------------------
        matrix_type field_strength_algebra(std::size_t face_idx) const {
            auto tri = mesh_.triangle_at(face_idx);
            auto g01 = get_link(tri[0], tri[1]);
            auto g12 = get_link(tri[1], tri[2]);
            auto g20 = get_link(tri[2], tri[0]);
            Group plaquette = g01 * g12 * g20;
            // Возвращаем (U - U†)/2
            return (plaquette.matrix() - plaquette.matrix().adjoint()) / typename Group::value_type(2);
        }

        // -------------------------------------------------------------------------
        // Калибровочное преобразование: U_{xy} -> g_x * U_{xy} * g_y^{-1}
        // gauge_factors[i] – групповой элемент в вершине с индексом i
        // -------------------------------------------------------------------------
        void gauge_transform(const std::vector<Group>& gauge_factors) {
            if (gauge_factors.size() != mesh_.num_vertices()) {
                throw std::invalid_argument("gauge_transform: number of gauge factors must equal number of vertices");
            }
            for (std::size_t e = 0; e < mesh_.num_edges(); ++e) {
                auto [v0, v1] = mesh_.edge_at(e);
                Group g0 = gauge_factors[v0];
                Group g1 = gauge_factors[v1];
                links_[e] = g0 * links_[e] * g1.inverse();
            }
        }

        // -------------------------------------------------------------------------
        // Вариация действия Вильсона по линку (сила) для группы U(1)
        // -------------------------------------------------------------------------
        template<typename = std::enable_if_t<std::is_same_v<Group, U1>>>
        double variation(std::size_t edge_idx, double beta = 1.0) const {
            double sum = 0.0;
            auto [v0, v1] = mesh_.edge_at(edge_idx);
            auto neighbors = mesh_.edge_neighbors(edge_idx);
            std::size_t left = neighbors.first;
            std::optional<std::size_t> right = neighbors.second;

            auto contribute = [&](std::size_t tri_idx, int orient) {
                auto tri = mesh_.triangle_at(tri_idx);
                Group plaquette;
                if (orient > 0) {
                    plaquette = get_link(tri[0], tri[1]) * get_link(tri[1], tri[2]) * get_link(tri[2], tri[0]);
                }
                else {
                    // Если ребро в обратной ориентации, берём обратную петлю
                    plaquette = get_link(tri[2], tri[1]) * get_link(tri[1], tri[0]) * get_link(tri[0], tri[2]);
                }
                double theta = plaquette.log(); // для U(1) log возвращает мнимую часть (фазу)
                sum += orient * std::sin(theta);
                };

            contribute(left, +1);
            if (right.has_value()) {
                contribute(*right, -1);
            }
            return beta * sum;
        }

        // -------------------------------------------------------------------------
        // Вариация для SU(2) (аналитическая формула)
        // -------------------------------------------------------------------------
        template<typename = std::enable_if_t<std::is_same_v<Group, SU2>>>
        Eigen::Matrix2cd variation(std::size_t edge_idx, double beta = 1.0) const {
            using Matrix = Eigen::Matrix2cd;
            Matrix result = Matrix::Zero();
            auto [v0, v1] = mesh_.edge_at(edge_idx);
            auto neighbors = mesh_.edge_neighbors(edge_idx);
            std::size_t left = neighbors.first;
            std::optional<std::size_t> right = neighbors.second;

            auto contribute = [&](std::size_t tri_idx, int orient) {
                auto tri = mesh_.triangle_at(tri_idx);
                // Найдём позицию ребра (v0,v1) в треугольнике
                int pos = -1;
                for (int i = 0; i < 3; ++i) {
                    if (tri[i] == v0 && tri[(i + 1) % 3] == v1) { pos = i; break; }
                }
                if (pos == -1) return; // не должно случаться

                // Произведение двух других рёбер
                Group other;
                if (pos == 0) {
                    other = get_link(tri[1], tri[2]) * get_link(tri[2], tri[0]);
                }
                else if (pos == 1) {
                    other = get_link(tri[2], tri[0]) * get_link(tri[0], tri[1]);
                }
                else { // pos == 2
                    other = get_link(tri[0], tri[1]) * get_link(tri[1], tri[2]);
                }
                Group plaquette = other * get_link(v0, v1); // полная петля
                Matrix diff = (plaquette.matrix() - plaquette.matrix().adjoint()) / 2.0;
                result += orient * other.matrix().adjoint() * diff;
                };

            contribute(left, +1);
            if (right.has_value()) {
                contribute(*right, -1);
            }
            return -beta / 2.0 * result;
        }

        // -------------------------------------------------------------------------
        // Вариация для SU(3) (аналитическая формула)
        // -------------------------------------------------------------------------
        template<typename = std::enable_if_t<std::is_same_v<Group, SU3>>>
        Eigen::Matrix3cd variation(std::size_t edge_idx, double beta = 1.0) const {
            using Matrix = Eigen::Matrix3cd;
            Matrix result = Matrix::Zero();
            auto [v0, v1] = mesh_.edge_at(edge_idx);
            auto neighbors = mesh_.edge_neighbors(edge_idx);
            std::size_t left = neighbors.first;
            std::optional<std::size_t> right = neighbors.second;

            auto contribute = [&](std::size_t tri_idx, int orient) {
                auto tri = mesh_.triangle_at(tri_idx);
                // Найдём позицию ребра (v0,v1) в треугольнике
                int pos = -1;
                for (int i = 0; i < 3; ++i) {
                    if (tri[i] == v0 && tri[(i + 1) % 3] == v1) { pos = i; break; }
                }
                if (pos == -1) return; // не должно случаться

                // Произведение двух других рёбер
                Group other;
                if (pos == 0) {
                    other = get_link(tri[1], tri[2]) * get_link(tri[2], tri[0]);
                }
                else if (pos == 1) {
                    other = get_link(tri[2], tri[0]) * get_link(tri[0], tri[1]);
                }
                else { // pos == 2
                    other = get_link(tri[0], tri[1]) * get_link(tri[1], tri[2]);
                }
                Group plaquette = other * get_link(v0, v1); // полная петля
                Matrix diff = (plaquette.matrix() - plaquette.matrix().adjoint()) / 2.0;
                // Проекция на бесследовую часть (алгебра su(3))
                diff -= diff.trace() / 3.0 * Matrix::Identity();
                result += orient * other.matrix().adjoint() * diff;
                };

            contribute(left, +1);
            if (right.has_value()) {
                contribute(*right, -1);
            }
            return -beta / 2.0 * result;
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

        // Доступ к сетке (может пригодиться)
        const Complex& mesh() const { return mesh_; }

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
        for (std::size_t e = 0; e < gf.size(); ++e) {
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
        using Group = typename Connection<Addr, Scalar, Dim>::matrix_type;
        GaugeField<Group, Complex> gf(mesh);
        for (const auto& [edge, mat] : conn) {
            gf.set_link(edge.first, edge.second, Group(mat));
        }
        return gf;
    }

} // namespace delta::numerical