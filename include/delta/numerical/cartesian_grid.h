// include/delta/numerical/cartesian_grid.h
#pragma once

#include "delta/core/uniform_grid.h"

namespace delta::numerical {

    // Концепт трёхмерной декартовой сетки
    template<typename G>
    concept CartesianGrid3D = requires(G g) {
        typename G::value_type;
        requires std::is_same_v<typename G::value_type, Eigen::Matrix<typename G::scalar_type, 3, 1>>;
        { g.x_grid() } -> std::same_as<const UniformGrid<typename G::scalar_type>&>;
        { g.y_grid() } -> std::same_as<const UniformGrid<typename G::scalar_type>&>;
        { g.z_grid() } -> std::same_as<const UniformGrid<typename G::scalar_type>&>;
        { g.size() } -> std::convertible_to<std::size_t>;
        { g.point_at(std::size_t{}, std::size_t{}, std::size_t{}) } -> std::convertible_to<typename G::value_type>;
    };

    // Простая реализация для равномерной сетки
    template<typename Scalar>
    class UniformCartesianGrid3D {
    public:
        using value_type = Eigen::Matrix<Scalar, 3, 1>;
        using scalar_type = Scalar;

        UniformCartesianGrid3D(UniformGrid<Scalar> x, UniformGrid<Scalar> y, UniformGrid<Scalar> z)
            : x_(std::move(x)), y_(std::move(y)), z_(std::move(z)) {
        }

        const UniformGrid<Scalar>& x_grid() const { return x_; }
        const UniformGrid<Scalar>& y_grid() const { return y_; }
        const UniformGrid<Scalar>& z_grid() const { return z_; }

        std::size_t size() const { return x_.size() * y_.size() * z_.size(); }

        value_type point_at(std::size_t i, std::size_t j, std::size_t k) const {
            return value_type(x_[i], y_[j], z_[k]);
        }

        // Преобразование линейного индекса в тройку
        std::tuple<std::size_t, std::size_t, std::size_t> unindex(std::size_t idx) const {
            std::size_t nx = x_.size();
            std::size_t ny = y_.size();
            std::size_t nz = z_.size();
            std::size_t i = idx / (ny * nz);
            std::size_t j = (idx % (ny * nz)) / nz;
            std::size_t k = idx % nz;
            return { i, j, k };
        }

    private:
        UniformGrid<Scalar> x_, y_, z_;
    };

} // namespace delta::numerical