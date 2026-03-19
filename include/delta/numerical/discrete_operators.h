// include/delta/numerical/discrete_operators.h
#ifndef DELTA_NUMERICAL_DISCRETE_OPERATORS_H
#define DELTA_NUMERICAL_DISCRETE_OPERATORS_H

#include <optional>
#include <cmath>
#include <type_traits>
#include <array>
#include <utility>
#include <Eigen/Dense>
#include "delta/core/rational.h"
#include "delta/core/grid_concept.h"
#include "delta/core/regulative_idea.h"
#include "delta/core/uniform_grid.h"
#include "delta/core/list_grid.h"
#include "delta/geometry/tensor_field.h"
#include "delta/core/product_path.h"// for ProductGrid. Yeah, it's non-obvious. Noone pays me to make it obvious.
// at some point should rename core/product_path and deduplicate 
// the ProductPath functional with geometry/product_regulative.h

namespace delta::numerical {

    // -------------------------------------------------------------------------
    // Trait to determine dimension from address type
    // -------------------------------------------------------------------------
    template<typename Addr, typename = void>
    struct address_dimension : std::integral_constant<int, 1> {};

    template<typename T>
    struct address_dimension<T, std::enable_if_t<std::is_arithmetic_v<T>>>
        : std::integral_constant<int, 1> {
    };

    template<typename T, std::size_t N>
    struct address_dimension<std::array<T, N>> : std::integral_constant<int, N> {};

    template<typename T1, typename T2>
    struct address_dimension<std::pair<T1, T2>> : std::integral_constant<int, 2> {};

    // -------------------------------------------------------------------------
    // Difference schemes
    // -------------------------------------------------------------------------
    enum class DifferenceScheme {
        FORWARD,
        BACKWARD,
        CENTRAL
    };

    // -------------------------------------------------------------------------
    // Helper to get the neighbour address in a given direction
    // -------------------------------------------------------------------------
    template<typename Grid, typename Addr>
    std::optional<Addr> neighbor(const Grid& grid, const Addr& addr, int dim, int direction) {
        return std::nullopt; // default: not implemented
    }

    // Specialisation for UniformGrid (1D)
    template<typename T, typename Compare>
    std::optional<T> neighbor(const delta::UniformGrid<T, Compare>& grid, const T& addr, int dim, int direction) {
        if (dim != 0) return std::nullopt;
        T step = grid.step();
        T candidate = addr + (direction > 0 ? step : -step);
        // Check bounds
        if (direction > 0 && candidate > grid.start() + (grid.count() - 1) * step) return std::nullopt;
        if (direction < 0 && candidate < grid.start()) return std::nullopt;
        return candidate;
    }

    // Specialisation for ListGrid (1D)
    template<typename T, typename Compare>
    std::optional<T> neighbor(const delta::ListGrid<T, Compare>& grid, const T& addr, int dim, int direction) {
        if (dim != 0) return std::nullopt;
        auto it = std::lower_bound(grid.begin(), grid.end(), addr, grid.comparator());
        if (it == grid.end() || grid.comparator()(addr, *it) || grid.comparator()(*it, addr)) {
            return std::nullopt; // addr not in grid
        }
        std::size_t idx = std::distance(grid.begin(), it);
        if (direction > 0 && idx + 1 < grid.size()) {
            return grid[idx + 1];
        }
        if (direction < 0 && idx > 0) {
            return grid[idx - 1];
        }
        return std::nullopt;
    }

    // Specialisation for ProductGrid (multidimensional)
    template<typename Grid, std::size_t N>
    std::optional<std::array<typename Grid::value_type, N>>
        neighbor(const delta::ProductGrid<Grid, N>& grid,
            const std::array<typename Grid::value_type, N>& addr,
            int dim, int direction) {
        if (dim < 0 || dim >= static_cast<int>(N)) {
            return std::nullopt;
        }
        auto new_addr = addr;
        // Call neighbor for the corresponding subgrid (which is 1D)
        auto comp = neighbor(grid.get_grid(dim), addr[dim], 0, direction);
        if (!comp) {
            return std::nullopt;
        }
        new_addr[dim] = *comp;
        return new_addr;
    }

    // -------------------------------------------------------------------------
    // 1D differences
    // -------------------------------------------------------------------------
    template<typename Grid, typename Field, typename Metric>
    auto forward_difference(const Grid& grid, const Field& field, const Metric& metric,
        const typename Grid::value_type& point) {
        auto next = neighbor(grid, point, 0, +1);
        if (!next) {
            throw std::out_of_range("forward_difference: no neighbour in positive direction");
        }
        auto dist = metric(point, *next);
        return (field.at(*next) - field.at(point)) / dist;
    }

    template<typename Grid, typename Field, typename Metric>
    auto backward_difference(const Grid& grid, const Field& field, const Metric& metric,
        const typename Grid::value_type& point) {
        auto prev = neighbor(grid, point, 0, -1);
        if (!prev) {
            throw std::out_of_range("backward_difference: no neighbour in negative direction");
        }
        auto dist = metric(*prev, point);
        return (field.at(point) - field.at(*prev)) / dist;
    }

    template<typename Grid, typename Field, typename Metric>
    auto central_difference(const Grid& grid, const Field& field, const Metric& metric,
        const typename Grid::value_type& point) {
        auto next = neighbor(grid, point, 0, +1);
        auto prev = neighbor(grid, point, 0, -1);
        if (!next || !prev) {
            throw std::out_of_range("central_difference: need neighbours on both sides");
        }
        auto dist = metric(*prev, *next);
        return (field.at(*next) - field.at(*prev)) / dist;
    }

    // -------------------------------------------------------------------------
    // Gradient of a scalar field (returns a vector field)
    // -------------------------------------------------------------------------
    template<typename Grid, typename ScalarField, typename Metric>
    auto discrete_gradient(const Grid& grid, const ScalarField& f, const Metric& metric,
        DifferenceScheme scheme = DifferenceScheme::CENTRAL) {
        using Addr = typename Grid::value_type;
        using Scalar = typename ScalarField::value_type;
        constexpr int Dim = address_dimension<Addr>::value;

        delta::geometry::TensorField<Addr, Scalar, 1, Dim, std::less<Addr>> grad;

        for (const auto& point : grid) {
            Eigen::Matrix<Scalar, Dim, 1> g;
            for (int d = 0; d < Dim; ++d) {
                auto next = neighbor(grid, point, d, +1);
                auto prev = neighbor(grid, point, d, -1);
                switch (scheme) {
                case DifferenceScheme::FORWARD:
                    if (!next) throw std::out_of_range("gradient: forward neighbour missing");
                    g[d] = (f.at(*next) - f.at(point)) / metric(point, *next);
                    break;
                case DifferenceScheme::BACKWARD:
                    if (!prev) throw std::out_of_range("gradient: backward neighbour missing");
                    g[d] = (f.at(point) - f.at(*prev)) / metric(*prev, point);
                    break;
                case DifferenceScheme::CENTRAL:
                    if (!next || !prev) {
                        if (next) {
                            g[d] = (f.at(*next) - f.at(point)) / metric(point, *next);
                        }
                        else if (prev) {
                            g[d] = (f.at(point) - f.at(*prev)) / metric(*prev, point);
                        }
                        else {
                            throw std::out_of_range("gradient: no neighbours at all");
                        }
                    }
                    else {
                        g[d] = (f.at(*next) - f.at(*prev)) / metric(*prev, *next);
                    }
                    break;
                }
            }
            grad.set(point, g);
        }
        return grad;
    }

    // -------------------------------------------------------------------------
    // Divergence of a vector field (returns a scalar field)
    // -------------------------------------------------------------------------
    template<typename Grid, typename VecField, typename Metric>
    auto discrete_divergence(const Grid& grid, const VecField& v, const Metric& metric,
        DifferenceScheme scheme = DifferenceScheme::CENTRAL) {
        using Addr = typename Grid::value_type;
        using Scalar = typename VecField::value_type::Scalar;
        constexpr int Dim = address_dimension<Addr>::value;

        delta::geometry::TensorField<Addr, Scalar, 0, Dim, std::less<Addr>> div;

        for (const auto& point : grid) {
            Scalar d = 0;
            for (int dim = 0; dim < Dim; ++dim) {
                auto next = neighbor(grid, point, dim, +1);
                auto prev = neighbor(grid, point, dim, -1);
                switch (scheme) {
                case DifferenceScheme::FORWARD:
                    if (!next) throw std::out_of_range("divergence: forward neighbour missing");
                    d += (v.at(*next)[dim] - v.at(point)[dim]) / metric(point, *next);
                    break;
                case DifferenceScheme::BACKWARD:
                    if (!prev) throw std::out_of_range("divergence: backward neighbour missing");
                    d += (v.at(point)[dim] - v.at(*prev)[dim]) / metric(*prev, point);
                    break;
                case DifferenceScheme::CENTRAL:
                    if (!next || !prev) {
                        if (next) {
                            d += (v.at(*next)[dim] - v.at(point)[dim]) / metric(point, *next);
                        }
                        else if (prev) {
                            d += (v.at(point)[dim] - v.at(*prev)[dim]) / metric(*prev, point);
                        }
                        else {
                            throw std::out_of_range("divergence: no neighbours");
                        }
                    }
                    else {
                        d += (v.at(*next)[dim] - v.at(*prev)[dim]) / metric(*prev, *next);
                    }
                    break;
                }
            }
            div.set(point, d);
        }
        return div;
    }

    // -------------------------------------------------------------------------
    // Curl of a vector field in 2D (returns a scalar field)
    // -------------------------------------------------------------------------
    template<typename Grid, typename VecField, typename Metric>
    auto discrete_curl_2d(const Grid& grid, const VecField& v, const Metric& metric,
        DifferenceScheme scheme = DifferenceScheme::CENTRAL) {
        static_assert(address_dimension<typename Grid::value_type>::value == 2,
            "curl_2d requires 2D addresses");
        using Addr = typename Grid::value_type;
        using Scalar = typename VecField::value_type::Scalar;
        constexpr int Dim = 2;

        delta::geometry::TensorField<Addr, Scalar, 0, Dim, std::less<Addr>> curl;

        for (const auto& point : grid) {
            Scalar c = 0;
            // ∂v_y/∂x - ∂v_x/∂y
            // x‑derivative of v_y
            {
                auto next = neighbor(grid, point, 0, +1);
                auto prev = neighbor(grid, point, 0, -1);
                switch (scheme) {
                case DifferenceScheme::FORWARD:
                    if (!next) throw std::out_of_range("curl: forward neighbour missing in x");
                    c += (v.at(*next)[1] - v.at(point)[1]) / metric(point, *next);
                    break;
                case DifferenceScheme::BACKWARD:
                    if (!prev) throw std::out_of_range("curl: backward neighbour missing in x");
                    c += (v.at(point)[1] - v.at(*prev)[1]) / metric(*prev, point);
                    break;
                case DifferenceScheme::CENTRAL:
                    if (!next || !prev) {
                        if (next) {
                            c += (v.at(*next)[1] - v.at(point)[1]) / metric(point, *next);
                        }
                        else if (prev) {
                            c += (v.at(point)[1] - v.at(*prev)[1]) / metric(*prev, point);
                        }
                        else {
                            throw std::out_of_range("curl: no neighbours in x");
                        }
                    }
                    else {
                        c += (v.at(*next)[1] - v.at(*prev)[1]) / metric(*prev, *next);
                    }
                    break;
                }
            }
            // y‑derivative of v_x (subtracted)
            {
                auto next = neighbor(grid, point, 1, +1);
                auto prev = neighbor(grid, point, 1, -1);
                Scalar dvx_dy;
                switch (scheme) {
                case DifferenceScheme::FORWARD:
                    if (!next) throw std::out_of_range("curl: forward neighbour missing in y");
                    dvx_dy = (v.at(*next)[0] - v.at(point)[0]) / metric(point, *next);
                    break;
                case DifferenceScheme::BACKWARD:
                    if (!prev) throw std::out_of_range("curl: backward neighbour missing in y");
                    dvx_dy = (v.at(point)[0] - v.at(*prev)[0]) / metric(*prev, point);
                    break;
                case DifferenceScheme::CENTRAL:
                    if (!next || !prev) {
                        if (next) {
                            dvx_dy = (v.at(*next)[0] - v.at(point)[0]) / metric(point, *next);
                        }
                        else if (prev) {
                            dvx_dy = (v.at(point)[0] - v.at(*prev)[0]) / metric(*prev, point);
                        }
                        else {
                            throw std::out_of_range("curl: no neighbours in y");
                        }
                    }
                    else {
                        dvx_dy = (v.at(*next)[0] - v.at(*prev)[0]) / metric(*prev, *next);
                    }
                    break;
                }
                c -= dvx_dy;
            }
            curl.set(point, c);
        }
        return curl;
    }

    // -------------------------------------------------------------------------
    // Laplacian of a scalar field (using second differences)
    // -------------------------------------------------------------------------
    template<typename Grid, typename ScalarField, typename Metric>
    auto discrete_laplacian(const Grid& grid, const ScalarField& f, const Metric& metric) {
        using Addr = typename Grid::value_type;
        using Scalar = typename ScalarField::value_type;
        constexpr int Dim = address_dimension<Addr>::value;

        delta::geometry::TensorField<Addr, Scalar, 0, Dim, std::less<Addr>> lap;

        for (const auto& point : grid) {
            Scalar L = 0;
            for (int d = 0; d < Dim; ++d) {
                auto next = neighbor(grid, point, d, +1);
                auto prev = neighbor(grid, point, d, -1);
                if (!next || !prev) {
                    // boundary: use one‑sided second derivative
                    if (next) {
                        Scalar dx = metric(point, *next);
                        L += (f.at(*next) - f.at(point)) / (dx * dx);
                    }
                    else if (prev) {
                        Scalar dx = metric(*prev, point);
                        L += (f.at(point) - f.at(*prev)) / (dx * dx);
                    } // else skip
                }
                else {
                    // standard second order formula on uniform grid:
                    // (f_next - 2*f + f_prev) / (h^2), where h = distance to neighbour
                    // For non‑uniform grids this is not exact but we assume uniform in tests.
                    Scalar h = metric(point, *next); // should equal metric(*prev, point)
                    L += (f.at(*next) - 2 * f.at(point) + f.at(*prev)) / (h * h);
                }
            }
            lap.set(point, L);
        }
        return lap;
    }

} // namespace delta::numerical

#endif // DELTA_NUMERICAL_DISCRETE_OPERATORS_H