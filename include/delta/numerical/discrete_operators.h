// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/numerical/discrete_operators.h
// ============================================================================
// DISCRETE OPERATORS: GRADIENT, DIVERGENCE, CURL, LAPLACIAN
// ============================================================================
//
// This file implements finite‑difference approximations of differential
// operators on arbitrary grids (uniform, list, product) using any metric.
//
// ----------------------------------------------------------------------------
// SUPPORTED GRIDS
// ----------------------------------------------------------------------------
//
// - UniformGrid<Scalar>: regular 1D grid with constant step.
// - ListGrid<Scalar>: arbitrary sorted 1D grid.
// - ProductGrid<Grid, N>: Cartesian product of N copies of a 1D grid.
//   The address type is `std::array<Scalar, N>`.
//
// ----------------------------------------------------------------------------
// OPERATORS
// ----------------------------------------------------------------------------
//
// - discrete_gradient(f)     : scalar field f → vector field ∇f
// - discrete_divergence(v)   : vector field v → scalar field ∇·v
// - discrete_curl_2d(v)      : 2D vector field → scalar field (∂v_y/∂x - ∂v_x/∂y)
// - discrete_curl_3d(v)      : 3D vector field → vector field ∇×v
// - discrete_laplacian(f)    : scalar field f → Δf (sum of second derivatives)
//
// ----------------------------------------------------------------------------
// DIFFERENCE SCHEMES
// ----------------------------------------------------------------------------
//
// - FORWARD  : (f(x+h) - f(x)) / h
// - BACKWARD : (f(x) - f(x-h)) / h
// - CENTRAL  : (f(x+h) - f(x-h)) / (2h)  (default, more accurate)
//
// At boundaries, central difference falls back to one‑sided (forward/backward)
// when the neighbour on one side is missing.
//
// ----------------------------------------------------------------------------
// PARALLELISATION
// ----------------------------------------------------------------------------
//
// OpenMP parallelisation is enabled when the number of points exceeds 1000.
// Each iteration reads field values at neighbouring points (read‑only) and
// writes to its own result slot. No data races occur.
//
// To disable OpenMP, compile with `-D_OPENMP=0` or remove the pragmas.
//
// ============================================================================

// ToDo: Generalize all the boilerplate code for computing derivatives, which will shrink the file in volume by roughly 30%
// Priority: Medium. Difficulty:Low

#ifndef DELTA_NUMERICAL_DISCRETE_OPERATORS_H
#define DELTA_NUMERICAL_DISCRETE_OPERATORS_H

#include <optional>
#include <cmath>
#include <type_traits>
#include <array>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include "delta/core/rational.h"
#include "delta/core/grid_concept.h"
#include "delta/core/regulative_idea.h"
#include "delta/core/uniform_grid.h"
#include "delta/core/list_grid.h"
#include "delta/geometry/tensor_field.h"
#include "delta/core/product_grid.h"   // for ProductGrid.

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
        FORWARD,    ///< (f(x+h) - f(x)) / h
        BACKWARD,   ///< (f(x) - f(x-h)) / h
        CENTRAL     ///< (f(x+h) - f(x-h)) / (2h)
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
    // 1D differences (kept as is for potential direct use)
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

        // 1. Collect all points
        std::vector<Addr> points = grid.collect_points();
        std::vector<Eigen::Matrix<Scalar, Dim, 1>> gradients(points.size());

        // 2. Compute gradients in parallel (or sequentially)
#ifdef _OPENMP
        static constexpr std::size_t OMP_MIN_SIZE = 1000;
        if (points.size() >= OMP_MIN_SIZE) {
#pragma omp parallel for
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(points.size()); ++i) {
                const Addr& point = points[i];
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
                gradients[i] = std::move(g);
            }
        }
        else
#endif
        {
            for (std::size_t i = 0; i < points.size(); ++i) {
                const Addr& point = points[i];
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
                gradients[i] = std::move(g);
            }
        }

        // 3. Fill the TensorField
        delta::geometry::TensorField<Addr, Scalar, 1, Dim, std::less<Addr>> grad;
        for (std::size_t i = 0; i < points.size(); ++i) {
            grad.set(points[i], gradients[i]);
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

        std::vector<Addr> points = grid.collect_points();
        std::vector<Scalar> divergences(points.size());

#ifdef _OPENMP
        static constexpr std::size_t OMP_MIN_SIZE = 1000;
        if (points.size() >= OMP_MIN_SIZE) {
#pragma omp parallel for
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(points.size()); ++i) {
                const Addr& point = points[i];
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
                divergences[i] = d;
            }
        }
        else
#endif
        {
            for (std::size_t i = 0; i < points.size(); ++i) {
                const Addr& point = points[i];
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
                divergences[i] = d;
            }
        }

        delta::geometry::TensorField<Addr, Scalar, 0, Dim, std::less<Addr>> div;
        for (std::size_t i = 0; i < points.size(); ++i) {
            div.set(points[i], divergences[i]);
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

        std::vector<Addr> points = grid.collect_points();
        std::vector<Scalar> curls(points.size());

#ifdef _OPENMP
        static constexpr std::size_t OMP_MIN_SIZE = 1000;
        if (points.size() >= OMP_MIN_SIZE) {
#pragma omp parallel for
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(points.size()); ++i) {
                const Addr& point = points[i];
                Scalar c = 0;
                // x-derivative of v_y
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
                // y-derivative of v_x (subtracted)
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
                curls[i] = c;
            }
        }
        else
#endif
        {
            for (std::size_t i = 0; i < points.size(); ++i) {
                const Addr& point = points[i];
                Scalar c = 0;
                // x-derivative of v_y
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
                // y-derivative of v_x (subtracted)
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
                curls[i] = c;
            }
        }

        delta::geometry::TensorField<Addr, Scalar, 0, 2, std::less<Addr>> curl;
        for (std::size_t i = 0; i < points.size(); ++i) {
            curl.set(points[i], curls[i]);
        }
        return curl;
    }

    // -------------------------------------------------------------------------
    // Curl of a vector field in 3D (returns a vector field)
    // -------------------------------------------------------------------------
    template<typename Grid, typename VecField, typename Metric>
    auto discrete_curl_3d(const Grid& grid, const VecField& v, const Metric& metric,
        DifferenceScheme scheme = DifferenceScheme::CENTRAL) {
        static_assert(address_dimension<typename Grid::value_type>::value == 3,
            "curl_3d requires 3D addresses");
        using Addr = typename Grid::value_type;
        using Scalar = typename VecField::value_type::Scalar;
        constexpr int Dim = 3;

        std::vector<Addr> points = grid.collect_points();
        std::vector<Eigen::Matrix<Scalar, 3, 1>> curls(points.size());

#ifdef _OPENMP
        static constexpr std::size_t OMP_MIN_SIZE = 1000;
        if (points.size() >= OMP_MIN_SIZE) {
#pragma omp parallel for
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(points.size()); ++i) {
                const Addr& point = points[i];
                Eigen::Matrix<Scalar, 3, 1> c;
                // curl_x = dvz_dy - dvy_dz
                {
                    auto next_y = neighbor(grid, point, 1, +1);
                    auto prev_y = neighbor(grid, point, 1, -1);
                    auto next_z = neighbor(grid, point, 2, +1);
                    auto prev_z = neighbor(grid, point, 2, -1);
                    Scalar dvz_dy = 0, dvy_dz = 0;
                    // Compute dvz_dy
                    if (next_y && prev_y) {
                        Scalar hy = metric(*prev_y, *next_y);
                        dvz_dy = (v.at(*next_y)[2] - v.at(*prev_y)[2]) / hy;
                    }
                    else if (next_y) {
                        Scalar hy = metric(point, *next_y);
                        dvz_dy = (v.at(*next_y)[2] - v.at(point)[2]) / hy;
                    }
                    else if (prev_y) {
                        Scalar hy = metric(*prev_y, point);
                        dvz_dy = (v.at(point)[2] - v.at(*prev_y)[2]) / hy;
                    }
                    else {
                        throw std::out_of_range("curl: no neighbours in y direction");
                    }
                    // Compute dvy_dz
                    if (next_z && prev_z) {
                        Scalar hz = metric(*prev_z, *next_z);
                        dvy_dz = (v.at(*next_z)[1] - v.at(*prev_z)[1]) / hz;
                    }
                    else if (next_z) {
                        Scalar hz = metric(point, *next_z);
                        dvy_dz = (v.at(*next_z)[1] - v.at(point)[1]) / hz;
                    }
                    else if (prev_z) {
                        Scalar hz = metric(*prev_z, point);
                        dvy_dz = (v.at(point)[1] - v.at(*prev_z)[1]) / hz;
                    }
                    else {
                        throw std::out_of_range("curl: no neighbours in z direction");
                    }
                    c[0] = dvz_dy - dvy_dz;
                }
                // curl_y = dvx_dz - dvz_dx
                {
                    auto next_z = neighbor(grid, point, 2, +1);
                    auto prev_z = neighbor(grid, point, 2, -1);
                    auto next_x = neighbor(grid, point, 0, +1);
                    auto prev_x = neighbor(grid, point, 0, -1);
                    Scalar dvx_dz = 0, dvz_dx = 0;
                    // dvx_dz
                    if (next_z && prev_z) {
                        Scalar hz = metric(*prev_z, *next_z);
                        dvx_dz = (v.at(*next_z)[0] - v.at(*prev_z)[0]) / hz;
                    }
                    else if (next_z) {
                        Scalar hz = metric(point, *next_z);
                        dvx_dz = (v.at(*next_z)[0] - v.at(point)[0]) / hz;
                    }
                    else if (prev_z) {
                        Scalar hz = metric(*prev_z, point);
                        dvx_dz = (v.at(point)[0] - v.at(*prev_z)[0]) / hz;
                    }
                    else {
                        throw std::out_of_range("curl: no neighbours in z direction");
                    }
                    // dvz_dx
                    if (next_x && prev_x) {
                        Scalar hx = metric(*prev_x, *next_x);
                        dvz_dx = (v.at(*next_x)[2] - v.at(*prev_x)[2]) / hx;
                    }
                    else if (next_x) {
                        Scalar hx = metric(point, *next_x);
                        dvz_dx = (v.at(*next_x)[2] - v.at(point)[2]) / hx;
                    }
                    else if (prev_x) {
                        Scalar hx = metric(*prev_x, point);
                        dvz_dx = (v.at(point)[2] - v.at(*prev_x)[2]) / hx;
                    }
                    else {
                        throw std::out_of_range("curl: no neighbours in x direction");
                    }
                    c[1] = dvx_dz - dvz_dx;
                }
                // curl_z = dvy_dx - dvx_dy
                {
                    auto next_x = neighbor(grid, point, 0, +1);
                    auto prev_x = neighbor(grid, point, 0, -1);
                    auto next_y = neighbor(grid, point, 1, +1);
                    auto prev_y = neighbor(grid, point, 1, -1);
                    Scalar dvy_dx = 0, dvx_dy = 0;
                    // dvy_dx
                    if (next_x && prev_x) {
                        Scalar hx = metric(*prev_x, *next_x);
                        dvy_dx = (v.at(*next_x)[1] - v.at(*prev_x)[1]) / hx;
                    }
                    else if (next_x) {
                        Scalar hx = metric(point, *next_x);
                        dvy_dx = (v.at(*next_x)[1] - v.at(point)[1]) / hx;
                    }
                    else if (prev_x) {
                        Scalar hx = metric(*prev_x, point);
                        dvy_dx = (v.at(point)[1] - v.at(*prev_x)[1]) / hx;
                    }
                    else {
                        throw std::out_of_range("curl: no neighbours in x direction");
                    }
                    // dvx_dy
                    if (next_y && prev_y) {
                        Scalar hy = metric(*prev_y, *next_y);
                        dvx_dy = (v.at(*next_y)[0] - v.at(*prev_y)[0]) / hy;
                    }
                    else if (next_y) {
                        Scalar hy = metric(point, *next_y);
                        dvx_dy = (v.at(*next_y)[0] - v.at(point)[0]) / hy;
                    }
                    else if (prev_y) {
                        Scalar hy = metric(*prev_y, point);
                        dvx_dy = (v.at(point)[0] - v.at(*prev_y)[0]) / hy;
                    }
                    else {
                        throw std::out_of_range("curl: no neighbours in y direction");
                    }
                    c[2] = dvy_dx - dvx_dy;
                }
                curls[i] = std::move(c);
            }
        }
        else
#endif
        {
            for (std::size_t i = 0; i < points.size(); ++i) {
                const Addr& point = points[i];
                Eigen::Matrix<Scalar, 3, 1> c;
                // curl_x = dvz_dy - dvy_dz
                {
                    auto next_y = neighbor(grid, point, 1, +1);
                    auto prev_y = neighbor(grid, point, 1, -1);
                    auto next_z = neighbor(grid, point, 2, +1);
                    auto prev_z = neighbor(grid, point, 2, -1);
                    Scalar dvz_dy = 0, dvy_dz = 0;
                    // Compute dvz_dy
                    if (next_y && prev_y) {
                        Scalar hy = metric(*prev_y, *next_y);
                        dvz_dy = (v.at(*next_y)[2] - v.at(*prev_y)[2]) / hy;
                    }
                    else if (next_y) {
                        Scalar hy = metric(point, *next_y);
                        dvz_dy = (v.at(*next_y)[2] - v.at(point)[2]) / hy;
                    }
                    else if (prev_y) {
                        Scalar hy = metric(*prev_y, point);
                        dvz_dy = (v.at(point)[2] - v.at(*prev_y)[2]) / hy;
                    }
                    else {
                        throw std::out_of_range("curl: no neighbours in y direction");
                    }
                    // Compute dvy_dz
                    if (next_z && prev_z) {
                        Scalar hz = metric(*prev_z, *next_z);
                        dvy_dz = (v.at(*next_z)[1] - v.at(*prev_z)[1]) / hz;
                    }
                    else if (next_z) {
                        Scalar hz = metric(point, *next_z);
                        dvy_dz = (v.at(*next_z)[1] - v.at(point)[1]) / hz;
                    }
                    else if (prev_z) {
                        Scalar hz = metric(*prev_z, point);
                        dvy_dz = (v.at(point)[1] - v.at(*prev_z)[1]) / hz;
                    }
                    else {
                        throw std::out_of_range("curl: no neighbours in z direction");
                    }
                    c[0] = dvz_dy - dvy_dz;
                }
                // curl_y = dvx_dz - dvz_dx
                {
                    auto next_z = neighbor(grid, point, 2, +1);
                    auto prev_z = neighbor(grid, point, 2, -1);
                    auto next_x = neighbor(grid, point, 0, +1);
                    auto prev_x = neighbor(grid, point, 0, -1);
                    Scalar dvx_dz = 0, dvz_dx = 0;
                    // dvx_dz
                    if (next_z && prev_z) {
                        Scalar hz = metric(*prev_z, *next_z);
                        dvx_dz = (v.at(*next_z)[0] - v.at(*prev_z)[0]) / hz;
                    }
                    else if (next_z) {
                        Scalar hz = metric(point, *next_z);
                        dvx_dz = (v.at(*next_z)[0] - v.at(point)[0]) / hz;
                    }
                    else if (prev_z) {
                        Scalar hz = metric(*prev_z, point);
                        dvx_dz = (v.at(point)[0] - v.at(*prev_z)[0]) / hz;
                    }
                    else {
                        throw std::out_of_range("curl: no neighbours in z direction");
                    }
                    // dvz_dx
                    if (next_x && prev_x) {
                        Scalar hx = metric(*prev_x, *next_x);
                        dvz_dx = (v.at(*next_x)[2] - v.at(*prev_x)[2]) / hx;
                    }
                    else if (next_x) {
                        Scalar hx = metric(point, *next_x);
                        dvz_dx = (v.at(*next_x)[2] - v.at(point)[2]) / hx;
                    }
                    else if (prev_x) {
                        Scalar hx = metric(*prev_x, point);
                        dvz_dx = (v.at(point)[2] - v.at(*prev_x)[2]) / hx;
                    }
                    else {
                        throw std::out_of_range("curl: no neighbours in x direction");
                    }
                    c[1] = dvx_dz - dvz_dx;
                }
                // curl_z = dvy_dx - dvx_dy
                {
                    auto next_x = neighbor(grid, point, 0, +1);
                    auto prev_x = neighbor(grid, point, 0, -1);
                    auto next_y = neighbor(grid, point, 1, +1);
                    auto prev_y = neighbor(grid, point, 1, -1);
                    Scalar dvy_dx = 0, dvx_dy = 0;
                    // dvy_dx
                    if (next_x && prev_x) {
                        Scalar hx = metric(*prev_x, *next_x);
                        dvy_dx = (v.at(*next_x)[1] - v.at(*prev_x)[1]) / hx;
                    }
                    else if (next_x) {
                        Scalar hx = metric(point, *next_x);
                        dvy_dx = (v.at(*next_x)[1] - v.at(point)[1]) / hx;
                    }
                    else if (prev_x) {
                        Scalar hx = metric(*prev_x, point);
                        dvy_dx = (v.at(point)[1] - v.at(*prev_x)[1]) / hx;
                    }
                    else {
                        throw std::out_of_range("curl: no neighbours in x direction");
                    }
                    // dvx_dy
                    if (next_y && prev_y) {
                        Scalar hy = metric(*prev_y, *next_y);
                        dvx_dy = (v.at(*next_y)[0] - v.at(*prev_y)[0]) / hy;
                    }
                    else if (next_y) {
                        Scalar hy = metric(point, *next_y);
                        dvx_dy = (v.at(*next_y)[0] - v.at(point)[0]) / hy;
                    }
                    else if (prev_y) {
                        Scalar hy = metric(*prev_y, point);
                        dvx_dy = (v.at(point)[0] - v.at(*prev_y)[0]) / hy;
                    }
                    else {
                        throw std::out_of_range("curl: no neighbours in y direction");
                    }
                    c[2] = dvy_dx - dvx_dy;
                }
                curls[i] = std::move(c);
            }
        }

        delta::geometry::TensorField<Addr, Scalar, 1, 3, std::less<Addr>> curl;
        for (std::size_t i = 0; i < points.size(); ++i) {
            curl.set(points[i], curls[i]);
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

        std::vector<Addr> points = grid.collect_points();
        std::vector<Scalar> laplacians(points.size());

#ifdef _OPENMP
        static constexpr std::size_t OMP_MIN_SIZE = 1000;
        if (points.size() >= OMP_MIN_SIZE) {
#pragma omp parallel for
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(points.size()); ++i) {
                const Addr& point = points[i];
                Scalar L = 0;
                for (int d = 0; d < Dim; ++d) {
                    auto next = neighbor(grid, point, d, +1);
                    auto prev = neighbor(grid, point, d, -1);
                    if (next && prev) {
                        // interior point: use formula with two steps
                        Scalar h_plus = metric(point, *next);
                        Scalar h_minus = metric(*prev, point);
                        Scalar right_diff = (f.at(*next) - f.at(point)) / h_plus;
                        Scalar left_diff = (f.at(point) - f.at(*prev)) / h_minus;
                        L += (Scalar(2) / (h_plus + h_minus)) * (right_diff - left_diff);
                    }
                    else if (next) {
                        // right boundary: one-sided second derivative
                        Scalar h = metric(point, *next);
                        L += (f.at(*next) - f.at(point)) / (h * h);
                    }
                    else if (prev) {
                        // left boundary: one-sided second derivative
                        Scalar h = metric(*prev, point);
                        L += (f.at(point) - f.at(*prev)) / (h * h);
                    }
                    // isolated point: skip
                }
                laplacians[i] = L;
            }
        }
        else
#endif
        {
            for (std::size_t i = 0; i < points.size(); ++i) {
                const Addr& point = points[i];
                Scalar L = 0;
                for (int d = 0; d < Dim; ++d) {
                    auto next = neighbor(grid, point, d, +1);
                    auto prev = neighbor(grid, point, d, -1);
                    if (next && prev) {
                        Scalar h_plus = metric(point, *next);
                        Scalar h_minus = metric(*prev, point);
                        Scalar right_diff = (f.at(*next) - f.at(point)) / h_plus;
                        Scalar left_diff = (f.at(point) - f.at(*prev)) / h_minus;
                        L += (Scalar(2) / (h_plus + h_minus)) * (right_diff - left_diff);
                    }
                    else if (next) {
                        Scalar h = metric(point, *next);
                        L += (f.at(*next) - f.at(point)) / (h * h);
                    }
                    else if (prev) {
                        Scalar h = metric(*prev, point);
                        L += (f.at(point) - f.at(*prev)) / (h * h);
                    }
                }
                laplacians[i] = L;
            }
        }

        delta::geometry::TensorField<Addr, Scalar, 0, Dim, std::less<Addr>> lap;
        for (std::size_t i = 0; i < points.size(); ++i) {
            lap.set(points[i], laplacians[i]);
        }
        return lap;
    }

} // namespace delta::numerical

#endif // DELTA_NUMERICAL_DISCRETE_OPERATORS_H