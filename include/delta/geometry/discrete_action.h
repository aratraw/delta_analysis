// include/delta/geometry/discrete_action.h
#pragma once

#include "delta/core/path_concept.h"
#include "delta/core/operational_function.h"
#include <functional>
#include <cstddef>
#include <vector>

namespace delta::geometry {

    // -------------------------------------------------------------------------
    // Helper: find index of an address in a grid
    // -------------------------------------------------------------------------
    template<typename Grid, typename Addr>
    std::ptrdiff_t find_address_index(const Grid& grid, const Addr& addr) {
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (grid[i] == addr) return static_cast<std::ptrdiff_t>(i);
        }
        return -1;
    }

    // -------------------------------------------------------------------------
    // Base class for discrete actions
    // -------------------------------------------------------------------------

    /**
     * @class DiscreteAction
     * @brief Abstract base class for discrete actions in Δ‑analysis.
     *
     * An action is a functional S[φ] defined on a field φ over a grid.
     * It provides evaluation of the action and its variation with respect
     * to a single degree of freedom.
     *
     * @tparam Field Type of the field (must provide value_type, address_type,
     *               and operator()(const address_type&) const).
     * @tparam Path  Type of the path (must satisfy the Path concept and
     *               provide a metric() method returning the address metric).
     */
    template<typename Field, typename Path>
    class DiscreteAction {
    public:
        using value_type = typename Field::value_type;
        using address_type = typename Field::address_type;

        virtual ~DiscreteAction() = default;

        /**
         * @brief Evaluate the total action S[φ].
         * @param field The field.
         * @param path  The path providing the current grid and metric.
         * @return The value of the action.
         */
        virtual value_type evaluate(const Field& field, const Path& path) const = 0;

        /**
         * @brief Compute the variation δS/δφ(addr).
         * @param field The field.
         * @param path  The path.
         * @param addr  The address (vertex, edge, etc.) at which to vary.
         * @return The derivative of the action with respect to the field value at addr.
         */
        virtual value_type variation(const Field& field, const Path& path,
            const address_type& addr) const = 0;
    };

    // -------------------------------------------------------------------------
    // Free particle action (1D trajectory)
    // -------------------------------------------------------------------------

    /**
     * @class FreeParticleAction
     * @brief Discrete action for a free particle: S = ∫ (m/2) ẋ² dt.
     *
     * The trajectory is represented by a scalar field on a 1D grid (time axis).
     * The metric of the path is assumed to give the time step between points.
     */
    template<typename Field, typename Path>
    class FreeParticleAction : public DiscreteAction<Field, Path> {
    public:
        using value_type = typename Field::value_type;
        using address_type = typename Field::address_type;

        /**
         * @param mass Particle mass (default 1).
         */
        explicit FreeParticleAction(value_type mass = value_type{ 1 }) : mass_(mass) {}

        value_type evaluate(const Field& field, const Path& path) const override {
            const auto& grid = path.current_grid();
            if (grid.size() < 2) return value_type{ 0 };
            value_type sum{ 0 };
            for (std::size_t i = 0; i + 1 < grid.size(); ++i) {
                const auto& x0 = grid[i];
                const auto& x1 = grid[i + 1];
                value_type dt = path.metric()(x1, x0);          // time step
                value_type v = (field(x1) - field(x0)) / dt;
                sum += mass_ * v * v * dt / value_type{ 2 };
            }
            return sum;
        }

        value_type variation(const Field& field, const Path& path,
            const address_type& addr) const override {
            const auto& grid = path.current_grid();
            std::ptrdiff_t idx = find_address_index(grid, addr);
            // Boundary points (Dirichlet) have zero variation
            if (idx <= 0 || idx >= static_cast<std::ptrdiff_t>(grid.size()) - 1)
                return value_type{ 0 };

            const auto& x_prev = grid[idx - 1];
            const auto& x_next = grid[idx + 1];
            value_type dt_prev = path.metric()(addr, x_prev);
            value_type dt_next = path.metric()(x_next, addr);
            value_type v_prev = (field(addr) - field(x_prev)) / dt_prev;
            value_type v_next = (field(x_next) - field(addr)) / dt_next;
            return mass_ * (v_prev - v_next);
        }

    private:
        value_type mass_;
    };

    // -------------------------------------------------------------------------
    // Scalar field action (1D) with potential
    // -------------------------------------------------------------------------

    /**
     * @class ScalarFieldAction
     * @brief Discrete action for a scalar field: S = ∫ [½ (∂φ)² + V(φ)] dx.
     *
     * The field is defined on a 1D grid. The action uses a left‑point Riemann
     * sum for the potential term.
     *
     * @note The derivative of the potential V'(φ) is currently not implemented;
     *       the variation() method returns zero. To use this class in actual
     *       computations, override variation() in a derived class or provide
     *       a mechanism for V'(φ).
     */
    template<typename Field, typename Path>
    class ScalarFieldAction : public DiscreteAction<Field, Path> {
    public:
        using value_type = typename Field::value_type;
        using address_type = typename Field::address_type;

        /// Type of the potential function: V(φ) → value
        using Potential = std::function<value_type(const value_type&)>;

        /**
         * @param V         Potential function.
         * @param stiffness Coefficient in front of the gradient term (default 1).
         */
        ScalarFieldAction(Potential V, value_type stiffness = value_type{ 1 })
            : V_(V), stiffness_(stiffness) {
        }

        value_type evaluate(const Field& field, const Path& path) const override {
            const auto& grid = path.current_grid();
            if (grid.size() < 2) return value_type{ 0 };
            value_type sum{ 0 };
            for (std::size_t i = 0; i + 1 < grid.size(); ++i) {
                const auto& x0 = grid[i];
                const auto& x1 = grid[i + 1];
                value_type h = path.metric()(x1, x0);          // grid spacing
                value_type grad = (field(x1) - field(x0)) / h;
                sum += (stiffness_ * grad * grad / value_type{ 2 } + V_(field(x0))) * h;
            }
            return sum;
        }

        value_type variation(const Field& field, const Path& path,
            const address_type& addr) const override {
            // Currently returns zero because V'(φ) is not available.
            // Override in a derived class to include the potential derivative.
            return value_type{ 0 };
        }

    private:
        Potential V_;
        value_type stiffness_;
    };

} // namespace delta::geometry