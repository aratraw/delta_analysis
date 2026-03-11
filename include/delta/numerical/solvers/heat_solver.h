// include/delta/numerical/solvers/heat_solver.h
#pragma once

#include "delta/geometry/simplicial_complex.h"
#include "delta/numerical/discrete_operators.h"
#include <functional>
#include <vector>

namespace delta::numerical {

    /**
     * @brief Solve the heat equation ∂u/∂t = α Δu + f using explicit Euler.
     *
     * @tparam Complex  SimplicialComplex.
     * @tparam Metric   Metric.
     * @param mesh      The mesh.
     * @param u0        Initial condition at vertices.
     * @param f         Source term (function of time and vertex index, or constant).
     * @param alpha     Thermal diffusivity.
     * @param dt        Time step.
     * @param num_steps Number of steps.
     * @param metric    Metric.
     * @return std::vector<std::vector<Value>> Solution at each time step.
     */
    template<typename Complex, typename Metric>
    std::vector<std::vector<typename Complex::value_type>>
        solve_heat_explicit(const Complex& mesh,
            const std::vector<typename Complex::value_type>& u0,
            const std::function<typename Complex::value_type(double, std::size_t)>& f,
            typename Complex::value_type alpha,
            double dt,
            std::size_t num_steps,
            const Metric& metric) {
        using Value = typename Complex::value_type;
        std::size_t n = mesh.size();

        std::vector<Value> u = u0;
        std::vector<std::vector<Value>> result;
        result.push_back(u);

        for (std::size_t step = 0; step < num_steps; ++step) {
            double t = step * dt;
            auto Lu = discrete_laplacian_cotangent(mesh, u, metric);
            std::vector<Value> u_new(n);
            for (std::size_t i = 0; i < n; ++i) {
                Value fi = f(t, i);
                u_new[i] = u[i] + dt * (alpha * Lu[i] + fi);
            }
            u = u_new;
            result.push_back(u);
        }
        return result;
    }

} // namespace delta::numerical