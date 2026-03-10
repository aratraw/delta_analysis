// include/delta/numerical/finite_differences.h
#pragma once

#include <cstddef>
#include <functional>

namespace delta::numerical {

    /**
     * @brief Forward difference: (f(x+h) - f(x)) / h
     */
    template<typename Func>
    double forward_diff(const Func& f, double x, double h) {
        return (f(x + h) - f(x)) / h;
    }

    /**
     * @brief Backward difference: (f(x) - f(x-h)) / h
     */
    template<typename Func>
    double backward_diff(const Func& f, double x, double h) {
        return (f(x) - f(x - h)) / h;
    }

    /**
     * @brief Central difference: (f(x+h) - f(x-h)) / (2h)
     */
    template<typename Func>
    double central_diff(const Func& f, double x, double h) {
        return (f(x + h) - f(x - h)) / (2.0 * h);
    }

} // namespace delta::numerical