// include/delta/numerical/finite_differences.h
#pragma once

#include <functional>
#include <cmath>

namespace delta::numerical {

    /**
     * @brief Forward difference: (f(x+h) - f(x)) / h
     */
    template<typename T, typename Func>
    T forward_diff(const Func& f, const T& x, const T& h) {
        return (f(x + h) - f(x)) / h;
    }

    /**
     * @brief Backward difference: (f(x) - f(x-h)) / h
     */
    template<typename T, typename Func>
    T backward_diff(const Func& f, const T& x, const T& h) {
        return (f(x) - f(x - h)) / h;
    }

    /**
     * @brief Central difference: (f(x+h) - f(x-h)) / (2h)
     */
    template<typename T, typename Func>
    T central_diff(const Func& f, const T& x, const T& h) {
        return (f(x + h) - f(x - h)) / (T(2) * h);
    }

} // namespace delta::numerical