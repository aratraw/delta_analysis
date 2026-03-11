// include/delta/numerical/integrals.h
#pragma once

#include <functional>
#include <vector>

namespace delta::numerical {

    // 1D uniform grid (left Riemann sum)
    template<typename T, typename Func>
    T integral_1d_uniform(const std::vector<T>& x, const Func& f) {
        if (x.size() < 2) return T(0);
        T h = x[1] - x[0];
        T sum = T(0);
        for (const auto& xi : x) {
            sum += f(xi);
        }
        return sum * h;
    }

    // 1D trapezoidal rule
    template<typename T, typename Func>
    T integral_1d_trapezoidal(const std::vector<T>& x, const Func& f) {
        if (x.size() < 2) return T(0);
        T sum = T(0);
        for (std::size_t i = 0; i < x.size() - 1; ++i) {
            T h = x[i + 1] - x[i];
            sum += (f(x[i]) + f(x[i + 1])) * h / T(2);
        }
        return sum;
    }

    // 2D uniform grid (product grid, left Riemann sum)
    template<typename T, typename Func>
    T integral_2d_uniform(const std::vector<T>& x, const std::vector<T>& y, const Func& f) {
        if (x.size() < 2 || y.size() < 2) return T(0);
        T hx = x[1] - x[0];
        T hy = y[1] - y[0];
        T sum = T(0);
        for (const auto& xi : x) {
            for (const auto& yj : y) {
                sum += f(xi, yj);
            }
        }
        return sum * hx * hy;
    }

    // 2D non-uniform rectangular grid (left Riemann sum on each cell)
    template<typename T, typename Func>
    T integral_2d_left(const std::vector<T>& x, const std::vector<T>& y, const Func& f) {
        if (x.size() < 2 || y.size() < 2) return T(0);
        T sum = T(0);
        for (std::size_t i = 0; i < x.size() - 1; ++i) {
            T hx = x[i + 1] - x[i];
            for (std::size_t j = 0; j < y.size() - 1; ++j) {
                T hy = y[j + 1] - y[j];
                sum += f(x[i], y[j]) * hx * hy;
            }
        }
        return sum;
    }

    // Можно добавить другие квадратурные формулы по мере необходимости

} // namespace delta::numerical