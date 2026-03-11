// include/delta/numerical/gradient.h
#pragma once

#include <functional>
#include <Eigen/Dense>
#include "finite_differences.h"

namespace delta::numerical {

    // 2D central
    template<typename T, typename Func>
    Eigen::Matrix<T, 2, 1> gradient_2d_central(const Func& f, const T& x, const T& y, const T& hx, const T& hy) {
        T dfdx = central_diff<T>([&](const T& x) { return f(x, y); }, x, hx);
        T dfdy = central_diff<T>([&](const T& y) { return f(x, y); }, y, hy);
        return Eigen::Matrix<T, 2, 1>(dfdx, dfdy);
    }

    // 2D forward
    template<typename T, typename Func>
    Eigen::Matrix<T, 2, 1> gradient_2d_forward(const Func& f, const T& x, const T& y, const T& hx, const T& hy) {
        T dfdx = forward_diff<T>([&](const T& x) { return f(x, y); }, x, hx);
        T dfdy = forward_diff<T>([&](const T& y) { return f(x, y); }, y, hy);
        return Eigen::Matrix<T, 2, 1>(dfdx, dfdy);
    }

    // 2D backward
    template<typename T, typename Func>
    Eigen::Matrix<T, 2, 1> gradient_2d_backward(const Func& f, const T& x, const T& y, const T& hx, const T& hy) {
        T dfdx = backward_diff<T>([&](const T& x) { return f(x, y); }, x, hx);
        T dfdy = backward_diff<T>([&](const T& y) { return f(x, y); }, y, hy);
        return Eigen::Matrix<T, 2, 1>(dfdx, dfdy);
    }

    // 3D central
    template<typename T, typename Func>
    Eigen::Matrix<T, 3, 1> gradient_3d_central(const Func& f, const T& x, const T& y, const T& z,
        const T& hx, const T& hy, const T& hz) {
        T dfdx = central_diff<T>([&](const T& x) { return f(x, y, z); }, x, hx);
        T dfdy = central_diff<T>([&](const T& y) { return f(x, y, z); }, y, hy);
        T dfdz = central_diff<T>([&](const T& z) { return f(x, y, z); }, z, hz);
        return Eigen::Matrix<T, 3, 1>(dfdx, dfdy, dfdz);
    }

    // 3D forward
    template<typename T, typename Func>
    Eigen::Matrix<T, 3, 1> gradient_3d_forward(const Func& f, const T& x, const T& y, const T& z,
        const T& hx, const T& hy, const T& hz) {
        T dfdx = forward_diff<T>([&](const T& x) { return f(x, y, z); }, x, hx);
        T dfdy = forward_diff<T>([&](const T& y) { return f(x, y, z); }, y, hy);
        T dfdz = forward_diff<T>([&](const T& z) { return f(x, y, z); }, z, hz);
        return Eigen::Matrix<T, 3, 1>(dfdx, dfdy, dfdz);
    }

    // 3D backward
    template<typename T, typename Func>
    Eigen::Matrix<T, 3, 1> gradient_3d_backward(const Func& f, const T& x, const T& y, const T& z,
        const T& hx, const T& hy, const T& hz) {
        T dfdx = backward_diff<T>([&](const T& x) { return f(x, y, z); }, x, hx);
        T dfdy = backward_diff<T>([&](const T& y) { return f(x, y, z); }, y, hy);
        T dfdz = backward_diff<T>([&](const T& z) { return f(x, y, z); }, z, hz);
        return Eigen::Matrix<T, 3, 1>(dfdx, dfdy, dfdz);
    }

} // namespace delta::numerical