// include/delta/numerical/gradient.h
#pragma once

#include <functional>
#include <Eigen/Dense>
#include "finite_differences.h"

namespace delta::numerical {

    /**
     * @brief Gradient of a 2D scalar field using central differences.
     * @tparam Func  Callable with signature double(double x, double y)
     * @param f      Function to differentiate
     * @param x,y    Coordinates of the point
     * @param hx,hy  Step sizes in x and y directions
     * @return Eigen::Vector2d (df/dx, df/dy)
     */
    template<typename Func>
    Eigen::Vector2d gradient_2d_central(const Func& f, double x, double y, double hx, double hy) {
        double dfdx = central_diff([&](double x) { return f(x, y); }, x, hx);
        double dfdy = central_diff([&](double y) { return f(x, y); }, y, hy);
        return Eigen::Vector2d(dfdx, dfdy);
    }

    /**
     * @brief Gradient of a 3D scalar field using central differences.
     */
    template<typename Func>
    Eigen::Vector3d gradient_3d_central(const Func& f, double x, double y, double z,
        double hx, double hy, double hz) {
        double dfdx = central_diff([&](double x) { return f(x, y, z); }, x, hx);
        double dfdy = central_diff([&](double y) { return f(x, y, z); }, y, hy);
        double dfdz = central_diff([&](double z) { return f(x, y, z); }, z, hz);
        return Eigen::Vector3d(dfdx, dfdy, dfdz);
    }

    /**
 * @brief Gradient of a 2D scalar field using forward differences.
 */
    template<typename Func>
    Eigen::Vector2d gradient_2d_forward(const Func& f, double x, double y, double hx, double hy) {
        double dfdx = forward_diff([&](double x) { return f(x, y); }, x, hx);
        double dfdy = forward_diff([&](double y) { return f(x, y); }, y, hy);
        return Eigen::Vector2d(dfdx, dfdy);
    }

    /**
     * @brief Gradient of a 2D scalar field using backward differences.
     */
    template<typename Func>
    Eigen::Vector2d gradient_2d_backward(const Func& f, double x, double y, double hx, double hy) {
        double dfdx = backward_diff([&](double x) { return f(x, y); }, x, hx);
        double dfdy = backward_diff([&](double y) { return f(x, y); }, y, hy);
        return Eigen::Vector2d(dfdx, dfdy);
    }

    // Аналогично для 3D
    template<typename Func>
    Eigen::Vector3d gradient_3d_forward(const Func& f, double x, double y, double z,
        double hx, double hy, double hz) {
        double dfdx = forward_diff([&](double x) { return f(x, y, z); }, x, hx);
        double dfdy = forward_diff([&](double y) { return f(x, y, z); }, y, hy);
        double dfdz = forward_diff([&](double z) { return f(x, y, z); }, z, hz);
        return Eigen::Vector3d(dfdx, dfdy, dfdz);
    }

    template<typename Func>
    Eigen::Vector3d gradient_3d_backward(const Func& f, double x, double y, double z,
        double hx, double hy, double hz) {
        double dfdx = backward_diff([&](double x) { return f(x, y, z); }, x, hx);
        double dfdy = backward_diff([&](double y) { return f(x, y, z); }, y, hy);
        double dfdz = backward_diff([&](double z) { return f(x, y, z); }, z, hz);
        return Eigen::Vector3d(dfdx, dfdy, dfdz);
    }
} // namespace delta::numerical