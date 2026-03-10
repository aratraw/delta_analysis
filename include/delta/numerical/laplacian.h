// include/delta/numerical/laplacian.h
#pragma once

#include <functional>

namespace delta::numerical {

    /**
     * @brief 5‑point stencil Laplacian for a 2D uniform grid (equal steps h).
     * Δf ≈ (f(x+h,y) + f(x-h,y) + f(x,y+h) + f(x,y-h) - 4f(x,y)) / h²
     */
    template<typename Func>
    double laplacian_2d_5point(const Func& f, double x, double y, double h) {
        double center = f(x, y);
        double right = f(x + h, y);
        double left = f(x - h, y);
        double up = f(x, y + h);
        double down = f(x, y - h);
        return (right + left + up + down - 4.0 * center) / (h * h);
    }

    /**
     * @brief 5‑point stencil Laplacian for a 2D rectangular grid (different steps).
     * Δf ≈ ( (f(x+hx,y) - 2f(x,y) + f(x-hx,y))/hx² + (f(x,y+hy) - 2f(x,y) + f(x,y-hy))/hy² )
     */
    template<typename Func>
    double laplacian_2d_5point(const Func& f, double x, double y, double hx, double hy) {
        double center = f(x, y);
        double right = f(x + hx, y);
        double left = f(x - hx, y);
        double up = f(x, y + hy);
        double down = f(x, y - hy);
        double d2x = (right - 2.0 * center + left) / (hx * hx);
        double d2y = (up - 2.0 * center + down) / (hy * hy);
        return d2x + d2y;
    }

    /**
     * @brief 7‑point stencil Laplacian for a 3D uniform grid.
     * Δf ≈ sum over neighbours (f(x±h,...) - 2f) / h²
     */
    template<typename Func>
    double laplacian_3d_7point(const Func& f, double x, double y, double z, double h) {
        double center = f(x, y, z);
        double right = f(x + h, y, z);
        double left = f(x - h, y, z);
        double up = f(x, y + h, z);
        double down = f(x, y - h, z);
        double front = f(x, y, z + h);
        double back = f(x, y, z - h);
        return (right + left + up + down + front + back - 6.0 * center) / (h * h);
    }

} // namespace delta::numerical