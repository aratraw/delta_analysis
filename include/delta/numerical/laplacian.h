// include/delta/numerical/laplacian.h
#pragma once

#include <functional>

namespace delta::numerical {

    // 2D uniform grid (equal steps)
    template<typename T, typename Func>
    T laplacian_2d_5point(const Func& f, const T& x, const T& y, const T& h) {
        T center = f(x, y);
        T right = f(x + h, y);
        T left = f(x - h, y);
        T up = f(x, y + h);
        T down = f(x, y - h);
        return (right + left + up + down - T(4) * center) / (h * h);
    }

    // 2D rectangular grid (different steps)
    template<typename T, typename Func>
    T laplacian_2d_5point(const Func& f, const T& x, const T& y, const T& hx, const T& hy) {
        T center = f(x, y);
        T right = f(x + hx, y);
        T left = f(x - hx, y);
        T up = f(x, y + hy);
        T down = f(x, y - hy);
        T d2x = (right - T(2) * center + left) / (hx * hx);
        T d2y = (up - T(2) * center + down) / (hy * hy);
        return d2x + d2y;
    }

    // 3D uniform grid
    template<typename T, typename Func>
    T laplacian_3d_7point(const Func& f, const T& x, const T& y, const T& z, const T& h) {
        T center = f(x, y, z);
        T right = f(x + h, y, z);
        T left = f(x - h, y, z);
        T up = f(x, y + h, z);
        T down = f(x, y - h, z);
        T front = f(x, y, z + h);
        T back = f(x, y, z - h);
        return (right + left + up + down + front + back - T(6) * center) / (h * h);
    }

} // namespace delta::numerical