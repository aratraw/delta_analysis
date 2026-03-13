// tests/numerical/test_finite_differences.cpp
#include <gtest/gtest.h>
#include "delta/numerical/finite_differences.h"
#include "delta/numerical/gradient.h"
#include "delta/numerical/laplacian.h"

// -----------------------------------------------------------------------------
// Tests for finite difference utilities (1D, 2D, 3D)
// -----------------------------------------------------------------------------

TEST(FiniteDifferencesTest, OneDimensional) {
    auto f = [](double x) { return x * x; }; // f'(x) = 2x
    double x = 2.0;
    double h = 0.1;

    double fd = delta::numerical::forward_diff(f, x, h);
    double bd = delta::numerical::backward_diff(f, x, h);
    double cd = delta::numerical::central_diff(f, x, h);

    EXPECT_NEAR(fd, 4.0, 0.11);   // O(h) error
    EXPECT_NEAR(bd, 4.0, 0.11);
    EXPECT_NEAR(cd, 4.0, 0.01);   // O(h²) error
}

TEST(GradientTest, Quadratic2D) {
    auto f = [](double x, double y) { return x * x + y * y; };
    double h = 0.1;
    double x = 1.0, y = 2.0;
    auto grad = delta::numerical::gradient_2d_central(f, x, y, h, h);
    EXPECT_NEAR(grad[0], 2.0, 1e-6);
    EXPECT_NEAR(grad[1], 4.0, 1e-6);
}

TEST(GradientTest, Linear2D) {
    auto f = [](double x, double y) { return 3.0 * x + 5.0 * y; };
    double h = 0.1;
    double x = 1.0, y = 2.0;
    auto grad_central = delta::numerical::gradient_2d_central(f, x, y, h, h);
    EXPECT_NEAR(grad_central[0], 3.0, 1e-12);
    EXPECT_NEAR(grad_central[1], 5.0, 1e-12);

    auto grad_forward = delta::numerical::gradient_2d_forward(f, x, y, h, h);
    EXPECT_NEAR(grad_forward[0], 3.0, 1e-12);
    EXPECT_NEAR(grad_forward[1], 5.0, 1e-12);

    auto grad_backward = delta::numerical::gradient_2d_backward(f, x, y, h, h);
    EXPECT_NEAR(grad_backward[0], 3.0, 1e-12);
    EXPECT_NEAR(grad_backward[1], 5.0, 1e-12);
}

TEST(GradientTest, Quadratic2D_ForwardBackward) {
    auto f = [](double x, double y) { return x * x + y * y; };
    double h = 0.1;
    double x = 1.0, y = 2.0;

    auto grad_f = delta::numerical::gradient_2d_forward(f, x, y, h, h);
    auto grad_b = delta::numerical::gradient_2d_backward(f, x, y, h, h);

    EXPECT_NEAR(grad_f[0], 2.0, 0.11);
    EXPECT_NEAR(grad_f[1], 4.0, 0.11);
    EXPECT_NEAR(grad_b[0], 2.0, 0.11);
    EXPECT_NEAR(grad_b[1], 4.0, 0.11);
}

TEST(LaplacianTest, Quadratic2D) {
    auto f = [](double x, double y) { return x * x + y * y; }; // Δf = 4
    double h = 0.1;
    double x = 1.0, y = 2.0;
    double lap = delta::numerical::laplacian_2d_5point(f, x, y, h);
    EXPECT_NEAR(lap, 4.0, 1e-6);

    // different steps
    double hx = 0.1, hy = 0.2;
    lap = delta::numerical::laplacian_2d_5point(f, x, y, hx, hy);
    EXPECT_NEAR(lap, 4.0, 1e-6);
}

TEST(LaplacianTest, Cubic2D) {
    auto f = [](double x, double y) { return x * x * x + y * y * y; }; // Δf = 6x + 6y
    double x = 1.0, y = 2.0;
    double expected = 6.0 * x + 6.0 * y; // = 18

    double h = 0.1;
    double lap = delta::numerical::laplacian_2d_5point(f, x, y, h);
    EXPECT_NEAR(lap, expected, 0.1); // O(h²) error ~0.1

    double h_small = 0.01;
    double lap_small = delta::numerical::laplacian_2d_5point(f, x, y, h_small);
    EXPECT_NEAR(lap_small, expected, 0.001);
}

#ifdef DELTA_USE_3D
TEST(LaplacianTest, Quadratic3D) {
    auto f = [](double x, double y, double z) { return x * x + y * y + z * z; }; // Δf = 6
    double h = 0.1;
    double x = 1.0, y = 2.0, z = 3.0;
    double lap = delta::numerical::laplacian_3d_7point(f, x, y, z, h);
    EXPECT_NEAR(lap, 6.0, 1e-6);
}
#endif