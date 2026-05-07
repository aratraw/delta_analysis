// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/geometry/discrete_action.h
// ============================================================================
// DISCRETE ACTIONS FOR VARIATIONAL PRINCIPLES
// (compile‑time polymorphism, multidimensional, zero overhead)
// ============================================================================
#pragma once

#include <cstddef>
#include <vector>
#include <array>
#include "delta/core/rational.h"
#include "delta/core/uniform_grid.h"
#include "delta/core/product_grid.h"

namespace delta::geometry {

    // =========================================================================
    // Free particle action (1D trajectory in time)
    // =========================================================================
    template<typename Scalar>
    class FreeParticleAction {
    public:
        explicit FreeParticleAction(Scalar mass = Scalar(1)) : mass_(mass) {}

        Scalar evaluate(const std::vector<Scalar>& traj, Scalar dt) const {
            std::size_t n = traj.size();
            if (n < 2) return Scalar(0);
            Scalar S = 0;
            for (std::size_t i = 0; i + 1 < n; ++i) {
                Scalar dx = traj[i + 1] - traj[i];
                S += (mass_ / Scalar(2)) * (dx * dx) / dt;
            }
            return S * dt;
        }

        Scalar variation(const std::vector<Scalar>& traj, std::size_t i, Scalar dt) const {
            if (i == 0 || i + 1 >= traj.size()) return Scalar(0);
            return -(mass_ / dt) * (traj[i + 1] - Scalar(2) * traj[i] + traj[i - 1]);
        }

    private:
        Scalar mass_;
    };

    // =========================================================================
    // Scalar field action on ProductGrid<UniformGrid<Scalar>, Dim>
    // (works for Dim = 1, 2, 3, …)
    // =========================================================================
    template<typename Scalar, int Dim, typename Pot, typename PotDeriv>
    class ScalarFieldAction {
    public:
        ScalarFieldAction(Pot V, PotDeriv dV, Scalar stiffness = Scalar(1))
            : V_(std::move(V)), dV_(std::move(dV)), stiffness_(stiffness) {
        }

        /// Evaluate action for a field stored as flat vector (row‑major order)
        Scalar evaluate(const std::vector<Scalar>& phi,
            const std::array<std::size_t, Dim>& sizes,
            const std::array<Scalar, Dim>& steps) const
        {
            Scalar S = 0;
            const Scalar vol = cell_volume(steps);
            auto loop = [&](auto&& self, int d, std::array<std::size_t, Dim>& idx) -> void {
                if constexpr (d == Dim) {
                    for (int k = 0; k < Dim; ++k)
                        if (idx[k] == 0 || idx[k] == sizes[k] - 1)
                            return;
                    std::size_t pos = flat(idx, sizes);
                    Scalar phi_cur = phi[pos];
                    Scalar grad2 = 0;
                    for (int k = 0; k < Dim; ++k) {
                        std::array<std::size_t, Dim> next = idx, prev = idx;
                        ++next[k];
                        --prev[k];
                        Scalar phi_next = phi[flat(next, sizes)];
                        Scalar phi_prev = phi[flat(prev, sizes)];
                        Scalar diff = (phi_next - phi_cur) / steps[k];
                        grad2 += diff * diff;
                    }
                    S += (stiffness_ / Scalar(2)) * grad2 * vol;
                    S += V_(phi_cur) * vol;
                }
                else {
                    for (idx[d] = 0; idx[d] < sizes[d]; ++idx[d])
                        self(self, d + 1, idx);
                }
                };
            std::array<std::size_t, Dim> idx{};
            loop(loop, 0, idx);
            return S;
        }

        /// Variation at a flat index `pos` (must be interior)
        Scalar variation(const std::vector<Scalar>& phi,
            std::size_t pos,
            const std::array<std::size_t, Dim>& sizes,
            const std::array<Scalar, Dim>& steps) const
        {
            std::size_t n = 1;
            for (int d = 0; d < Dim; ++d) n *= sizes[d];
            if (pos >= n) return Scalar(0);

            // Convert flat index to multi-index
            std::array<std::size_t, Dim> idx;
            std::size_t rem = pos;
            std::array<std::size_t, Dim> strides;
            std::size_t stride = 1;
            for (int d = Dim - 1; d >= 0; --d) {
                strides[d] = stride;
                stride *= sizes[d];
            }
            for (int d = 0; d < Dim; ++d) {
                idx[d] = (pos / strides[d]) % sizes[d];
            }

            // Check interior
            for (int d = 0; d < Dim; ++d)
                if (idx[d] == 0 || idx[d] == sizes[d] - 1)
                    return Scalar(0);

            Scalar laplacian = 0;
            for (int d = 0; d < Dim; ++d) {
                std::array<std::size_t, Dim> next = idx, prev = idx;
                ++next[d];
                --prev[d];
                Scalar phi_next = phi[flat(next, sizes)];
                Scalar phi_prev = phi[flat(prev, sizes)];
                laplacian += (phi_next - Scalar(2) * phi[pos] + phi_prev) / (steps[d] * steps[d]);
            }
            const Scalar vol = cell_volume(steps);
            // Correct formula: dS/dφ = h^d * (-stiffness * Δφ + dV)
            return (-stiffness_ * laplacian + dV_(phi[pos])) * vol;
        }

    private:
        Pot V_;
        PotDeriv dV_;
        Scalar stiffness_;

        static Scalar cell_volume(const std::array<Scalar, Dim>& steps) {
            Scalar v = 1;
            for (int d = 0; d < Dim; ++d) v *= steps[d];
            return v;
        }

        static std::size_t flat(const std::array<std::size_t, Dim>& idx,
            const std::array<std::size_t, Dim>& sizes) {
            std::size_t pos = 0;
            std::size_t stride = 1;
            for (int d = Dim - 1; d >= 0; --d) {
                pos += idx[d] * stride;
                stride *= sizes[d];
            }
            return pos;
        }
    };

} // namespace delta::geometry