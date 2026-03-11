// include/delta/numerical/gauge_theory.h
#pragma once

#include "delta/geometry/simplicial_complex.h"
#include <complex>
#include <vector>
#include <cmath>

namespace delta::numerical {

    /**
     * @brief Lattice gauge field for U(1) on a simplicial complex.
     *
     * Stores a complex phase on each oriented edge (U(1) group element).
     */
    template<typename Complex>
    class U1GaugeField {
    public:
        using value_type = std::complex<double>;
        using edge_type = typename Complex::edge_type;

        explicit U1GaugeField(const Complex& mesh) : mesh_(mesh), links_(mesh.num_edges(), 1.0) {}

        // Access link variable on edge (oriented)
        value_type& link(std::size_t e) { return links_[e]; }
        const value_type& link(std::size_t e) const { return links_[e]; }

        // Wilson action: S = β * Σ_plaquettes (1 - Re U_plaquette)
        double wilson_action(double beta) const {
            double sum = 0.0;
            for (std::size_t t = 0; t < mesh_.num_triangles(); ++t) {
                auto tri = mesh_.triangle(t);
                auto e01 = mesh_.find_edge(tri[0], tri[1]);
                auto e12 = mesh_.find_edge(tri[1], tri[2]);
                auto e20 = mesh_.find_edge(tri[2], tri[0]);
                if (e01 < 0 || e12 < 0 || e20 < 0) continue;
                // Plaquette product: U01 * U12 * U20 (oriented consistently)
                value_type prod = links_[e01] * links_[e12] * links_[e20];
                sum += 1.0 - std::real(prod);
            }
            return beta * sum;
        }

        // Randomize links (for testing)
        void randomize(double amplitude) {
            for (auto& u : links_) {
                double angle = amplitude * (2.0 * std::rand() / RAND_MAX - 1.0);
                u = std::exp(std::complex<double>(0.0, angle));
            }
        }

    private:
        const Complex& mesh_;
        std::vector<value_type> links_;
    };

    // Variation of action with respect to a link (for U(1))
    template<typename Complex>
    std::complex<double> wilson_variation(const U1GaugeField<Complex>& field,
        std::size_t edge_idx) {
        // Compute the "force" on a link: derivative of action w.r.t. link variable.
        // For U(1), it's the sum of plaquettes containing that edge.
        // Not fully implemented.
        return 0.0;
    }

} // namespace delta::numerical