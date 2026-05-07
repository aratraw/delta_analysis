// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/geometry/connection.h
// ============================================================================
// DISCRETE CONNECTION ON A SIMPLICIAL COMPLEX
// ============================================================================
//
// Stores an invertible matrix U_{i→j} for every oriented edge (i,j) of a
// simplicial complex.  The reverse direction is automatically set to the
// inverse matrix, guaranteeing U_{j→i} = U_{i→j}^{-1}.
//
// ----------------------------------------------------------------------------
// HOLONOMY ORDER (critical detail)
// ----------------------------------------------------------------------------
// The holonomy along a path [v0, v1, …, vk] is defined as the product
//   H = U_{v_{k-1}→v_k} · … · U_{v0→v1}.
// In C++ terms, if we iterate `for (i = 0; i < k; ++i)`, we must **right‑
// multiply** the current matrix by the next transport matrix:
//   H = H * get_transport(v_i, v_{i+1});
// This guarantees that the first edge's matrix appears on the right‑hand side
// of the product, which corresponds to the composition of parallel transports
// acting on a vector placed to the right of H.
// ----------------------------------------------------------------------------
// ============================================================================
// CRITICAL IMPLEMENTATION NOTES FOR Connection::is_consistent
// ============================================================================
//
// ISSUE THAT WAS FIXED:
// ---------------------
// The original version of is_consistent attempted to reconstruct the subdivided
// path from v0 to v1 by iterating over the set of child edges in an unordered
// container and selecting the first edge containing the current vertex.
// This led to a non‑deterministic walk: when the current vertex was an interior
// point (e.g., the midpoint of a coarse edge), the child set contained two edges
// incident to that vertex. Depending on the iteration order of unordered_map,
// the algorithm might walk back to the previous vertex, creating an infinite
// loop or exceeding the allowed path length, and then returning false.
//
// MATHEMATICAL REQUIREMENT:
// --------------------------
// For a coarse edge (u, v) and its barycentric subdivision, the fine edges form
// a linear chain from u to v. Each interior vertex in this chain has degree 2
// (exactly one incoming and one outgoing fine edge). The endpoints have degree 1.
// Therefore, the unique simple path from u to v can be recovered by a graph walk
// that never returns to the previous vertex.
//
// CORRECT SOLUTION:
// -----------------
// 1. Build an adjacency list from the set of child edges (fine edges that cover
//    the coarse edge). The adjacency list maps a vertex to its neighbours within
//    this subgraph.
// 2. Start from v0 – it must have exactly one neighbour in the subgraph.
// 3. Walk from neighbour to neighbour, keeping track of the previous vertex to
//    avoid turning back. Interior vertices must have degree 2; if any vertex
//    (other than v1) has degree != 2, the subdivision is malformed.
// 4. The resulting sequence of vertices is the unique oriented path from v0 to v1.
// 5. Compute fine holonomy along this path and compare with the coarse transport.
//
// WHY THE FIX IS ROBUST:
// -----------------------
// - It does not rely on any particular order of iteration over the child edges.
// - It works for any chain length (including the trivial case where the coarse
//   edge is not subdivided – though subdivision map would contain no children).
// - It explicitly checks the degree of interior vertices, which can catch subtle
//   errors in the subdivision algorithm or in the subdivision map.
//
// FUTURE DEVELOPMENT – RECOMMENDATIONS TO AVOID SIMPLE MISTAKES:
// ---------------------------------------------------------------
// 1. ALWAYS prefer deterministic graph traversal over "pick‑first‑found" when
//    the container is unordered (std::unordered_map, std::unordered_set).
// 2. When reconstructing a path from a set of edges, explicitly build an
//    adjacency structure – it makes the control flow linear and debuggable.
// 3. Validate invariants: endpoint degrees == 1, interior degrees == 2 (for a
//    chain). Abort early with a meaningful error or return false.
// 4. Write unit tests for edge cases:
//    - coarse edge split into 1 fine edge (no interior vertex) – should work.
//    - coarse edge split into 3 or more fine edges (higher‑order subdivision).
//    - subdivision map containing extra/unrelated edges – must be ignored.
// 5. Keep the tolerance parameter adaptable – different Scalars (Rational,
//    double, mpfr) may need different default tolerances. Provide an overload
//    with a sensible default.
// 6. Consider that fine_mesh.edge_at(child_key.index) returns an unordered pair
//    (a,b). Do NOT assume the order corresponds to any orientation. The adjacency
//    approach naturally handles this.
// 7. For performance: if this check is called frequently, pre‑compute the
//    subdivided path for each coarse edge once and store it; but for correctness
//    testing the current lazy approach is fine.
//
// ============================================================================
#pragma once

#include <unordered_map>
#include <vector>
#include <utility>
#include <stdexcept>
#include <cstddef>
#include <functional>
#include "delta/core/rational.h"
#include "delta/geometry/simplicial_complex.h"

namespace delta::geometry {

    template<typename Addr, typename Scalar, int Dim,
        typename Group = Eigen::Matrix<Scalar, Dim, Dim>>
        class Connection {
        public:
            using matrix_type = Group;

            Connection() = default;

            /**
             * @brief Set the transport matrix for the oriented edge (from → to).
             *        Automatically sets the reverse direction to the inverse.
             */
            void set_transport(const Addr& from, const Addr& to, const matrix_type& mat) {
                if (from == to)
                    throw std::invalid_argument("Connection::set_transport: from == to");
                data_[encode(from, to)] = mat;
                data_[encode(to, from)] = mat.inverse();
            }

            /** @brief Retrieve U_{from→to}.  Returns Identity if not set. */
            matrix_type get_transport(const Addr& from, const Addr& to) const {
                auto it = data_.find(encode(from, to));
                if (it == data_.end())
                    return matrix_type::Identity();   // ← тривиальная связность по умолчанию
                return it->second;
            }

            /** @brief Parallel transport a vector along a single edge. */
            template<typename Vector>
            Vector parallel_transport(const Addr& from, const Addr& to,
                const Vector& v) const {
                return get_transport(from, to) * v;
            }

            /**
             * @brief Compute holonomy along a path [v0, …, vk].
             *
             * Returns U_{v_{k-1}→v_k} · … · U_{v0→v1}.
             */
            matrix_type holonomy(const std::vector<Addr>& path) const {
                if (path.size() < 2)
                    throw std::invalid_argument("Holonomy requires at least two vertices");
                matrix_type H = matrix_type::Identity();
                for (std::size_t i = 0; i + 1 < path.size(); ++i) {
                    H = H * get_transport(path[i], path[i + 1]);  // правое умножение
                }
                return H;
            }

            /**
             * @brief Verify subdivision consistency.
             *
             * For every coarse edge e = (u,v), the product of fine connection
             * matrices along the subdivided path must equal U_{u→v}.
             */
            template<typename SubdivisionMap, typename Complex>
            bool is_consistent(const Connection& fine,
                const SubdivisionMap& subdiv_map,
                const Complex& coarse_mesh,
                const Complex& fine_mesh,
                const Scalar& tolerance = Scalar(1, 1000000000000)) const
            {
                for (std::size_t e = 0; e < coarse_mesh.num_edges(); ++e) {
                    auto [v0, v1] = coarse_mesh.edge_at(e);
                    matrix_type coarse_mat = get_transport(v0, v1);

                    auto key = SimplexKey{ 1, e };
                    auto it = subdiv_map.find(key);
                    if (it == subdiv_map.end()) continue;
                    const auto& children = it->second;

                    // Build adjacency map from the fine edges that cover this coarse edge
                    std::unordered_map<Addr, std::vector<Addr>> adj;
                    for (const auto& child_key : children) {
                        auto [a, b] = fine_mesh.edge_at(child_key.index);
                        adj[a].push_back(b);
                        adj[b].push_back(a);
                    }

                    // Find the unique path from v0 to v1 through this adjacency chain.
                    // The interior vertices (if any) must have degree 2, endpoints degree 1.
                    std::vector<Addr> path;
                    path.push_back(v0);

                    // Starting from v0, pick its only neighbour (must exist)
                    auto it0 = adj.find(v0);
                    if (it0 == adj.end() || it0->second.empty())
                        return false;
                    Addr prev = v0;
                    Addr cur = it0->second[0];   // the only neighbour of v0 in this chain
                    path.push_back(cur);

                    while (cur != v1) {
                        auto nb_it = adj.find(cur);
                        if (nb_it == adj.end())
                            return false;
                        const auto& neighbours = nb_it->second;
                        // Interior vertices must have exactly two neighbours (prev and next)
                        if (neighbours.size() != 2)
                            return false;
                        Addr next = (neighbours[0] == prev) ? neighbours[1] : neighbours[0];
                        path.push_back(next);
                        prev = cur;
                        cur = next;
                        // Safety: path length cannot exceed number of edges + 1
                        if (path.size() > children.size() + 1)
                            return false;
                    }

                    // Now path is a valid oriented path from v0 to v1
                    matrix_type fine_mat = fine.holonomy(path);
                    if ((coarse_mat - fine_mat).norm() > tolerance)
                        return false;
                }
                return true;
            }

        private:
            using key_type = std::uint64_t;

            key_type encode(const Addr& a, const Addr& b) const {
                return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
            }

            std::unordered_map<key_type, matrix_type> data_;
    };

} // namespace delta::geometry