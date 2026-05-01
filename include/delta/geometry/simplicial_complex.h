// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

//include/delta/geometry/simplicial_complex.h
#pragma once

#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <map>
#include <optional>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <Eigen/Dense>
#include "delta/core/rational.h"
#include "delta/core/grid_concept.h"
#include "delta/core/regulative_idea.h"
#include "delta/geometry/constructive_core.h"  // ДОБАВЛЕНО для Vector

namespace delta::geometry {
    using namespace delta;
    // Forward declarations
    template<int Dim, typename Coord>
    class SimplicialComplex;

    // -------------------------------------------------------------------------
    // SimplexKey and SubdivisionMap
    // -------------------------------------------------------------------------

    /**
     * @brief Key for identifying a simplex in a complex.
     *
     * Used for subdivision mapping: (dimension, index) uniquely identifies a simplex.
     */
    struct SimplexKey {
        int dim;
        std::size_t index;

        bool operator==(const SimplexKey& other) const noexcept {
            return dim == other.dim && index == other.index;
        }
    };

    /**
     * @brief Hash functor for SimplexKey.
     */
    struct SimplexKeyHash {
        std::size_t operator()(const SimplexKey& k) const noexcept {
            return std::hash<int>{}(k.dim) ^ (std::hash<std::size_t>{}(k.index) << 1);
        }
    };

    /**
     * @brief Map from coarse simplex to list of fine simplices after subdivision.
     *
     * For each simplex in the original complex (identified by (dim, index)),
     * stores a vector of SimplexKeys in the subdivided complex that cover it.
     */
    using SubdivisionMap = std::unordered_map<SimplexKey, std::vector<SimplexKey>, SimplexKeyHash>;

    // -------------------------------------------------------------------------
    // SimplexHasher (for sorting vertices in maps)
    // -------------------------------------------------------------------------

    /**
     * @brief Hash functor for a vector of vertex indices (used for simplex lookup).
     *
     * Combines hashes of individual indices using XOR with shifts.
     */
    struct SimplexHasher {
        template<typename T>
        std::size_t operator()(const std::vector<T>& v) const noexcept {
            std::size_t seed = v.size();
            for (const auto& i : v) {
                seed ^= std::hash<T>{}(i)+0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    // -------------------------------------------------------------------------
    // SimplicialComplex class
    // -------------------------------------------------------------------------

    /**
     * @brief A simplicial complex of fixed dimension Dim with coordinates of type Coord.
     *
     * Stores vertices and all simplices up to dimension Dim.
     * Satisfies OrderedGrid concept (vertices are ordered and accessible).
     *
     * @tparam Dim Dimension of the complex (1,2,3,...)
     * @tparam Coord Coordinate type (typically Rational)
     */
    template<int Dim, typename Coord = Rational>
    class SimplicialComplex {
        static_assert(Dim > 0, "Dimension must be positive");

    public:
        static constexpr int Dimension = Dim;
        // ---------------------------------------------------------------------
        // Type aliases
        // ---------------------------------------------------------------------
        using point_type = Eigen::Matrix<Coord, Dim, 1>;
        using scalar_type = Coord;
        using vertex_index = std::size_t;
        using simplex = std::vector<vertex_index>;
        using edge_type = std::array<vertex_index, 2>;
        using triangle_type = std::array<vertex_index, 3>;
        using tetrahedron_type = std::array<vertex_index, 4>;

        // For OrderedGrid concept - кастомный компаратор для точек
        struct PointLess {
            bool operator()(const point_type& a, const point_type& b) const {
                for (int i = 0; i < Dim; ++i) {
                    if (a[i] < b[i]) return true;
                    if (b[i] < a[i]) return false;
                }
                return false;  // равны
            }
        };
        using comparator_type = PointLess;

        // For Grid concept compatibility
        using value_type = point_type;
        using size_type = std::size_t;
        using const_iterator = typename std::vector<point_type>::const_iterator;

        // ---------------------------------------------------------------------
        // Constructors
        // ---------------------------------------------------------------------
        SimplicialComplex() = default;

        // Copy constructor (explicitly defaulted)
        SimplicialComplex(const SimplicialComplex&) = default;

        // Move constructor
        SimplicialComplex(SimplicialComplex&&) = default;

        // ---------------------------------------------------------------------
        // Vertex management
        // ---------------------------------------------------------------------

        /**
         * @brief Add a vertex to the complex.
         * @param p Point coordinates (must be non-zero? No, but is_in_K can check later)
         * @return Index of the new vertex.
         */
        vertex_index add_vertex(const point_type& p) {
            vertex_index idx = vertices_.size();
            vertices_.push_back(p);
            return idx;
        }

        /**
         * @brief Get vertex by index.
         * @param i Vertex index.
         * @return Const reference to the vertex.
         * @throws std::out_of_range if index is invalid.
         */
        const point_type& vertex(vertex_index i) const {
            if (i >= vertices_.size()) {
                throw std::out_of_range("SimplicialComplex::vertex: index out of range");
            }
            return vertices_[i];
        }

        /**
         * @brief Number of vertices.
         */
        std::size_t num_vertices() const noexcept {
            return vertices_.size();
        }

        // ---------------------------------------------------------------------
        // Edge management
        // ---------------------------------------------------------------------

        /**
         * @brief Add an edge between two vertices.
         * @param v0 First vertex index.
         * @param v1 Second vertex index.
         * @return true if edge was added, false if it already exists or vertices invalid.
         */
        bool add_edge(vertex_index v0, vertex_index v1) {
            // Validate vertices
            if (v0 >= vertices_.size() || v1 >= vertices_.size() || v0 == v1) {
                return false;
            }

            // Normalize orientation (store smaller index first)
            if (v0 > v1) std::swap(v0, v1);
            simplex edge = { v0, v1 };

            // Check if already exists
            auto& edge_map = simplices_map_[1];
            if (edge_map.find(edge) != edge_map.end()) {
                return false;
            }

            // Add to storage
            std::size_t idx = simplices_[1].size();
            simplices_[1].push_back(edge);
            edge_map[edge] = idx;
            return true;
        }

        /**
         * @brief Get edge by index.
         * @param idx Edge index (0..num_edges()-1).
         * @return Array of two vertex indices.
         * @throws std::out_of_range if index invalid.
         */
        edge_type edge_at(std::size_t idx) const {
            const auto& edges = simplices_.find(1);
            if (edges == simplices_.end() || idx >= edges->second.size()) {
                throw std::out_of_range("SimplicialComplex::edge_at: index out of range");
            }
            const auto& e = edges->second[idx];
            if (e.size() != 2) {
                throw std::logic_error("SimplicialComplex::edge_at: stored simplex is not an edge");
            }
            return { e[0], e[1] };
        }

        /**
         * @brief Number of edges.
         */
        std::size_t num_edges() const noexcept {
            auto it = simplices_.find(1);
            return it == simplices_.end() ? 0 : it->second.size();
        }

        // ---------------------------------------------------------------------
        // Triangle management (2-simplices)
        // ---------------------------------------------------------------------

        /**
         * @brief Add a triangle.
         * @param v0,v1,v2 Vertex indices.
         * @return true if added, false if degenerate, already exists, or vertices invalid.
         */
        bool add_triangle(vertex_index v0, vertex_index v1, vertex_index v2) {
            // Validate vertices
            if (v0 >= vertices_.size() || v1 >= vertices_.size() || v2 >= vertices_.size()) {
                return false;
            }
            if (v0 == v1 || v0 == v2 || v1 == v2) {
                return false;  // degenerate
            }

            // Check non-degeneracy (collinearity)
            if (!is_non_degenerate({ v0, v1, v2 })) {
                return false;
            }

            // Normalize orientation (sort for storage)
            std::vector<vertex_index> tri = { v0, v1, v2 };
            std::sort(tri.begin(), tri.end());

            // Check if already exists
            auto& tri_map = simplices_map_[2];
            if (tri_map.find(tri) != tri_map.end()) {
                return false;
            }

            // Add to storage
            std::size_t idx = simplices_[2].size();
            simplices_[2].push_back(tri);
            tri_map[tri] = idx;
            return true;
        }

        /**
         * @brief Get triangle by index.
         * @param idx Triangle index (0..num_triangles()-1).
         * @return Array of three vertex indices.
         * @throws std::out_of_range if index invalid.
         */
        triangle_type triangle_at(std::size_t idx) const {
            const auto& tris = simplices_.find(2);
            if (tris == simplices_.end() || idx >= tris->second.size()) {
                throw std::out_of_range("SimplicialComplex::triangle_at: index out of range");
            }
            const auto& t = tris->second[idx];
            if (t.size() != 3) {
                throw std::logic_error("SimplicialComplex::triangle_at: stored simplex is not a triangle");
            }
            return { t[0], t[1], t[2] };
        }

        /**
         * @brief Number of triangles.
         */
        std::size_t num_triangles() const noexcept {
            auto it = simplices_.find(2);
            return it == simplices_.end() ? 0 : it->second.size();
        }

        // ---------------------------------------------------------------------
        // Tetrahedron management (3-simplices)
        // ---------------------------------------------------------------------

        /**
         * @brief Add a tetrahedron.
         * @param v0,v1,v2,v3 Vertex indices.
         * @return true if added, false if degenerate, already exists, or vertices invalid.
         */
        bool add_tetrahedron(vertex_index v0, vertex_index v1, vertex_index v2, vertex_index v3) requires (Dim >= 3) {
            // Validate vertices
            if (v0 >= vertices_.size() || v1 >= vertices_.size() ||
                v2 >= vertices_.size() || v3 >= vertices_.size()) {
                return false;
            }
            if (v0 == v1 || v0 == v2 || v0 == v3 || v1 == v2 || v1 == v3 || v2 == v3) {
                return false;  // degenerate
            }
            // Check non-degeneracy (coplanarity)
            if (!is_non_degenerate({ v0, v1, v2, v3 })) {
                return false;
            }
            // Normalize orientation (sort for storage)
            std::vector<vertex_index> tet = { v0, v1, v2, v3 };
            std::sort(tet.begin(), tet.end());
            // Check if already exists
            auto& tet_map = simplices_map_[3];
            if (tet_map.find(tet) != tet_map.end()) {
                return false;
            }
            // Add to storage
            std::size_t idx = simplices_[3].size();
            simplices_[3].push_back(tet);
            tet_map[tet] = idx;
            return true;
        }
        /**
         * @brief Get tetrahedron by index.
         * @param idx Tetrahedron index (0..num_tetrahedra()-1).
         * @return Array of four vertex indices.
         * @throws std::out_of_range if index invalid.
         */
        tetrahedron_type tetrahedron_at(std::size_t idx) const requires (Dim >= 3) {
            const auto& tets = simplices_.find(3);
            if (tets == simplices_.end() || idx >= tets->second.size()) {
                throw std::out_of_range("SimplicialComplex::tetrahedron_at: index out of range");
            }
            const auto& t = tets->second[idx];
            if (t.size() != 4) {
                throw std::logic_error("SimplicialComplex::tetrahedron_at: stored simplex is not a tetrahedron");
            }
            return { t[0], t[1], t[2], t[3] };
        }

        /**
         * @brief Number of tetrahedra.
         */
        std::size_t num_tetrahedra() const noexcept {
            auto it = simplices_.find(3);
            return it == simplices_.end() ? 0 : it->second.size();
        }

        // ---------------------------------------------------------------------
        // General simplex access
        // ---------------------------------------------------------------------

        /**
         * @brief Get simplex by dimension and index.
         * @param dim Dimension of simplex (0=vertex,1=edge,2=triangle,3=tetrahedron).
         * @param idx Index within that dimension.
         * @return Const reference to vector of vertex indices.
         * @throws std::out_of_range if dimension or index invalid.
         */
        const simplex& get_simplex(int dim, std::size_t idx) const {
            if (dim == 0) {
                if (idx >= vertices_.size()) {
                    throw std::out_of_range("SimplicialComplex::get_simplex: vertex index out of range");
                }
                // For dim=0, return a singleton vector (for API consistency)
                static thread_local simplex singleton;
                singleton = { idx };
                return singleton;
            }

            auto it = simplices_.find(dim);
            if (it == simplices_.end() || idx >= it->second.size()) {
                throw std::out_of_range("SimplicialComplex::get_simplex: index out of range");
            }
            return it->second[idx];
        }

        /**
         * @brief Number of simplices of given dimension.
         * @param dim Dimension (0,1,2,3).
         * @return Count.
         */
        std::size_t num_simplices(int dim) const noexcept {
            if (dim == 0) return vertices_.size();
            auto it = simplices_.find(dim);
            return it == simplices_.end() ? 0 : it->second.size();
        }

        /**
         * @brief Find index of a simplex by its vertices.
         * @param dim Dimension of simplex.
         * @param vertices List of vertex indices (order doesn't matter).
         * @return Index if found, -1 otherwise.
         */
        std::ptrdiff_t find_simplex(int dim, const std::vector<vertex_index>& vertices) const {
            if (dim == 0) {
                if (vertices.size() != 1) return -1;
                return vertices[0] < vertices_.size() ? static_cast<std::ptrdiff_t>(vertices[0]) : -1;
            }

            auto it = simplices_map_.find(dim);
            if (it == simplices_map_.end()) return -1;

            std::vector<vertex_index> sorted = vertices;
            std::sort(sorted.begin(), sorted.end());

            auto jt = it->second.find(sorted);
            return jt == it->second.end() ? -1 : static_cast<std::ptrdiff_t>(jt->second);
        }

        // ---------------------------------------------------------------------
        // Incidence relations
        // ---------------------------------------------------------------------

        /**
         * @brief Get faces of codimension 1 incident to a simplex.
         *
         * For a top-dim simplex, returns all (top_dim-1)-simplices that are its faces,
         * with signs following the (-1)^i convention (where i is the omitted vertex index).
         *
         * @param top_dim Dimension of the higher-dimensional simplex.
         * @param idx Index of the higher-dimensional simplex.
         * @param low_dim Dimension of faces to return (must be top_dim - 1).
         * @return Vector of pairs (face_index, sign).
         * @throws std::invalid_argument if low_dim != top_dim - 1.
         */
        std::vector<std::pair<std::size_t, int>> incident_faces(
            int top_dim, std::size_t idx, int low_dim) const {

            if (low_dim != top_dim - 1) {
                throw std::invalid_argument(
                    "SimplicialComplex::incident_faces: only codimension 1 supported");
            }

            const auto& top_simp = get_simplex(top_dim, idx);
            std::vector<std::pair<std::size_t, int>> result;

            for (std::size_t i = 0; i < top_simp.size(); ++i) {
                // Build face by omitting i-th vertex
                std::vector<vertex_index> face_vertices;
                for (std::size_t j = 0; j < top_simp.size(); ++j) {
                    if (j != i) face_vertices.push_back(top_simp[j]);
                }

                std::ptrdiff_t face_idx = find_simplex(low_dim, face_vertices);
                if (face_idx == -1) {
                    throw std::logic_error(
                        "SimplicialComplex::incident_faces: face not found - complex may be inconsistent");
                }

                // Sign follows (-1)^i convention
                int sign = (i % 2 == 0) ? 1 : -1;
                result.emplace_back(static_cast<std::size_t>(face_idx), sign);
            }

            return result;
        }

        // ---------------------------------------------------------------------
        // Geometric queries with metric
        // ---------------------------------------------------------------------

        /**
         * @brief Compute length of an edge using given metric.
         * @tparam Metric Type satisfying Metric concept.
         * @param edge_idx Index of the edge.
         * @param metric Metric object.
         * @return Length (distance between vertices).
         */
        template<typename Metric>
        scalar_type edge_length(std::size_t edge_idx, const Metric& metric) const {
            auto [v0, v1] = edge_at(edge_idx);
            return metric(vertex(v0), vertex(v1));
        }

        /**
         * @brief Compute volume of a cell (triangle in 2D, tetrahedron in 3D).
         * @tparam Metric Type satisfying Metric concept.
         * @param cell_idx Index of the cell.
         * @param metric Metric object (Euclidean expected for area/volume).
         * @return Area or volume.
         */
        template<typename Metric>
        scalar_type cell_volume(std::size_t cell_idx, const Metric& metric) const {
            if constexpr (Dim == 2) {
                auto tri = triangle_at(cell_idx);
                return triangle_volume(
                    vertex(tri[0]), vertex(tri[1]), vertex(tri[2]), metric);
            }
            else if constexpr (Dim == 3) {
                auto tet = tetrahedron_at(cell_idx);
                return tetrahedron_volume(
                    vertex(tet[0]), vertex(tet[1]), vertex(tet[2]), vertex(tet[3]), metric);
            }
            else {
                static_assert(Dim == 2 || Dim == 3,
                    "cell_volume only implemented for 2D and 3D");
                return scalar_type{ 0 };
            }
        }
                /**
          * @brief Compute the volume (measure) of a k-simplex.
          *
          * For k=0: returns 1 (point measure).
          * For k=1: returns edge length.
          * For k=2: returns triangle area.
          * For k=3: returns tetrahedron volume.
          *
          * @tparam Metric Type satisfying Metric concept.
          * @param dim Dimension of the simplex (0..Dim).
          * @param idx Index of the simplex.
          * @param metric Metric object.
          * @return Volume as scalar_type.
          * @throws std::invalid_argument if dim is out of range.
          */
        template<typename Metric>
        scalar_type simplex_volume(int simp_dim, std::size_t idx, const Metric& metric) const {
            if (simp_dim == 0) return scalar_type(1);
            if (simp_dim == 1) {
                auto [v0, v1] = edge_at(idx);
                return metric(vertex(v0), vertex(v1));
            }
            if (simp_dim == 2) {
                auto tri = triangle_at(idx);
                return triangle_volume(vertex(tri[0]), vertex(tri[1]), vertex(tri[2]), metric);
            }
            if (simp_dim == 3) {
                if constexpr (Dim >= 3) {  // Dim — это Dimension комплекса
                    auto tet = tetrahedron_at(idx);
                    return tetrahedron_volume(vertex(tet[0]), vertex(tet[1]), vertex(tet[2]), vertex(tet[3]), metric);
                }
                else {
                    throw std::invalid_argument("simplex_volume: dimension 3 not supported in complex of dimension " + std::to_string(Dim));
                }
            }
            throw std::invalid_argument("simplex_volume: unsupported simplex dimension");
        }
        /**
         * @brief Compute outward normal for an edge in 2D.
         */
        template<typename Metric>
        point_type edge_normal_2d(std::size_t edge_idx, const Metric& metric) const requires (Dim == 2) {
            auto [v0, v1] = edge_at(edge_idx);
            point_type e = (vertex(v1) - vertex(v0)).data();
            point_type n;
            n << e[1], -e[0];
            scalar_type eucl_len = e.norm();
            if (eucl_len > 0) {
                scalar_type met_len = metric(vertex(v0), vertex(v1));
                n *= (met_len / eucl_len);
            }
            return n;
        }
        /**
         * @brief Get neighboring triangles of an edge in 2D.
         *
         * @param edge_idx Index of the edge.
         * @return Pair (left_triangle_index, optional_right_triangle_index).
         *         For boundary edges, right is std::nullopt.
         */
        std::pair<std::size_t, std::optional<std::size_t>> edge_neighbors_2d(
            std::size_t edge_idx) const requires (Dim == 2) {
            ensure_edge_to_triangles();
            const auto& entry = (*edge_to_triangles_)[edge_idx];
            return entry;
        }

        // ---------------------------------------------------------------------
        // Barycentric subdivision
        // ---------------------------------------------------------------------

        /**
         * @brief Perform barycentric subdivision of the complex.
         *
         * Creates a new, finer complex by subdividing each simplex at its
         * barycenter. Returns the new complex and a map from original simplices
         * to the set of simplices in the subdivided complex that cover them.
         *
         * @return Pair (subdivided_complex, subdivision_map).
         */
        std::pair<SimplicialComplex, SubdivisionMap> barycentric_subdivide() const {
            SimplicialComplex fine;
            SubdivisionMap subdiv_map;

            // Map from original vertices to their indices in fine complex
            std::unordered_map<vertex_index, vertex_index> vertex_map;

            // Step 1: Copy all original vertices to fine complex
            for (std::size_t i = 0; i < vertices_.size(); ++i) {
                vertex_map[i] = fine.add_vertex(vertices_[i]);
            }

            // Step 2: For each edge, add its midpoint and record mapping
            std::unordered_map<std::size_t, vertex_index> edge_midpoints;
            for (std::size_t e = 0; e < num_edges(); ++e) {
                auto [v0, v1] = edge_at(e);
                point_type mid = (vertex(v0) + vertex(v1)) / 2_r;
                vertex_index mid_idx = fine.add_vertex(mid);
                edge_midpoints[e] = mid_idx;

                // Record in subdivision map: original edge -> two new edges
                SimplexKey orig_key{ 1, e };
                subdiv_map[orig_key].push_back(SimplexKey{ 1, fine.num_edges() });
                subdiv_map[orig_key].push_back(SimplexKey{ 1, fine.num_edges() + 1 });

                // Add the two half-edges to fine complex
                fine.add_edge(vertex_map[v0], mid_idx);
                fine.add_edge(mid_idx, vertex_map[v1]);
            }

            // Step 3: For each triangle, add centroid and subdivide
            std::unordered_map<std::size_t, vertex_index> triangle_centroids;
            for (std::size_t t = 0; t < num_triangles(); ++t) {
                auto [v0, v1, v2] = triangle_at(t);
                point_type centroid = (vertex(v0) + vertex(v1) + vertex(v2)) / 3_r;
                vertex_index c_idx = fine.add_vertex(centroid);
                triangle_centroids[t] = c_idx;

                // Find edge midpoints for this triangle's edges
                auto e01_idx = find_simplex(1, { v0, v1 });
                auto e12_idx = find_simplex(1, { v1, v2 });
                auto e20_idx = find_simplex(1, { v2, v0 });

                if (e01_idx == -1 || e12_idx == -1 || e20_idx == -1) {
                    throw std::logic_error(
                        "SimplicialComplex::barycentric_subdivide: missing edges for triangle");
                }

                vertex_index m01 = edge_midpoints[e01_idx];
                vertex_index m12 = edge_midpoints[e12_idx];
                vertex_index m20 = edge_midpoints[e20_idx];

                // Record in subdivision map: original triangle -> 6 new triangles
                for (int i = 0; i < 6; ++i) {
                    subdiv_map[SimplexKey{ 2, t }].push_back(SimplexKey{ 2, fine.num_triangles() + i });
                }

                // Add the 6 small triangles (order matters for orientation)
                // Triangle (v0, m01, c)
                fine.add_triangle(vertex_map[v0], m01, c_idx);
                // Triangle (v0, c, m20)
                fine.add_triangle(vertex_map[v0], c_idx, m20);
                // Triangle (v1, m12, c)
                fine.add_triangle(vertex_map[v1], m12, c_idx);
                // Triangle (v1, c, m01)
                fine.add_triangle(vertex_map[v1], c_idx, m01);
                // Triangle (v2, m20, c)
                fine.add_triangle(vertex_map[v2], m20, c_idx);
                // Triangle (v2, c, m12)
                fine.add_triangle(vertex_map[v2], c_idx, m12);
            }
            // Add edges from centroid to vertices and midpoints for each triangle
            for (std::size_t t = 0; t < num_triangles(); ++t) {
                auto [v0, v1, v2] = triangle_at(t);
                vertex_index c_idx = triangle_centroids[t];
                auto e01_idx = find_simplex(1, { v0, v1 });
                auto e12_idx = find_simplex(1, { v1, v2 });
                auto e20_idx = find_simplex(1, { v2, v0 });
                // Проверка, что индексы рёбер найдены (должны быть, так как мы их добавляли ранее)
                if (e01_idx == -1 || e12_idx == -1 || e20_idx == -1) {
                    throw std::logic_error("Missing edges in barycentric subdivision");
                }
                vertex_index m01 = edge_midpoints[e01_idx];
                vertex_index m12 = edge_midpoints[e12_idx];
                vertex_index m20 = edge_midpoints[e20_idx];

                fine.add_edge(vertex_map[v0], c_idx);
                fine.add_edge(vertex_map[v1], c_idx);
                fine.add_edge(vertex_map[v2], c_idx);
                fine.add_edge(m01, c_idx);
                fine.add_edge(m12, c_idx);
                fine.add_edge(m20, c_idx);
            }
            // Step 4: For 3D, handle tetrahedra (if Dim >= 3)
            if constexpr (Dim >= 3) {
                // Similar logic for tetrahedra would go here
                // For Stage 0, we only need triangles
            }

            return { std::move(fine), std::move(subdiv_map) };
        }

        // ---------------------------------------------------------------------
        // OrderedGrid concept requirements
        // ---------------------------------------------------------------------

        /**
         * @brief Number of vertices (for OrderedGrid concept).
         */
        std::size_t size() const noexcept {
            return vertices_.size();
        }

        /**
         * @brief Access vertex by index (for OrderedGrid concept).
         */
        const point_type& operator[](std::size_t idx) const noexcept {
            // No bounds check for performance, but vertex() provides checked access
            return vertices_[idx];
        }

        /**
         * @brief Begin iterator over vertices.
         */
        const_iterator begin() const noexcept {
            return vertices_.begin();
        }

        /**
         * @brief End iterator over vertices.
         */
        const_iterator end() const noexcept {
            return vertices_.end();
        }

        /**
         * @brief Comparator for vertices (required by OrderedGrid concept).
         */
        comparator_type comparator() const noexcept {
            return comparator_type{};
        }

    private:
        // ---------------------------------------------------------------------
        // Private helper methods
        // ---------------------------------------------------------------------

        /**
         * @brief Check if a simplex is non-degenerate.
         *
         * For a triangle: checks that points are not collinear (area > 0).
         * For a tetrahedron: checks that points are not coplanar (volume > 0).
         */
        bool is_non_degenerate(const std::vector<vertex_index>& indices) const {
            if (indices.size() == 2) {
                return true;
            }
            else if (indices.size() == 3) {
                const auto& a = vertex(indices[0]);
                const auto& b = vertex(indices[1]);
                const auto& c = vertex(indices[2]);

                if constexpr (Dim >= 2) {
                    auto ab = b - a;
                    auto ac = c - a;

                    if constexpr (Dim == 2) {
                        scalar_type cross = ab.data().x() * ac.data().y() - ab.data().y() * ac.data().x();
                        return cross != 0_r;
                    }
                    else {
                        scalar_type cross_xy = ab.data().x() * ac.data().y() - ab.data().y() * ac.data().x();
                        if (cross_xy != 0_r) return true;
                        scalar_type cross_xz = ab.data().x() * ac.data().z() - ab.data().z() * ac.data().x();
                        if (cross_xz != 0_r) return true;
                        scalar_type cross_yz = ab.data().y() * ac.data().z() - ab.data().z() * ac.data().y();
                        return cross_yz != 0_r;
                    }
                }
                return true;
            }
            else if (indices.size() == 4) {
                // Тетраэдр возможен только при Dim >= 3
                if constexpr (Dim >= 3) {
                    const auto& a = vertex(indices[0]);
                    const auto& b = vertex(indices[1]);
                    const auto& c = vertex(indices[2]);
                    const auto& d = vertex(indices[3]);

                    auto ab = b - a;
                    auto ac = c - a;
                    auto ad = d - a;

                    if constexpr (Dim == 3) {
                        scalar_type vol = delta::abs(ab.data().dot(ac.data().cross(ad.data())));
                        return vol != 0_r;
                    }
                    else {
                        Eigen::Matrix<scalar_type, 3, 1> ab3(ab.data().x(), ab.data().y(), ab.data().z());
                        Eigen::Matrix<scalar_type, 3, 1> ac3(ac.data().x(), ac.data().y(), ac.data().z());
                        Eigen::Matrix<scalar_type, 3, 1> ad3(ad.data().x(), ad.data().y(), ad.data().z());
                        scalar_type vol = delta::abs(ab3.dot(ac3.cross(ad3)));
                        return vol != 0_r;
                    }
                }
                else {
                    // Для Dim < 3 тетраэдры не поддерживаются, но эта ветка не должна достигаться.
                    return false;
                }
            }
            return true;
        }
        /**
         * @brief Compute triangle area using Heron's formula (for Euclidean metric).
         */
        template<typename Metric>
        scalar_type triangle_volume(const point_type& a,
            const point_type& b,
            const point_type& c,
            const Metric& metric) const {
            auto ab = metric(a, b);
            auto bc = metric(b, c);
            auto ca = metric(c, a);
            auto s = (ab + bc + ca) / 2_r;

            // Heron's formula: sqrt(s * (s-ab) * (s-bc) * (s-ca))
            using delta::sqrt;
            auto prod = s * (s - ab) * (s - bc) * (s - ca);
            // Ensure non-negative due to rounding
            if (prod < 0_r) prod = 0_r;
            return sqrt(prod);
        }

        /**
         * @brief Compute tetrahedron volume 
         */
        template<typename Metric>
        scalar_type tetrahedron_volume(const point_type& a,
            const point_type& b,
            const point_type& c,
            const point_type& d,
            const Metric& /*metric*/) const {
            static_assert(Dim == 3, "Tetrahedron volume only for 3D");
            // Используем смешанное произведение векторов рёбер
            auto ab = (b - a).data();
            auto ac = (c - a).data();
            auto ad = (d - a).data();
            scalar_type vol = delta::abs(ab.cross(ac).dot(ad)) / 6;
            return vol;
        }

        /**
         * @brief Build edge-to-triangle adjacency map for 2D.
         */
        void ensure_edge_to_triangles() const {
            if (edge_to_triangles_.has_value()) return;

            std::vector<std::pair<std::size_t, std::optional<std::size_t>>> result(num_edges());

            // Initialize all edges as boundary (no right neighbor)
            for (std::size_t e = 0; e < num_edges(); ++e) {
                result[e] = { static_cast<std::size_t>(-1), std::nullopt };
            }

            // For each triangle, record its edges with orientation
            for (std::size_t t = 0; t < num_triangles(); ++t) {
                auto tri = triangle_at(t);
                // Edges in order: (v0,v1), (v1,v2), (v2,v0)
                std::array<std::pair<vertex_index, vertex_index>, 3> edges = { {
                    {tri[0], tri[1]},
                    {tri[1], tri[2]},
                    {tri[2], tri[0]}
                } };

                for (const auto& [v0, v1] : edges) {
                    auto e_idx = find_simplex(1, { v0, v1 });
                    if (e_idx == -1) continue;

                    // For each edge, we want left triangle to be the one
                    // where the edge orientation matches triangle orientation.
                    // For simplicity, we just store triangles in order of discovery:
                    // first triangle becomes left, second becomes right.
                    if (result[e_idx].first == static_cast<std::size_t>(-1)) {
                        result[e_idx].first = t;
                    }
                    else {
                        result[e_idx].second = t;
                    }
                }
            }

            edge_to_triangles_ = std::move(result);
        }

        // ---------------------------------------------------------------------
        // Member variables
        // ---------------------------------------------------------------------
        std::vector<point_type> vertices_;

        // simplices_[dim] = list of simplices of that dimension
        std::unordered_map<int, std::vector<simplex>> simplices_;

        // simplices_map_[dim][sorted_vertices] = index
        std::unordered_map<int, std::unordered_map<simplex, std::size_t, SimplexHasher>> simplices_map_;

        // Cache for edge neighbors (2D only)
        mutable std::optional<std::vector<std::pair<std::size_t, std::optional<std::size_t>>>>
            edge_to_triangles_;
    };

    // -------------------------------------------------------------------------
    // Concept checks
    // -------------------------------------------------------------------------

    // Verify that SimplicialComplex satisfies OrderedGrid concept
    static_assert(delta::OrderedGrid<SimplicialComplex<2>>,
        "SimplicialComplex<2> must satisfy OrderedGrid concept");
    static_assert(delta::OrderedGrid<SimplicialComplex<3>>,
        "SimplicialComplex<3> must satisfy OrderedGrid concept");

} // namespace delta::geometry