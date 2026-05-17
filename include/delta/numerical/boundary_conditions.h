// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/numerical/boundary_conditions.h
// ============================================================================
// IMPLEMENTATION NOTES – BOUNDARY CONDITION APPLICATION
// ============================================================================
//
// This header provides a generic routine to modify a sparse linear system
// A·u = b according to user‑defined boundary conditions.
//
// --------------------------------------------------------------------------
// IMPORTANT DESIGN DECISIONS
// --------------------------------------------------------------------------
// 1. RowMajor storage
//    The function internally converts the matrix to RowMajor format before
//    any modification. This allows efficient iteration over rows using
//    Eigen's InnerIterator without crossing major boundaries. After all
//    transformations, the result is written back to the caller's matrix
//    (which may be ColMajor). The temporary RowMajor copy has negligible
//    overhead for the sizes we target.
//
// 2. Dirichlet handling
//    - Column k (k ≠ i) is zeroed before row i is cleared, because
//      InnerIterator on row i might access coefficients that still have a
//      non‑zero column counterpart; zeroing the column first guarantees
//      symmetry (A(i,k) and A(k,i) both become zero).
//    - The right‑hand side correction b(k) -= A(k,i)·u_i is performed
//      while A(k,i) is still valid.
//    - Only after all columns are processed we clear row i and set
//      A(i,i)=1, b(i)=u_i.  This order avoids any dangling references.
//
// 3. Periodic merging
//    The algorithm:
//      a) Collect row i and row j into associative containers (row_i, row_j).
//      b) Build a merged row i = row_i + row_j.
//      c) Combine the diagonal entries: the coefficient of u_i becomes
//         (A(i,i)+A(j,i)) + (A(i,j)+A(j,j)) = new_row_i[i] + new_row_i[j].
//         This is the only mathematically correct way to produce a symmetric
//         merged equation.
//      d) After writing new row i, add off‑diagonal entries of column j to
//         column i (for symmetry).  This is done via a separate pass that
//         reads A(k,j) for k≠i,j before clearing columns.
//      e) Row j is replaced by the constraint u_j − u_i = 0, i.e.
//         A(j,j)=1, A(j,i)=-1, b(j)=0. Column j is zeroed everywhere else.
//    This method preserves symmetry and ensures that the solution satisfies
//    u_j = u_i exactly.
//
// 4. Robin conditions
//    Only the diagonal and the right‑hand side are touched:
//      A(i,i) += a·M(i)   (a is the coefficient of u)
//      b(i)   += g·M(i)   (g is the source)
//    The term b·∂u/∂n is assumed to be already present in the assembled
//    matrix (typically through integration over the boundary).  If it is
//    not, the caller must incorporate it before invoking this function.
//
// 5. Static casts
//    Eigen uses `int` as the default index type for sparse matrices.
//    We use `std::size_t` for DOF counts here; all conversions are
//    guaranteed to be safe for the problem sizes we support (typically
//    ≪ INT_MAX).  Explicit `static_cast<int>` avoids compiler warnings
//    about sign/size mismatch.
//
// --------------------------------------------------------------------------
// PRESERVATION OF SYMMETRY
// --------------------------------------------------------------------------
// The function is designed to preserve symmetry of A (if the original system
// is symmetric) for Dirichlet, Neumann, Robin, and Periodic boundary
// conditions.  After a Dirichlet application the matrix remains symmetric
// (rows/columns i are decoupled).  Periodic merging is carried out in a
// symmetric fashion (combined row/column sums).  If symmetry is not
// required, the implementation still works correctly.
//
// --------------------------------------------------------------------------
// FUTURE MODIFICATIONS
// --------------------------------------------------------------------------
//   - Adding time‑ or parameter‑dependent boundary values (already supported
//     via the `t` and `dof_map` arguments, which are currently unused).
//   - Extending Robin to fully support the b·∂u/∂n term when it is not
//     already embedded in A.
//   - Optimising periodic merging by directly manipulating Eigen's internal
//     structures if performance becomes critical.
// When making such changes, run the full boundary_conditions_test suite
// (which is the reference for correctness) and ensure all tests pass.
//
// ============================================================================
#pragma once

#include <unordered_map>
#include <vector>
#include <utility>
#include <set>
#include <map>
#include <Eigen/Sparse>
#include "delta/rational/eigen_integration.h"

namespace delta::numerical {

    enum class BCType {
        Dirichlet,
        Neumann,
        Robin,
        Periodic
    };

    template<typename Value>
    class BoundaryConditions {
    public:
        void set(std::size_t dof, BCType type, Value value = Value{}) {
            conditions_[dof] = { type, value };
        }

        void set_robin(std::size_t dof, Value a, Value b, Value g) {
            conditions_[dof] = { BCType::Robin, a };
            robin_params_[dof] = { a, b, g };
        }

        void add_periodic_pair(std::size_t dof1, std::size_t dof2) {
            periodic_pairs_.emplace_back(dof1, dof2);
        }

        bool has(std::size_t dof) const {
            return conditions_.find(dof) != conditions_.end();
        }

        BCType type(std::size_t dof) const {
            return conditions_.at(dof).first;
        }

        Value value(std::size_t dof) const {
            return conditions_.at(dof).second;
        }

        bool has_robin(std::size_t dof) const {
            return robin_params_.find(dof) != robin_params_.end();
        }

        Value robin_a(std::size_t dof) const { return robin_params_.at(dof).a; }
        Value robin_b(std::size_t dof) const { return robin_params_.at(dof).b; }
        Value robin_g(std::size_t dof) const { return robin_params_.at(dof).g; }

        const auto& all() const { return conditions_; }
        const auto& periodic_pairs() const { return periodic_pairs_; }

    private:
        struct RobinParams { Value a, b, g; };
        std::unordered_map<std::size_t, std::pair<BCType, Value>> conditions_;
        std::unordered_map<std::size_t, RobinParams> robin_params_;
        std::vector<std::pair<std::size_t, std::size_t>> periodic_pairs_;
    };

    //P.S. - что это ещё за double t = и накой чёрт оно надо?
    template<typename Value>
    void apply_boundary_conditions(
        Eigen::SparseMatrix<Value>& A,
        Eigen::Matrix<Value, Eigen::Dynamic, 1>& b,
        const Eigen::Matrix<Value, Eigen::Dynamic, 1>& lumpedM,
        std::size_t n,
        BoundaryConditions<Value>& bc,
        double /*t*/=0,
        const std::vector<std::size_t> & /*dof_map*/ = {})
    {
        // Work in RowMajor to allow simple row operations via InnerIterator
        Eigen::SparseMatrix<Value, Eigen::RowMajor> A_rm = A;

        // First, apply Dirichlet, Neumann, Robin
        for (std::size_t i = 0; i < n; ++i) {
            if (!bc.has(i)) continue;

            if (bc.type(i) == BCType::Dirichlet) {
                Value u_i = bc.value(i);

                // Subtract Dirichlet contribution from RHS and zero column i (except row i)
                for (std::size_t k = 0; k < n; ++k) {
                    if (k == i) continue;
                    Value A_ki = A_rm.coeff(k, i);
                    if (A_ki != Value(0)) {
                        b(k) -= A_ki * u_i;
                        A_rm.coeffRef(k, i) = Value(0);
                    }
                }

                // Zero row i (now InnerIterator goes over the row)
                std::vector<int> cols;
                for (typename Eigen::SparseMatrix<Value, Eigen::RowMajor>::InnerIterator it(A_rm, static_cast<int>(i)); it; ++it)
                    cols.push_back(it.col());
                for (int col : cols)
                    A_rm.coeffRef(static_cast<int>(i), col) = Value(0);

                A_rm.coeffRef(static_cast<int>(i), static_cast<int>(i)) = Value(1);
                b(i) = u_i;
            }
            else if (bc.type(i) == BCType::Neumann) {
                b(i) += lumpedM(i) * bc.value(i);
            }
            else if (bc.type(i) == BCType::Robin) {
                Value a = bc.robin_a(i);
                Value g = bc.robin_g(i);
                A_rm.coeffRef(static_cast<int>(i), static_cast<int>(i)) += a * lumpedM(i);
                b(i) += g * lumpedM(i);
            }
        }

        // Then, apply periodic conditions by merging DOF pairs
        for (const auto& [i, j] : bc.periodic_pairs()) {
            if (i >= n || j >= n) continue;

            // ----- 1. Collect the rows i and j -----
            std::map<int, Value> row_i, row_j;
            for (typename Eigen::SparseMatrix<Value, Eigen::RowMajor>::InnerIterator it(A_rm, static_cast<int>(i)); it; ++it)
                row_i[it.col()] = it.value();
            for (typename Eigen::SparseMatrix<Value, Eigen::RowMajor>::InnerIterator it(A_rm, static_cast<int>(j)); it; ++it)
                row_j[it.col()] = it.value();

            // ----- 2. Build the new row i: sum of rows i and j -----
            std::map<int, Value> new_row_i;
            std::set<int> cols;
            for (auto& p : row_i) cols.insert(p.first);
            for (auto& p : row_j) cols.insert(p.first);
            for (int col : cols) {
                Value val = Value(0);
                auto it_i = row_i.find(col);
                if (it_i != row_i.end()) val += it_i->second;
                auto it_j = row_j.find(col);
                if (it_j != row_j.end()) val += it_j->second;
                if (val != Value(0))
                    new_row_i[col] = val;
            }

            // ----- 3. Combine the contributions of u_i and u_j into the diagonal of u_i -----
            // After substitution u_j = u_i, the coefficient at u_i is new_row_i[i] + new_row_i[j].
            // We move it all to column i and remove entry at column j.
            Value combined_diag = Value(0);
            auto it_i_diag = new_row_i.find(static_cast<int>(i));
            if (it_i_diag != new_row_i.end()) {
                combined_diag += it_i_diag->second;
                new_row_i.erase(it_i_diag);
            }
            auto it_j_diag = new_row_i.find(static_cast<int>(j));
            if (it_j_diag != new_row_i.end()) {
                combined_diag += it_j_diag->second;
                new_row_i.erase(it_j_diag);
            }
            if (combined_diag != Value(0))
                new_row_i[static_cast<int>(i)] = combined_diag;

            // ----- 4. For symmetry, add column j to column i for all off-diagonal rows -----
            // We collect current off-diagonal entries in column j and add them to column i.
            std::vector<std::pair<int, Value>> col_j_offdiag;
            for (int k = 0; k < static_cast<int>(n); ++k) {
                if (k == static_cast<int>(i) || k == static_cast<int>(j)) continue;
                Value v = A_rm.coeff(k, j);
                if (v != Value(0)) col_j_offdiag.emplace_back(k, v);
            }
            // No need to collect column i off-diagonals because we will rewrite row i

            // ----- 5. Clear row i and row j -----
            std::vector<int> cols_i, cols_j;
            for (typename Eigen::SparseMatrix<Value, Eigen::RowMajor>::InnerIterator it(A_rm, static_cast<int>(i)); it; ++it)
                cols_i.push_back(it.col());
            for (int c : cols_i) A_rm.coeffRef(static_cast<int>(i), c) = Value(0);
            for (typename Eigen::SparseMatrix<Value, Eigen::RowMajor>::InnerIterator it(A_rm, static_cast<int>(j)); it; ++it)
                cols_j.push_back(it.col());
            for (int c : cols_j) A_rm.coeffRef(static_cast<int>(j), c) = Value(0);

            // ----- 6. Write the new row i -----
            for (auto& p : new_row_i)
                A_rm.coeffRef(static_cast<int>(i), p.first) = p.second;

            // ----- 7. Update off-diagonal column i entries -----
            for (auto& p : col_j_offdiag) {
                A_rm.coeffRef(p.first, static_cast<int>(i)) += p.second;
            }

            // ----- 8. Clear column j entirely (except for the constraint row j) -----
            for (int k = 0; k < static_cast<int>(n); ++k) {
                if (k == static_cast<int>(j)) continue;
                A_rm.coeffRef(k, static_cast<int>(j)) = Value(0);
            }

            // ----- 9. Set row j as the constraint u_j - u_i = 0 -----
            A_rm.coeffRef(static_cast<int>(j), static_cast<int>(j)) = Value(1);
            A_rm.coeffRef(static_cast<int>(j), static_cast<int>(i)) = Value(-1);

            // ----- 10. Update RHS -----
            b(i) += b(j);
            b(j) = Value(0);
        }

        // Write back the result
        A = A_rm;
    }

} // namespace delta::numerical