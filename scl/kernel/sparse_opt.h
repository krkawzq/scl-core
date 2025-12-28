// =============================================================================
// FILE: scl/kernel/sparse_opt.h
// BRIEF: API reference for sparse matrix optimization operations
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::sparse_opt {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real EPSILON = Real(1e-15);
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Size SIMD_THRESHOLD = 16;
}

// =============================================================================
// Sparse Optimization Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: sparse_least_squares
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Solve sparse least squares problem: min ||Ax - b||^2
 *
 * PARAMETERS:
 *     A            [in]  Sparse matrix (CSR)
 *     b            [in]  Right-hand side vector [n_rows]
 *     n_rows       [in]  Number of rows
 *     n_cols       [in]  Number of columns
 *     x            [out] Solution vector [n_cols]
 *     max_iter     [in]  Maximum iterations
 *     tol          [in]  Convergence tolerance
 *
 * PRECONDITIONS:
 *     - x has capacity >= n_cols
 *
 * POSTCONDITIONS:
 *     - x contains approximate solution
 *
 * COMPLEXITY:
 *     Time:  O(max_iter * nnz)
 *     Space: O(n_cols) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized SpMV
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void sparse_least_squares(
    const Sparse<T, IsCSR>& A,              // Sparse matrix [n_rows x n_cols]
    const Real* b,                            // Right-hand side [n_rows]
    Index n_rows,                             // Number of rows
    Index n_cols,                             // Number of columns
    Array<Real> x,                            // Output solution [n_cols]
    Index max_iter = 100,                     // Max iterations
    Real tol = Real(1e-6)                     // Convergence tolerance
);

} // namespace scl::kernel::sparse_opt

