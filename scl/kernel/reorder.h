// =============================================================================
// FILE: scl/kernel/reorder.h
// BRIEF: API reference for matrix reordering and permutation operations
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::reorder {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size PARALLEL_THRESHOLD = 256;
}

// =============================================================================
// Reordering Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: reorder_rows
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Reorder rows of sparse matrix according to permutation.
 *
 * PARAMETERS:
 *     matrix       [in]  Input sparse matrix (CSR)
 *     permutation  [in]  Row permutation [n_rows]
 *     n_rows       [in]  Number of rows
 *     output       [out] Reordered matrix (CSR)
 *
 * PRECONDITIONS:
 *     - permutation is valid permutation
 *
 * POSTCONDITIONS:
 *     - output[i] contains row permutation[i] from input
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(nnz) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void reorder_rows(
    const Sparse<T, IsCSR>& matrix,         // Input matrix [n_rows x n_cols]
    Array<const Index> permutation,          // Row permutation [n_rows]
    Index n_rows,                            // Number of rows
    Sparse<T, IsCSR>& output                 // Output reordered matrix
);

/* -----------------------------------------------------------------------------
 * FUNCTION: reorder_columns
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Reorder columns of sparse matrix according to permutation.
 *
 * PARAMETERS:
 *     matrix       [in]  Input sparse matrix (CSR or CSC)
 *     permutation  [in]  Column permutation [n_cols]
 *     n_cols       [in]  Number of columns
 *     output       [out] Reordered matrix
 *
 * PRECONDITIONS:
 *     - permutation is valid permutation
 *
 * POSTCONDITIONS:
 *     - output has columns in permuted order
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(nnz) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void reorder_columns(
    const Sparse<T, IsCSR>& matrix,         // Input matrix [n_rows x n_cols]
    Array<const Index> permutation,          // Column permutation [n_cols]
    Index n_cols,                            // Number of columns
    Sparse<T, IsCSR>& output                 // Output reordered matrix
);

} // namespace scl::kernel::reorder

