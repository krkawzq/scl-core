// =============================================================================
// FILE: scl/kernel/merge.h
// BRIEF: API reference for matrix merging kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::merge {

/* -----------------------------------------------------------------------------
 * FUNCTION: vstack
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Vertically stack two sparse matrices (concatenate along primary axis).
 *
 * PARAMETERS:
 *     matrix1  [in]  First sparse matrix
 *     matrix2  [in]  Second sparse matrix
 *     strategy [in]  Block allocation strategy for result
 *
 * PRECONDITIONS:
 *     - For CSR: columns can differ (result uses max)
 *     - For CSC: rows can differ (result uses max)
 *
 * POSTCONDITIONS:
 *     - Result primary_dim = matrix1.primary_dim + matrix2.primary_dim
 *     - Result secondary_dim = max(matrix1.secondary_dim, matrix2.secondary_dim)
 *     - Rows 0..n1-1 from matrix1, rows n1..n1+n2-1 from matrix2
 *     - Indices unchanged (secondary dimension preserved)
 *
 * RETURNS:
 *     New sparse matrix with vertically stacked data
 *
 * ALGORITHM:
 *     1. Compute row lengths for result
 *     2. Allocate result matrix with combined structure
 *     3. Parallel copy matrix1 rows to result[0:n1]
 *     4. Parallel copy matrix2 rows to result[n1:n1+n2]
 *
 * COMPLEXITY:
 *     Time:  O(nnz1 + nnz2)
 *     Space: O(nnz1 + nnz2) for result
 *
 * THREAD SAFETY:
 *     Safe - parallel copy of independent regions
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Sparse<T, IsCSR> vstack(
    const Sparse<T, IsCSR>& matrix1,             // First matrix
    const Sparse<T, IsCSR>& matrix2,             // Second matrix
    BlockStrategy strategy = BlockStrategy::adaptive()  // Allocation strategy
);

/* -----------------------------------------------------------------------------
 * FUNCTION: hstack
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Horizontally stack two sparse matrices (concatenate along secondary axis).
 *
 * PARAMETERS:
 *     matrix1  [in]  First sparse matrix
 *     matrix2  [in]  Second sparse matrix
 *     strategy [in]  Block allocation strategy for result
 *
 * PRECONDITIONS:
 *     - matrix1.primary_dim == matrix2.primary_dim
 *
 * POSTCONDITIONS:
 *     - Result primary_dim = matrix1.primary_dim (unchanged)
 *     - Result secondary_dim = matrix1.secondary_dim + matrix2.secondary_dim
 *     - For each row: [matrix1 columns | matrix2 columns with offset]
 *     - matrix2 indices offset by matrix1.secondary_dim
 *
 * RETURNS:
 *     New sparse matrix with horizontally stacked data
 *
 * ALGORITHM:
 *     1. Verify primary dimensions match
 *     2. Compute combined row lengths
 *     3. Allocate result matrix
 *     4. Parallel over rows:
 *        - Copy matrix1 values and indices
 *        - Copy matrix2 values
 *        - Add offset to matrix2 indices (SIMD optimized)
 *
 * COMPLEXITY:
 *     Time:  O(nnz1 + nnz2)
 *     Space: O(nnz1 + nnz2) for result
 *
 * THREAD SAFETY:
 *     Safe - parallel over independent rows
 *
 * THROWS:
 *     DimensionError - if primary dimensions mismatch
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Sparse<T, IsCSR> hstack(
    const Sparse<T, IsCSR>& matrix1,             // First matrix
    const Sparse<T, IsCSR>& matrix2,             // Second matrix
    BlockStrategy strategy = BlockStrategy::adaptive()  // Allocation strategy
);

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::add_offset_simd
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Add constant offset to index array with SIMD optimization.
 *
 * PARAMETERS:
 *     src    [in]  Source index array
 *     dst    [out] Destination index array
 *     count  [in]  Number of elements
 *     offset [in]  Value to add to each index
 *
 * ALGORITHM:
 *     - offset == 0: direct memcpy (early exit)
 *     - 2-way SIMD unrolled loop for bulk processing
 *     - Scalar cleanup for remainder
 *
 * COMPLEXITY:
 *     Time:  O(count)
 *     Space: O(1)
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::parallel_memcpy
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Parallel memory copy for large data blocks.
 *
 * PARAMETERS:
 *     src        [in]  Source pointer
 *     dst        [out] Destination pointer
 *     count      [in]  Number of elements
 *     chunk_size [in]  Elements per parallel chunk (default: 65536)
 *
 * ALGORITHM:
 *     - count < chunk_size: single memcpy
 *     - Otherwise: parallel_for over chunks with prefetch
 *
 * COMPLEXITY:
 *     Time:  O(count / n_threads)
 *     Space: O(1)
 * -------------------------------------------------------------------------- */

} // namespace scl::kernel::merge
