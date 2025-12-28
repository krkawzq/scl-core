// =============================================================================
// FILE: scl/kernel/gram.h
// BRIEF: API reference for Gram matrix computation kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::gram {

/* -----------------------------------------------------------------------------
 * CONFIGURATION CONSTANTS
 * -----------------------------------------------------------------------------
 * PREFETCH_DISTANCE  - Elements to prefetch ahead in merge loop (32)
 * RATIO_THRESHOLD    - Size ratio threshold for binary search (32)
 * GALLOP_THRESHOLD   - Size ratio threshold for galloping search (256)
 * CHUNK_SIZE         - Row chunk size for parallelization (64)
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * FUNCTION: gram
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Gram matrix (inner product matrix) for sparse matrix rows.
 *     G[i,j] = dot(row_i, row_j)
 *
 * PARAMETERS:
 *     matrix [in]  Sparse matrix, shape (n_rows, n_cols)
 *     output [out] Gram matrix, size = n_rows * n_rows
 *
 * PRECONDITIONS:
 *     - output.len >= n_rows^2
 *
 * POSTCONDITIONS:
 *     - output is symmetric: output[i,j] == output[j,i]
 *     - Diagonal: output[i,i] = squared L2 norm of row i
 *     - output[i,j] = dot(row_i, row_j)
 *
 * OUTPUT LAYOUT:
 *     Row-major: output[i * n_rows + j] = dot(row_i, row_j)
 *
 * ALGORITHM:
 *     Key optimizations:
 *     1. Symmetric computation: only compute upper triangle
 *     2. Diagonal via vectorize::sum_squared (SIMD optimized)
 *     3. Adaptive sparse dot product selection based on size ratio:
 *        - ratio < 32:   Linear merge with 8/4-way skip
 *        - ratio < 256:  Binary search with range narrowing
 *        - ratio >= 256: Galloping (exponential) search
 *     4. O(1) range disjointness check before any computation
 *
 * COMPLEXITY:
 *     Time:  O(n_rows^2 * avg_nnz_per_row / n_threads)
 *     Space: O(1) beyond output
 *
 * THREAD SAFETY:
 *     Safe - parallel over rows with symmetric write
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void gram(
    const Sparse<T, IsCSR>& matrix,  // Input sparse matrix [n_rows x n_cols]
    Array<T> output                   // Output Gram matrix [n_rows x n_rows]
);

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::sparse_dot_adaptive
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute dot product of two sparse vectors with adaptive algorithm.
 *
 * PARAMETERS:
 *     idx1, val1, n1 - First sparse vector (indices, values, length)
 *     idx2, val2, n2 - Second sparse vector (indices, values, length)
 *
 * ALGORITHM SELECTION:
 *     Let n_small = min(n1, n2), n_large = max(n1, n2), ratio = n_large/n_small
 *
 *     1. ratio < 32 (similar sizes):
 *        dot_linear_branchless - Linear merge with skip optimization
 *
 *     2. 32 <= ratio < 256 (moderately different):
 *        dot_binary - Binary search in large array for each small element
 *
 *     3. ratio >= 256 (very different sizes):
 *        dot_gallop - Galloping search in large array
 *
 * EARLY EXIT:
 *     - Empty vectors: return 0
 *     - Range disjoint (max(idx1) < min(idx2) or vice versa): return 0
 *
 * COMPLEXITY:
 *     Linear:  O(n1 + n2)
 *     Binary:  O(n_small * log(n_large))
 *     Gallop:  O(n_small * log(ratio)) amortized
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::dot_linear_branchless
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Linear merge sparse dot product with skip optimization.
 *
 * ALGORITHM:
 *     1. 8-way skip: If idx1[i+7] < idx2[j] (8 elements too small), skip 8
 *     2. 4-way skip: Similar for 4 elements
 *     3. Linear merge with prefetch for remaining elements
 *
 * COMPLEXITY:
 *     Time:  O(n1 + n2), often much faster with skip
 *     Space: O(1)
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::dot_binary
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Binary search sparse dot product for imbalanced vector sizes.
 *
 * ALGORITHM:
 *     1. Narrow search range using lower_bound/upper_bound
 *     2. For each element in small vector:
 *        - Binary search in remaining large vector range
 *        - If found, accumulate product
 *        - Advance search base to found position
 *
 * COMPLEXITY:
 *     Time:  O(n_small * log(n_large))
 *     Space: O(1)
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::dot_gallop
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Galloping (exponential) search for highly imbalanced vectors.
 *
 * ALGORITHM:
 *     1. Narrow range using gallop + binary search for boundaries
 *     2. For each element in small vector:
 *        - Exponential search: check positions 1, 2, 4, 8, 16...
 *        - Binary search within found bounds
 *        - Advance search base to found position
 *
 * COMPLEXITY:
 *     Time:  O(n_small * log(ratio)) amortized
 *     Space: O(1)
 *
 * NOTE:
 *     Galloping is optimal when consecutive matches are nearby.
 * -------------------------------------------------------------------------- */

} // namespace scl::kernel::gram
