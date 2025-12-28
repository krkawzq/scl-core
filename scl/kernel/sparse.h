// =============================================================================
// FILE: scl/kernel/sparse.h
// BRIEF: API reference for sparse matrix statistics kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::sparse {

/* -----------------------------------------------------------------------------
 * FUNCTION: primary_sums
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute the sum of values along each primary dimension (row for CSR,
 *     column for CSC).
 *
 * PARAMETERS:
 *     matrix [in]  Sparse matrix in CSR or CSC format
 *     output [out] Pre-allocated buffer for sums, size = primary_dim
 *
 * PRECONDITIONS:
 *     - output.len == matrix.primary_dim()
 *     - matrix is valid sparse format
 *
 * POSTCONDITIONS:
 *     - output[i] = sum of all non-zero values in primary slice i
 *     - Empty slices have output[i] = 0
 *     - Matrix unchanged
 *
 * ALGORITHM:
 *     For each primary index in parallel:
 *         1. Get values span for the primary slice
 *         2. Use scl::vectorize::sum for SIMD-optimized reduction
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over primary dimension, no shared mutable state
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void primary_sums(
    const Sparse<T, IsCSR>& matrix,   // Input sparse matrix
    Array<T> output                    // Output sums [primary_dim]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: primary_means
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute the mean of values along each primary dimension, accounting
 *     for implicit zeros (dividing by secondary_dim, not nnz).
 *
 * PARAMETERS:
 *     matrix [in]  Sparse matrix
 *     output [out] Pre-allocated buffer for means
 *
 * PRECONDITIONS:
 *     - output.len == matrix.primary_dim()
 *     - matrix.secondary_dim() > 0
 *
 * POSTCONDITIONS:
 *     - output[i] = sum(primary_slice_i) / secondary_dim
 *     - Empty slices have output[i] = 0
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over primary dimension
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void primary_means(
    const Sparse<T, IsCSR>& matrix,   // Input sparse matrix
    Array<T> output                    // Output means [primary_dim]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: primary_variances
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute variance along each primary dimension using fused sum and
 *     sum-of-squares computation.
 *
 * PARAMETERS:
 *     matrix [in]  Sparse matrix
 *     output [out] Pre-allocated buffer for variances
 *     ddof   [in]  Delta degrees of freedom (default 1 for sample variance)
 *
 * PRECONDITIONS:
 *     - output.len == matrix.primary_dim()
 *     - ddof >= 0 and ddof < secondary_dim
 *
 * POSTCONDITIONS:
 *     - output[i] = var(primary_slice_i) with ddof adjustment
 *     - Variance clamped to >= 0 for numerical stability
 *
 * ALGORITHM:
 *     For each primary index in parallel:
 *         1. Use SIMD fused sum+sumsq helper (4-way unroll with FMA)
 *         2. Compute variance = (sumsq - sum*mean) / (N - ddof)
 *         3. Clamp negative values to zero
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over primary dimension
 *
 * NUMERICAL NOTES:
 *     Uses compensated summation pattern for improved accuracy
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void primary_variances(
    const Sparse<T, IsCSR>& matrix,   // Input sparse matrix
    Array<T> output,                   // Output variances [primary_dim]
    int ddof = 1                       // Degrees of freedom adjustment
);

/* -----------------------------------------------------------------------------
 * FUNCTION: primary_nnz
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Get the number of non-zero elements in each primary slice.
 *
 * PARAMETERS:
 *     matrix [in]  Sparse matrix
 *     output [out] Pre-allocated buffer for nnz counts
 *
 * PRECONDITIONS:
 *     - output.len == matrix.primary_dim()
 *
 * POSTCONDITIONS:
 *     - output[i] = number of stored elements in primary slice i
 *
 * ALGORITHM:
 *     - For small matrices (< PARALLEL_THRESHOLD): sequential loop
 *     - For large matrices: parallel batched processing (BATCH_SIZE rows per task)
 *
 * COMPLEXITY:
 *     Time:  O(primary_dim)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized with batching for cache efficiency
 *
 * PERFORMANCE NOTES:
 *     Uses batched parallel processing to reduce scheduling overhead
 *     for this lightweight operation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void primary_nnz(
    const Sparse<T, IsCSR>& matrix,   // Input sparse matrix
    Array<Index> output                // Output nnz counts [primary_dim]
);

} // namespace scl::kernel::sparse
