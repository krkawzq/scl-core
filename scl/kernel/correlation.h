// =============================================================================
// FILE: scl/kernel/correlation.h
// BRIEF: API reference for Pearson correlation kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::correlation {

/* -----------------------------------------------------------------------------
 * CONFIGURATION CONSTANTS
 * -----------------------------------------------------------------------------
 * CHUNK_SIZE        - Row chunk size for cache blocking (64)
 * STAT_CHUNK        - Statistics computation chunk size (256)
 * PREFETCH_DISTANCE - Elements to prefetch ahead (32)
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_stats
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute mean and inverse standard deviation for each row.
 *
 * PARAMETERS:
 *     matrix      [in]  Sparse matrix, shape (n_features, n_samples)
 *     out_means   [out] Mean of each row, size = n_features
 *     out_inv_stds [out] Inverse std dev of each row, size = n_features
 *
 * PRECONDITIONS:
 *     - out_means.len >= matrix.primary_dim()
 *     - out_inv_stds.len >= matrix.primary_dim()
 *
 * POSTCONDITIONS:
 *     - out_means[i] = sum(row_i) / n_samples
 *     - out_inv_stds[i] = 1/std(row_i), or 0 if variance is zero
 *
 * ALGORITHM:
 *     For each row in parallel:
 *         1. Fused sum + sum_squared computation with 4-way SIMD
 *         2. mean = sum / n
 *         3. var = sum_sq/n - mean^2
 *         4. inv_std = 1/sqrt(var) if var > 0, else 0
 *
 * COMPLEXITY:
 *     Time:  O(nnz / n_threads)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - parallel over independent rows
 *
 * NUMERICAL NOTES:
 *     - Variance clamped to >= 0 for numerical stability
 *     - Zero-variance rows get inv_std = 0
 * -------------------------------------------------------------------------- */
template <typename MatrixT>
    requires SparseLike<MatrixT>
void compute_stats(
    const MatrixT& matrix,                              // Input sparse matrix
    Array<typename MatrixT::ValueType> out_means,       // Output means [n_features]
    Array<typename MatrixT::ValueType> out_inv_stds     // Output inverse stds [n_features]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: pearson
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Pearson correlation matrix for sparse feature matrix.
 *
 * PARAMETERS:
 *     matrix   [in]  Sparse matrix, shape (n_features, n_samples)
 *     means    [in]  Precomputed means per feature (optional overload)
 *     inv_stds [in]  Precomputed inverse stds per feature (optional overload)
 *     output   [out] Correlation matrix, size = n_features * n_features
 *
 * PRECONDITIONS:
 *     - output.len >= n_features^2
 *     - means.len >= n_features (if provided)
 *     - inv_stds.len >= n_features (if provided)
 *
 * POSTCONDITIONS:
 *     - output is symmetric: output[i,j] == output[j,i]
 *     - Diagonal is 1 for non-zero-variance features, 0 otherwise
 *     - Values in range [-1, 1]
 *
 * OUTPUT LAYOUT:
 *     Row-major: output[i * n_features + j] = corr(feature_i, feature_j)
 *
 * ALGORITHM:
 *     Key optimizations:
 *     1. Symmetric computation: only compute upper triangle
 *     2. Sparse centered dot product using algebraic identity:
 *        cov(a,b) = sum(a*b) - mean_a*sum(b) - mean_b*sum(a) + n*mean_a*mean_b
 *     3. 8/4-way skip optimization in sparse merge
 *     4. Chunk-based parallelization for cache locality
 *     5. Early skip for zero-variance features
 *
 * COMPLEXITY:
 *     Time:  O(n_features^2 * avg_nnz_per_row / n_threads)
 *     Space: O(1) beyond output
 *
 * THREAD SAFETY:
 *     Safe - chunk-based parallelization with no conflicts
 *
 * NUMERICAL NOTES:
 *     - Correlation clamped to [-1, 1] for numerical stability
 *     - Zero-variance features have correlation 0 with all others
 * -------------------------------------------------------------------------- */
template <typename MatrixT>
    requires SparseLike<MatrixT>
void pearson(
    const MatrixT& matrix,                                      // Input [n_features x n_samples]
    Array<const typename MatrixT::ValueType> means,             // Precomputed means
    Array<const typename MatrixT::ValueType> inv_stds,          // Precomputed inverse stds
    Array<typename MatrixT::ValueType> output                   // Output [n_features x n_features]
);

template <typename MatrixT>
    requires SparseLike<MatrixT>
void pearson(
    const MatrixT& matrix,                                      // Input sparse matrix
    Array<typename MatrixT::ValueType> output                   // Output correlation matrix
);

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::compute_sum_sq_simd
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Fused single-pass sum and sum-of-squares computation.
 *
 * ALGORITHM:
 *     4-way SIMD unrolled loop:
 *         - Load 4 vectors per iteration
 *         - Dual accumulators for sum (v_sum0, v_sum1)
 *         - Dual accumulators for sum_sq (v_sq0, v_sq1)
 *         - FMA: v_sq = v_sq + v * v
 *     Reduces to scalar via SumOfLanes
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1)
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::sparse_centered_dot
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute covariance of two sparse vectors using algebraic identity.
 *
 * FORMULA:
 *     cov = sum_ab - mean_a * sum_b - mean_b * sum_a + n * mean_a * mean_b
 *
 *     Where:
 *         sum_ab = sum over matching indices of (a_i * b_i)
 *         sum_a  = sum of all a values
 *         sum_b  = sum of all b values
 *         n      = total dimension (including zeros)
 *
 * ALGORITHM:
 *     1. O(1) range disjointness check
 *     2. If disjoint: sum_ab = 0, compute sum_a and sum_b separately
 *     3. Otherwise: merge with 8/4-way skip optimization
 *        - Track sum_ab, sum_a, sum_b in single pass
 *
 * COMPLEXITY:
 *     Time:  O(len_a + len_b)
 *     Space: O(1)
 *
 * NOTE:
 *     This avoids materializing dense centered vectors.
 * -------------------------------------------------------------------------- */

} // namespace scl::kernel::correlation
