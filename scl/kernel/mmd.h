// =============================================================================
// FILE: scl/kernel/mmd.h
// BRIEF: API reference for Maximum Mean Discrepancy computation
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::mmd {

/* -----------------------------------------------------------------------------
 * FUNCTION: unary_exp_sum
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute sum of exp(-gamma * x^2) for all non-zero values, caching
 *     individual exp terms for reuse.
 *
 * PARAMETERS:
 *     vals    [in]  Non-zero values array
 *     nnz     [in]  Number of non-zero values
 *     gamma   [in]  RBF kernel parameter
 *     cache   [out] Pre-allocated cache for exp terms
 *     out_sum [out] Sum of exp(-gamma * x^2)
 *
 * PRECONDITIONS:
 *     - vals valid pointer if nnz > 0
 *     - cache.size >= nnz
 *     - gamma > 0
 *
 * POSTCONDITIONS:
 *     - cache[i] = exp(-gamma * vals[i]^2)
 *     - out_sum = sum(cache[0:nnz])
 *
 * ALGORITHM:
 *     8-way SIMD unroll with fused square-negate-multiply-exp operations:
 *     1. Load 8 SIMD vectors
 *     2. Compute x^2 for each
 *     3. Compute exp(-gamma * x^2) using SIMD exp
 *     4. Store to cache and accumulate to 8 separate accumulators
 *     5. Reduce accumulators at end
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(nnz) for cache
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 * -------------------------------------------------------------------------- */
template <typename T>
SCL_FORCE_INLINE void unary_exp_sum(
    const T* SCL_RESTRICT vals,    // Non-zero values
    Size nnz,                      // Number of non-zeros
    T gamma,                       // RBF kernel parameter
    T* SCL_RESTRICT cache,         // Output cache for exp terms [nnz]
    T& out_sum                     // Output sum
);

/* -----------------------------------------------------------------------------
 * FUNCTION: self_kernel_sum
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute sum of RBF kernel for all pairs within a single distribution,
 *     including implicit zeros.
 *
 * PARAMETERS:
 *     vals       [in]  Non-zero values
 *     nnz        [in]  Number of non-zeros
 *     N          [in]  Total dimension (including zeros)
 *     gamma      [in]  RBF kernel parameter
 *     sum_unary  [in]  Pre-computed sum from unary_exp_sum
 *     out_sum    [out] Total kernel sum
 *
 * PRECONDITIONS:
 *     - vals valid pointer if nnz > 0
 *     - N >= nnz
 *     - gamma > 0
 *     - sum_unary computed from unary_exp_sum with same vals, gamma
 *
 * POSTCONDITIONS:
 *     - out_sum = sum_{i,j} k(x_i, x_j) for all N elements
 *     - Includes: zero-zero pairs, zero-nonzero pairs, nonzero-nonzero pairs
 *
 * ALGORITHM:
 *     Exploits symmetry k(x,y) = k(y,x):
 *     1. Zero-zero contribution: (N-nnz)^2 (kernel value = 1)
 *     2. Zero-nonzero contribution: 2 * (N-nnz) * sum_unary
 *     3. Diagonal contribution: nnz (kernel value = 1)
 *     4. Off-diagonal: 2 * sum_{i<j} exp(-gamma * (x_i - x_j)^2)
 *
 *     Off-diagonal computed with 2-way SIMD unroll per row.
 *
 * COMPLEXITY:
 *     Time:  O(nnz^2)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 * -------------------------------------------------------------------------- */
template <typename T>
SCL_FORCE_INLINE void self_kernel_sum(
    const T* SCL_RESTRICT vals,    // Non-zero values
    Size nnz,                      // Number of non-zeros
    Size N,                        // Total dimension
    T gamma,                       // RBF kernel parameter
    T sum_unary,                   // Pre-computed unary sum
    T& out_sum                     // Output kernel sum
);

/* -----------------------------------------------------------------------------
 * FUNCTION: cross_kernel_sum
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute sum of RBF kernel between all pairs from two distributions.
 *
 * PARAMETERS:
 *     vals_x      [in]  Non-zero values from distribution X
 *     nnz_x       [in]  Number of non-zeros in X
 *     vals_y      [in]  Non-zero values from distribution Y
 *     nnz_y       [in]  Number of non-zeros in Y
 *     N_x         [in]  Total dimension of X (including zeros)
 *     N_y         [in]  Total dimension of Y (including zeros)
 *     gamma       [in]  RBF kernel parameter
 *     sum_x_unary [in]  Pre-computed unary sum for X
 *     sum_y_unary [in]  Pre-computed unary sum for Y
 *     out_sum     [out] Total cross-kernel sum
 *
 * PRECONDITIONS:
 *     - vals_x, vals_y valid pointers if respective nnz > 0
 *     - N_x >= nnz_x, N_y >= nnz_y
 *     - gamma > 0
 *     - sum_x_unary, sum_y_unary computed from unary_exp_sum
 *
 * POSTCONDITIONS:
 *     - out_sum = sum_{i,j} k(x_i, y_j) for all pairs
 *
 * ALGORITHM:
 *     Block tiling for cache efficiency (BLOCK_X=64, BLOCK_Y=512):
 *     1. Zero-zero contribution: zeros_x * zeros_y
 *     2. Zero-nonzero contributions using pre-computed unary sums
 *     3. Nonzero-nonzero: blocked iteration with 2-way SIMD unroll
 *
 * COMPLEXITY:
 *     Time:  O(nnz_x * nnz_y)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 * -------------------------------------------------------------------------- */
template <typename T>
SCL_FORCE_INLINE void cross_kernel_sum(
    const T* SCL_RESTRICT vals_x,  // Non-zero values from X
    Size nnz_x,                    // Number of non-zeros in X
    const T* SCL_RESTRICT vals_y,  // Non-zero values from Y
    Size nnz_y,                    // Number of non-zeros in Y
    Size N_x,                      // Total dimension of X
    Size N_y,                      // Total dimension of Y
    T gamma,                       // RBF kernel parameter
    T sum_x_unary,                 // Pre-computed unary sum for X
    T sum_y_unary,                 // Pre-computed unary sum for Y
    T& out_sum                     // Output cross-kernel sum
);

/* -----------------------------------------------------------------------------
 * FUNCTION: mmd_rbf
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Maximum Mean Discrepancy with RBF kernel between two
 *     distributions represented as sparse matrices.
 *
 * PARAMETERS:
 *     mat_x   [in]  Sparse matrix for distribution X (features x samples_x)
 *     mat_y   [in]  Sparse matrix for distribution Y (features x samples_y)
 *     output  [out] MMD^2 value for each feature
 *     gamma   [in]  RBF kernel parameter (default: 1.0)
 *
 * PRECONDITIONS:
 *     - mat_x.primary_dim() == mat_y.primary_dim()
 *     - output.len == mat_x.primary_dim()
 *     - gamma > 0
 *
 * POSTCONDITIONS:
 *     - output[i] = MMD^2 between X[i,:] and Y[i,:] for each feature i
 *     - output[i] >= 0 (negative values from numerical error clamped)
 *     - output[i] = 0 if both rows are all zeros
 *
 * ALGORITHM:
 *     MMD^2 = E[k(X,X)] + E[k(Y,Y)] - 2*E[k(X,Y)]
 *
 *     For each feature in parallel:
 *     1. Compute unary exp sums with caching
 *     2. Compute self-kernel sums (exploiting symmetry)
 *     3. Compute cross-kernel sum (with block tiling)
 *     4. Combine: (K_xx/n_x^2) + (K_yy/n_y^2) - 2*(K_xy/(n_x*n_y))
 *
 *     Optimizations:
 *     - Pre-computed reciprocals for normalization
 *     - 8-way SIMD unroll in unary computation
 *     - Block tiling (64x512) in cross-kernel for cache efficiency
 *     - Symmetric computation in self-kernel
 *     - DualWorkspacePool for thread-local caches
 *
 * COMPLEXITY:
 *     Time:  O(features * (nnz_x^2 + nnz_y^2 + nnz_x*nnz_y))
 *     Space: O(max(nnz_x, nnz_y)) per thread for caching
 *
 * THREAD SAFETY:
 *     Safe - parallelized over features with thread-local workspace
 *
 * THROWS:
 *     SCL_CHECK_DIM - if primary dimensions or output size mismatch
 *
 * NUMERICAL NOTES:
 *     - RBF kernel: k(x,y) = exp(-gamma * (x-y)^2)
 *     - MMD^2 can be slightly negative due to numerical error; clamped to 0
 *     - For all-zero features: MMD^2 = 0
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void mmd_rbf(
    const Sparse<T, IsCSR>& mat_x, // Sparse matrix for distribution X
    const Sparse<T, IsCSR>& mat_y, // Sparse matrix for distribution Y
    Array<T> output,               // Output MMD^2 values [n_features]
    T gamma = T(1)                 // RBF kernel parameter
);

} // namespace scl::kernel::mmd
