// =============================================================================
// FILE: scl/kernel/group.h
// BRIEF: API reference for group aggregation kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::group {

/* -----------------------------------------------------------------------------
 * CONFIGURATION CONSTANTS
 * -----------------------------------------------------------------------------
 * PREFETCH_DISTANCE - Elements to prefetch ahead for indirect access (64)
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * FUNCTION: group_stats
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute per-group mean and variance for each feature.
 *
 * PARAMETERS:
 *     matrix        [in]  Sparse matrix, shape (n_features, n_samples)
 *     group_ids     [in]  Group assignment per sample, size = n_samples
 *     n_groups      [in]  Number of groups
 *     group_sizes   [in]  Total size of each group, size = n_groups
 *     out_means     [out] Group means, size = n_features * n_groups
 *     out_vars      [out] Group variances, size = n_features * n_groups
 *     ddof          [in]  Delta degrees of freedom for variance (default: 1)
 *     include_zeros [in]  Include zero values in statistics (default: true)
 *
 * PRECONDITIONS:
 *     - group_ids[i] in range [0, n_groups) or negative (ignored)
 *     - group_sizes.len >= n_groups
 *     - out_means.len >= n_features * n_groups
 *     - out_vars.len >= n_features * n_groups
 *
 * POSTCONDITIONS:
 *     - For feature f, group g at index (f * n_groups + g):
 *       - out_means contains group mean
 *       - out_vars contains group variance with ddof correction
 *
 * OUTPUT LAYOUT:
 *     Row-major by feature: [feat0_g0, feat0_g1, ..., feat1_g0, ...]
 *
 * ALGORITHM:
 *     For each feature in parallel:
 *         1. Zero-initialize accumulators
 *         2. 4-way unrolled loop with prefetch:
 *            - Indirect access: group_ids[indices[k]]
 *            - Accumulate sum and sum_sq per group
 *            - Optionally count non-zeros
 *         3. Finalize statistics:
 *            - N = group_sizes[g] if include_zeros else nnz_counts[g]
 *            - mean = sum / N
 *            - var = (sum_sq - N*mean^2) / (N - ddof)
 *
 * COMPLEXITY:
 *     Time:  O(nnz + n_features * n_groups)
 *     Space: O(n_groups) per thread for nnz_counts (if not include_zeros)
 *
 * THREAD SAFETY:
 *     Safe - parallel over features, thread-local accumulators
 *
 * NUMERICAL NOTES:
 *     - Groups with N <= ddof have mean=0, var=0
 *     - Variance clamped to >= 0 for numerical stability
 *     - Uses Welford-style numerically stable variance formula
 *
 * MEMORY OPTIMIZATION:
 *     - Stack allocation for n_groups <= 256
 *     - Heap allocation for larger group counts
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void group_stats(
    const Sparse<T, IsCSR>& matrix,    // Input sparse matrix [n_features x n_samples]
    Array<const int32_t> group_ids,     // Group label per sample [n_samples]
    Size n_groups,                       // Number of groups
    Array<const Size> group_sizes,       // Size of each group [n_groups]
    Array<Real> out_means,               // Output means [n_features * n_groups]
    Array<Real> out_vars,                // Output variances [n_features * n_groups]
    int ddof = 1,                         // Degrees of freedom correction
    bool include_zeros = true             // Include zeros in statistics
);

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::finalize_stats
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Convert accumulated sums to mean and variance.
 *
 * PARAMETERS:
 *     mean_ptr     [in,out] Sum values, converted to means
 *     var_ptr      [in,out] Sum squared values, converted to variances
 *     group_sizes  [in]     Total size of each group
 *     nnz_counts   [in]     Non-zero counts (nullable if include_zeros)
 *     n_groups     [in]     Number of groups
 *     ddof         [in]     Degrees of freedom correction
 *     include_zeros [in]    Use group_sizes vs nnz_counts for N
 *
 * FORMULA:
 *     N = include_zeros ? group_sizes[g] : nnz_counts[g]
 *     mean = sum / N
 *     var = (sum_sq - N * mean^2) / (N - ddof)
 *
 * POSTCONDITIONS:
 *     - mean_ptr[g] contains mean for group g
 *     - var_ptr[g] contains variance for group g
 *     - Groups with N <= ddof have mean=0, var=0
 *
 * MUTABILITY:
 *     INPLACE - modifies mean_ptr and var_ptr arrays
 * -------------------------------------------------------------------------- */

} // namespace scl::kernel::group
