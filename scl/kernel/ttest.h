// =============================================================================
// FILE: scl/kernel/ttest.h
// BRIEF: API reference for T-test computation kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::ttest {

/* -----------------------------------------------------------------------------
 * FUNCTION: count_groups
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Count elements in each of two groups.
 *
 * PARAMETERS:
 *     group_ids  [in]  Array of group assignments (0 or 1)
 *     out_n1     [out] Count of elements with group_id == 0
 *     out_n2     [out] Count of elements with group_id == 1
 *
 * COMPLEXITY:
 *     Time:  O(n) with SIMD optimization
 *     Space: O(1)
 * -------------------------------------------------------------------------- */
void count_groups(
    Array<const int32_t> group_ids,         // Group assignments
    Size& out_n1,                           // Group 0 count
    Size& out_n2                            // Group 1 count
);

/* -----------------------------------------------------------------------------
 * FUNCTION: ttest
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Welch's or Student's t-test for each feature comparing two groups.
 *
 * PARAMETERS:
 *     matrix       [in]  Sparse matrix (features x samples)
 *     group_ids    [in]  Binary group assignment (0 or 1)
 *     out_t_stats  [out] T-statistics, size = n_features
 *     out_p_values [out] Two-tailed p-values, size = n_features
 *     out_log2_fc  [out] Log2 fold change (group1/group0), size = n_features
 *     use_welch    [in]  Use Welch's t-test (default true)
 *
 * PRECONDITIONS:
 *     - matrix.secondary_dim() == group_ids.len
 *     - Output arrays have size >= matrix.primary_dim()
 *     - group_ids contains only values 0 or 1
 *     - Both groups must have at least one member
 *
 * POSTCONDITIONS:
 *     - out_t_stats[i] contains t-statistic for feature i
 *     - out_p_values[i] contains two-tailed p-value
 *     - out_log2_fc[i] = log2((mean_group1 + eps) / (mean_group0 + eps))
 *
 * ALGORITHM:
 *     For each feature in parallel:
 *     1. Partition non-zero values by group (4-way unrolled, prefetch)
 *     2. Accumulate sum and sum_sq during partition
 *     3. Compute mean including zeros: mean = sum / n_total
 *     4. Compute variance with zero adjustment and Bessel correction
 *     5. Compute standard error:
 *        - Welch: se = sqrt(var1/n1 + var2/n2)
 *        - Pooled: se = sqrt(pooled_var * (1/n1 + 1/n2))
 *     6. t_stat = (mean2 - mean1) / se
 *     7. p_value via normal approximation (fast erfc)
 *
 * COMPLEXITY:
 *     Time:  O(features * nnz_per_feature)
 *     Space: O(threads * max_row_length) for workspace
 *
 * THREAD SAFETY:
 *     Safe - parallelized over features with thread-local workspace
 *
 * THROWS:
 *     ArgumentError - if either group is empty
 *
 * NUMERICAL NOTES:
 *     - EPS = 1e-9 added to means for log2FC stability
 *     - SIGMA_MIN = 1e-12 threshold for valid standard error
 *     - Variance clamped to >= 0 for numerical stability
 *     - Uses fast erfc approximation (max error < 1.5e-7)
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void ttest(
    const Sparse<T, IsCSR>& matrix,         // Input sparse matrix
    Array<const int32_t> group_ids,        // Binary group assignment
    Array<Real> out_t_stats,                // [n_features] T-statistics
    Array<Real> out_p_values,               // [n_features] P-values
    Array<Real> out_log2_fc,                // [n_features] Log2 fold change
    bool use_welch = true                   // Use Welch's t-test
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_group_stats (Legacy)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute per-group mean, variance, and count for each feature.
 *     Retained for backward compatibility.
 *
 * PARAMETERS:
 *     matrix      [in]  Sparse matrix, shape (n_features, n_samples)
 *     group_ids   [in]  Group assignment per sample, size = n_samples
 *     n_groups    [in]  Number of groups
 *     out_means   [out] Group means, size = n_features * n_groups
 *     out_vars    [out] Group variances, size = n_features * n_groups
 *     out_counts  [out] Group counts, size = n_features * n_groups
 *
 * PRECONDITIONS:
 *     - group_ids[i] in range [0, n_groups) or negative (ignored)
 *     - Output arrays sized >= n_features * n_groups
 *
 * POSTCONDITIONS:
 *     - For feature f, group g at index (f * n_groups + g):
 *       - out_means: sample mean of non-zero values
 *       - out_vars: sample variance with Bessel correction
 *       - out_counts: number of non-zero samples
 *
 * OUTPUT LAYOUT:
 *     Row-major: [feat0_g0, feat0_g1, ..., feat1_g0, feat1_g1, ...]
 *
 * NOTE:
 *     Consider using ttest() directly for two-group comparisons.
 *     This function is retained for k-group scenarios.
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void compute_group_stats(
    const Sparse<T, IsCSR>& matrix,         // Input matrix
    Array<const int32_t> group_ids,        // Group labels [n_samples]
    Size n_groups,                          // Number of groups
    Array<Real> out_means,                  // [n_features * n_groups]
    Array<Real> out_vars,                   // [n_features * n_groups]
    Array<Size> out_counts                  // [n_features * n_groups]
);

} // namespace scl::kernel::ttest
