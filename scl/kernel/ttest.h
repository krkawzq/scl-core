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
 * FUNCTION: compute_group_stats
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute per-group mean, variance, and count for each feature.
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
 *     - out_means.len >= n_features * n_groups
 *     - out_vars.len >= n_features * n_groups
 *     - out_counts.len >= n_features * n_groups
 *
 * POSTCONDITIONS:
 *     - For feature f, group g at index (f * n_groups + g):
 *       - out_means contains sample mean
 *       - out_vars contains sample variance (Bessel corrected)
 *       - out_counts contains number of non-zero samples
 *
 * OUTPUT LAYOUT:
 *     Row-major by feature: [feat0_g0, feat0_g1, ..., feat1_g0, ...]
 *
 * ALGORITHM:
 *     Two-pass approach:
 *     Pass 1: Parallel over features
 *         - 4-way unrolled loop accumulating sum and sum_sq per group
 *         - Indirect access: group_ids[indices[k]]
 *     Pass 2: Parallel finalization
 *         - mean = sum / count
 *         - var = (sum_sq/n - mean^2) * n/(n-1)  [Bessel correction]
 *
 * COMPLEXITY:
 *     Time:  O(nnz + n_features * n_groups)
 *     Space: O(1) beyond output
 *
 * THREAD SAFETY:
 *     Safe - parallel over features, no shared mutable state
 *
 * NUMERICAL NOTES:
 *     - Zero count groups have mean=0, var=0
 *     - Single sample groups have var=0
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void compute_group_stats(
    const Sparse<T, IsCSR>& matrix,   // Input sparse matrix [n_features x n_samples]
    Array<const int32_t> group_ids,    // Group label per sample [n_samples]
    Size n_groups,                      // Number of groups
    Array<Real> out_means,              // Output means [n_features * n_groups]
    Array<Real> out_vars,               // Output variances [n_features * n_groups]
    Array<Size> out_counts              // Output counts [n_features * n_groups]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: ttest
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute t-test statistics comparing each group against reference group.
 *
 * PARAMETERS:
 *     matrix       [in]  Sparse matrix, shape (n_features, n_samples)
 *     group_ids    [in]  Group assignment per sample
 *     n_groups     [in]  Number of groups (reference + targets)
 *     group_sizes  [in]  Total size of each group, size = n_groups
 *     out_t_stats  [out] T-statistics, size = n_features * (n_groups-1)
 *     out_p_values [out] P-values (two-tailed), size = n_features * (n_groups-1)
 *     out_log2_fc  [out] Log2 fold changes, size = n_features * (n_groups-1)
 *     use_welch    [in]  Use Welch's t-test (unequal variances)
 *
 * PRECONDITIONS:
 *     - group_sizes[0] is reference group size
 *     - group_sizes.len >= n_groups
 *     - Output arrays sized for n_features * (n_groups - 1)
 *
 * POSTCONDITIONS:
 *     - For feature f, target group t at index (f * n_targets + t):
 *       - out_t_stats: t-statistic (target vs reference)
 *       - out_p_values: two-tailed p-value
 *       - out_log2_fc: log2((mean_target + eps) / (mean_ref + eps))
 *
 * OUTPUT LAYOUT:
 *     For n_groups=3: [feat0_g1, feat0_g2, feat1_g1, feat1_g2, ...]
 *     Group 0 is always reference.
 *
 * ALGORITHM:
 *     1. Compute group statistics via compute_group_stats
 *     2. For each feature in parallel:
 *        For each target group:
 *            - Compute mean difference
 *            - Compute standard error:
 *              Welch: se = sqrt(var1/n1 + var2/n2)
 *              Pooled: se = sqrt(pooled_var * (1/n1 + 1/n2))
 *            - t_stat = mean_diff / se
 *            - p_value via erfc approximation
 *
 * COMPLEXITY:
 *     Time:  O(nnz + n_features * n_groups)
 *     Space: O(n_features * n_groups) for intermediate stats
 *
 * THREAD SAFETY:
 *     Safe - parallel over features
 *
 * NUMERICAL NOTES:
 *     - Pseudocount 1e-9 added for log2FC to avoid division by zero
 *     - P-value computed via fast erfc approximation (Horner polynomial)
 *     - Empty groups yield t_stat=0, p_value=1
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void ttest(
    const Sparse<T, IsCSR>& matrix,    // Input sparse matrix
    Array<const int32_t> group_ids,     // Group labels
    Size n_groups,                       // Number of groups
    Array<const Size> group_sizes,       // Size of each group
    Array<Real> out_t_stats,             // Output t-statistics
    Array<Real> out_p_values,            // Output p-values
    Array<Real> out_log2_fc,             // Output log2 fold changes
    bool use_welch                        // Use Welch's t-test
);

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::fast_erfc
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Fast complementary error function approximation.
 *
 * ALGORITHM:
 *     Horner polynomial approximation (5-term expansion):
 *         t = 1 / (1 + 0.3275911 * |x|)
 *         erfc(x) = (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-x^2)
 *
 *     Coefficients: Abramowitz and Stegun approximation 7.1.26
 *
 * ACCURACY:
 *     Maximum error < 1.5e-7
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::t_to_pvalue
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Convert t-statistic to two-tailed p-value.
 *
 * FORMULA:
 *     p = 2 * erfc(|t| / sqrt(2))
 *
 * NOTE:
 *     Uses fast_erfc for approximation.
 *     Assumes large degrees of freedom (normal approximation).
 * -------------------------------------------------------------------------- */

} // namespace scl::kernel::ttest
