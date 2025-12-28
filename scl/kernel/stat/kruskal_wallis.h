// =============================================================================
// FILE: scl/kernel/stat/kruskal_wallis.h
// BRIEF: API reference for Kruskal-Wallis H test
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

namespace scl::kernel::stat::kruskal_wallis {

/* -----------------------------------------------------------------------------
 * FUNCTION: kruskal_wallis
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Kruskal-Wallis H test for k groups (non-parametric ANOVA).
 *
 * PARAMETERS:
 *     matrix       [in]  Sparse matrix (features x samples)
 *     group_ids    [in]  Group assignment (0 to k-1)
 *     n_groups     [in]  Number of groups (k)
 *     out_H_stats  [out] H statistics, size = n_features
 *     out_p_values [out] P-values (chi-squared approximation)
 *
 * PRECONDITIONS:
 *     - n_groups >= 2
 *     - group_ids[i] in [0, n_groups) or negative (ignored)
 *     - At least 2 groups have members
 *     - Output arrays sized >= matrix.primary_dim()
 *
 * POSTCONDITIONS:
 *     - out_H_stats[i] contains H statistic for feature i
 *     - out_p_values[i] = P(chi^2(df) >= H) where df = k - 1
 *
 * ALGORITHM:
 *     For each feature in parallel:
 *     1. Extract non-zero values with group tags
 *     2. Sort values (argsort)
 *     3. Compute rank sums per group with tie handling
 *     4. H = 12/(N(N+1)) * sum(R_i^2/n_i) - 3(N+1)
 *     5. Apply tie correction: H / (1 - sum(t^3-t)/(N^3-N))
 *     6. P-value from chi-squared(df = k - 1)
 *
 * COMPLEXITY:
 *     Time:  O(features * nnz_per_feature * log(nnz_per_feature))
 *     Space: O(threads * (max_row_length + n_groups))
 *
 * THREAD SAFETY:
 *     Safe - parallelized over features with thread-local workspace
 *
 * THROWS:
 *     ArgumentError - if n_groups < 2 or fewer than 2 groups have members
 *
 * INTERPRETATION:
 *     H statistic measures whether samples from different groups
 *     come from the same distribution. Large H indicates differences.
 *     Robust to non-normal distributions and outliers.
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void kruskal_wallis(
    const Sparse<T, IsCSR>& matrix,         // Input sparse matrix
    Array<const int32_t> group_ids,        // Group assignment [0, k-1]
    Size n_groups,                          // Number of groups (k)
    Array<Real> out_H_stats,                // [n_features] H statistics
    Array<Real> out_p_values                // [n_features] P-values
);

} // namespace scl::kernel::stat::kruskal_wallis
