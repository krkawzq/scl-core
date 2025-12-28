// =============================================================================
// FILE: scl/kernel/stat/oneway_anova.h
// BRIEF: API reference for One-way ANOVA F-test
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

namespace scl::kernel::stat::oneway_anova {

/* -----------------------------------------------------------------------------
 * FUNCTION: count_k_groups
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Count the number of samples in each of k groups.
 *
 * PARAMETERS:
 *     group_ids   [in]  Group assignment array
 *     n_groups    [in]  Number of groups (k)
 *     out_counts  [out] Group counts, size = n_groups
 *
 * PRECONDITIONS:
 *     - out_counts sized >= n_groups
 *     - group_ids[i] in [0, n_groups) or negative (ignored)
 *
 * POSTCONDITIONS:
 *     - out_counts[g] = number of samples in group g
 * -------------------------------------------------------------------------- */
void count_k_groups(
    Array<const int32_t> group_ids,  // Group assignment
    Size n_groups,                    // Number of groups (k)
    Size* out_counts                  // [n_groups] output counts
);

/* -----------------------------------------------------------------------------
 * FUNCTION: oneway_anova
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute One-way ANOVA F-test for k groups (parametric).
 *
 * PARAMETERS:
 *     matrix       [in]  Sparse matrix (features x samples)
 *     group_ids    [in]  Group assignment (0 to k-1)
 *     n_groups     [in]  Number of groups (k)
 *     out_F_stats  [out] F statistics, size = n_features
 *     out_p_values [out] P-values (F distribution)
 *
 * PRECONDITIONS:
 *     - n_groups >= 2
 *     - group_ids[i] in [0, n_groups) or negative (ignored)
 *     - At least 2 groups have members
 *     - N > k (total samples > number of groups)
 *     - Output arrays sized >= matrix.primary_dim()
 *
 * POSTCONDITIONS:
 *     - out_F_stats[i] contains F statistic for feature i
 *     - out_p_values[i] = P(F(df1, df2) >= F) where df1=k-1, df2=N-k
 *
 * ALGORITHM:
 *     For each feature in parallel:
 *     1. Compute group sums and grand mean (including sparse zeros)
 *     2. Compute group means: mean_g = sum_g / n_g
 *     3. SS_between = sum(n_g * (mean_g - grand_mean)^2)
 *     4. SS_total = sum((x_i - grand_mean)^2) for all observations
 *     5. SS_within = SS_total - SS_between
 *     6. F = (SS_between / df_between) / (SS_within / df_within)
 *     7. P-value from F distribution (Wilson-Hilferty approximation)
 *
 * COMPLEXITY:
 *     Time:  O(features * nnz_per_feature)
 *     Space: O(threads * n_groups)
 *
 * THREAD SAFETY:
 *     Safe - parallelized over features with thread-local workspace
 *
 * THROWS:
 *     ArgumentError - if n_groups < 2 or fewer than 2 groups have members
 *     ArgumentError - if N <= k (insufficient degrees of freedom)
 *
 * NUMERICAL NOTES:
 *     - Assumes normality and homogeneity of variance
 *     - Handles sparse zeros by incorporating them into means and variances
 *     - Uses F-distribution approximation for p-values
 *     - Sensitive to outliers; consider Kruskal-Wallis for non-normal data
 *
 * INTERPRETATION:
 *     F statistic measures the ratio of between-group variance to
 *     within-group variance. Large F indicates group means differ
 *     significantly. ANOVA assumes equal variances across groups;
 *     consider Welch's ANOVA for unequal variances.
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void oneway_anova(
    const Sparse<T, IsCSR>& matrix,         // Input sparse matrix
    Array<const int32_t> group_ids,         // Group assignment [0, k-1]
    Size n_groups,                           // Number of groups (k)
    Array<Real> out_F_stats,                 // [n_features] F statistics
    Array<Real> out_p_values                 // [n_features] P-values
);

} // namespace scl::kernel::stat::oneway_anova
