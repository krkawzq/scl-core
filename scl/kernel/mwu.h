// =============================================================================
// FILE: scl/kernel/mwu.h
// BRIEF: API reference for Mann-Whitney U test
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::mwu {

/* -----------------------------------------------------------------------------
 * FUNCTION: count_groups
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Count the number of samples in each group (0 and 1).
 *
 * PARAMETERS:
 *     group_ids [in]  Array of group labels (0 or 1)
 *     out_n1    [out] Count of samples in group 0
 *     out_n2    [out] Count of samples in group 1
 *
 * PRECONDITIONS:
 *     - group_ids contains only values 0 or 1
 *
 * POSTCONDITIONS:
 *     - out_n1 = count of 0s in group_ids
 *     - out_n2 = count of 1s in group_ids
 *
 * ALGORITHM:
 *     Uses SIMD-optimized scl::vectorize::count for parallel counting.
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - read-only access to group_ids
 * -------------------------------------------------------------------------- */
inline void count_groups(
    Array<const int32_t> group_ids, // Group labels (0 or 1)
    Size& out_n1,                   // Output: count of group 0
    Size& out_n2                    // Output: count of group 1
);

/* -----------------------------------------------------------------------------
 * FUNCTION: mwu_test
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Perform Mann-Whitney U test for each feature (row/column) in a sparse
 *     matrix, comparing two groups of samples.
 *
 * PARAMETERS:
 *     matrix       [in]  Sparse matrix (features x samples)
 *     group_ids    [in]  Group labels for each sample (0 or 1)
 *     out_u_stats  [out] U statistics for each feature
 *     out_p_values [out] Two-sided p-values for each feature
 *     out_log2_fc  [out] Log2 fold change (group1 / group0) for each feature
 *     out_auroc    [out] Optional: AUROC values (U / (n1 * n2))
 *
 * PRECONDITIONS:
 *     - matrix.secondary_dim() == group_ids.len
 *     - out_u_stats.len == out_p_values.len == out_log2_fc.len == matrix.primary_dim()
 *     - group_ids contains only values 0 or 1
 *     - Both groups must have at least one member
 *     - If out_auroc provided: out_auroc.len == matrix.primary_dim()
 *
 * POSTCONDITIONS:
 *     - out_u_stats[i] contains U statistic for feature i
 *     - out_p_values[i] contains two-sided p-value (normal approximation)
 *     - out_log2_fc[i] contains log2(mean_group1 / mean_group0)
 *     - out_auroc[i] (if provided) contains AUROC = U / (n1 * n2)
 *     - For features with all zeros: U=0, p=1, AUROC=0.5
 *
 * ALGORITHM:
 *     For each feature in parallel:
 *     1. Partition non-zero values by group using pre-allocated buffers
 *     2. Sort each group using VQSort (SIMD-optimized)
 *     3. Merge sorted arrays to compute rank sum with tie correction
 *     4. Compute U statistic: U = R1 - n1*(n1+1)/2
 *     5. Apply continuity correction and normal approximation for p-value
 *     6. Compute log2 fold change from group means
 *     7. Optionally compute AUROC = U / (n1 * n2)
 *
 *     Optimizations:
 *     - Binary search for negative/positive boundary (O(log n))
 *     - Prefetch in merge loop for cache efficiency
 *     - Precomputed reciprocals to avoid division
 *     - 4-way unrolled partition loop
 *
 * COMPLEXITY:
 *     Time:  O(features * nnz_per_feature * log(nnz_per_feature))
 *     Space: O(max_nnz) per thread for sorting buffers
 *
 * THREAD SAFETY:
 *     Safe - parallelized over features with thread-local workspace
 *
 * THROWS:
 *     SCL_CHECK_ARG - if either group is empty
 *
 * NUMERICAL NOTES:
 *     - Uses normal approximation for p-value (valid for n1, n2 >= 10)
 *     - Continuity correction applied for discrete-to-continuous approximation
 *     - Tie correction applied to variance estimate
 *     - EPS (1e-9) added to means to prevent division by zero in fold change
 *     - AUROC in [0, 1] represents P(group1 > group0) + 0.5 * P(ties)
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void mwu_test(
    const Sparse<T, IsCSR>& matrix, // Sparse matrix (features x samples)
    Array<const int32_t> group_ids, // Group labels for each sample
    Array<Real> out_u_stats,        // Output: U statistics [n_features]
    Array<Real> out_p_values,       // Output: p-values [n_features]
    Array<Real> out_log2_fc,        // Output: log2 fold change [n_features]
    Array<Real> out_auroc = Array<Real>()  // Optional: AUROC [n_features]
);

} // namespace scl::kernel::mwu
