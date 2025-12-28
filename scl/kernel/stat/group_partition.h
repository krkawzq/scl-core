// =============================================================================
// FILE: scl/kernel/stat/group_partition.h
// BRIEF: API reference for group partitioning utilities
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

namespace scl::kernel::stat::partition {

/* -----------------------------------------------------------------------------
 * FUNCTION: partition_two_groups
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Partition sparse row/column values into two groups with sum accumulation.
 *
 * PARAMETERS:
 *     values     [in]  Sparse values array
 *     indices    [in]  Sparse indices array (column/row indices)
 *     len        [in]  Length of sparse arrays
 *     group_ids  [in]  Group assignment for each index (0 or 1)
 *     buf1       [out] Buffer for group 0 values
 *     n1         [out] Count of elements in group 0
 *     buf2       [out] Buffer for group 1 values
 *     n2         [out] Count of elements in group 1
 *     sum1       [out] Sum of values in group 0
 *     sum2       [out] Sum of values in group 1
 *
 * PRECONDITIONS:
 *     - buf1 and buf2 have sufficient capacity (>= len)
 *     - group_ids[indices[k]] is 0 or 1 for valid elements
 *
 * POSTCONDITIONS:
 *     - buf1[0:n1] contains values where group_ids == 0
 *     - buf2[0:n2] contains values where group_ids == 1
 *     - sum1 = sum of buf1, sum2 = sum of buf2
 *
 * ALGORITHM:
 *     4-way unrolled iteration with prefetch for indirect access.
 *
 * COMPLEXITY:
 *     Time:  O(len)
 *     Space: O(1) auxiliary
 * -------------------------------------------------------------------------- */
template <typename T>
void partition_two_groups(
    const T* values,                        // Sparse values
    const Index* indices,                   // Sparse indices
    Size len,                               // Length
    const int32_t* group_ids,              // Group assignment
    T* buf1, Size& n1,                     // Group 0 output
    T* buf2, Size& n2,                     // Group 1 output
    double& sum1, double& sum2             // Sums
);

/* -----------------------------------------------------------------------------
 * FUNCTION: partition_two_groups_moments
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Partition with sum and sum-of-squares accumulation for t-test.
 *
 * PARAMETERS:
 *     values     [in]  Sparse values array
 *     indices    [in]  Sparse indices array
 *     len        [in]  Length of sparse arrays
 *     group_ids  [in]  Group assignment (0 or 1)
 *     buf1       [out] Buffer for group 0 values
 *     n1         [out] Count in group 0
 *     buf2       [out] Buffer for group 1 values
 *     n2         [out] Count in group 1
 *     sum1       [out] Sum of group 0
 *     sum_sq1    [out] Sum of squares of group 0
 *     sum2       [out] Sum of group 1
 *     sum_sq2    [out] Sum of squares of group 1
 *
 * POSTCONDITIONS:
 *     - All sums and sum-of-squares computed in single pass
 *     - Enables online variance computation
 *
 * COMPLEXITY:
 *     Time:  O(len)
 *     Space: O(1) auxiliary
 * -------------------------------------------------------------------------- */
template <typename T>
void partition_two_groups_moments(
    const T* values,
    const Index* indices,
    Size len,
    const int32_t* group_ids,
    T* buf1, Size& n1,
    T* buf2, Size& n2,
    double& sum1, double& sum_sq1,
    double& sum2, double& sum_sq2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: partition_two_groups_simple
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Simple partition without sum accumulation.
 *
 * PARAMETERS:
 *     values     [in]  Sparse values array
 *     indices    [in]  Sparse indices array
 *     len        [in]  Length of sparse arrays
 *     group_ids  [in]  Group assignment (0 or 1)
 *     buf1       [out] Buffer for group 0 values
 *     n1         [out] Count in group 0
 *     buf2       [out] Buffer for group 1 values
 *     n2         [out] Count in group 1
 *
 * COMPLEXITY:
 *     Time:  O(len)
 *     Space: O(1) auxiliary
 * -------------------------------------------------------------------------- */
template <typename T>
void partition_two_groups_simple(
    const T* values,
    const Index* indices,
    Size len,
    const int32_t* group_ids,
    T* buf1, Size& n1,
    T* buf2, Size& n2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: partition_k_groups_moments
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Partition into k groups with moment accumulation for ANOVA.
 *
 * PARAMETERS:
 *     values     [in]  Sparse values array
 *     indices    [in]  Sparse indices array
 *     len        [in]  Length of sparse arrays
 *     group_ids  [in]  Group assignment (0 to k-1)
 *     n_groups   [in]  Number of groups (k)
 *     counts     [out] Array[k] of element counts per group
 *     sums       [out] Array[k] of sums per group
 *     sum_sqs    [out] Array[k] of sum-of-squares per group
 *
 * PRECONDITIONS:
 *     - counts, sums, sum_sqs have at least n_groups elements
 *     - group_ids values are in [0, n_groups) or negative (ignored)
 *
 * POSTCONDITIONS:
 *     - counts[g] = number of elements in group g
 *     - sums[g] = sum of values in group g
 *     - sum_sqs[g] = sum of squared values in group g
 *
 * COMPLEXITY:
 *     Time:  O(len)
 *     Space: O(1) auxiliary
 * -------------------------------------------------------------------------- */
template <typename T>
void partition_k_groups_moments(
    const T* values,
    const Index* indices,
    Size len,
    const int32_t* group_ids,
    Size n_groups,
    Size* counts,                           // [k] counts
    double* sums,                           // [k] sums
    double* sum_sqs                         // [k] sum-of-squares
);

/* -----------------------------------------------------------------------------
 * FUNCTION: partition_k_groups_to_buffer
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Partition all values into a single buffer with group tags.
 *
 * PARAMETERS:
 *     values      [in]  Sparse values array
 *     indices     [in]  Sparse indices array
 *     len         [in]  Length of sparse arrays
 *     group_ids   [in]  Group assignment (0 to k-1)
 *     n_groups    [in]  Number of groups
 *     out_values  [out] Output values buffer
 *     out_groups  [out] Output group assignment for each value
 *     out_total   [out] Total number of valid elements
 *
 * POSTCONDITIONS:
 *     - out_values[i] and out_groups[i] correspond for i in [0, out_total)
 *     - Ready for sorting and rank computation
 *
 * COMPLEXITY:
 *     Time:  O(len)
 *     Space: O(1) auxiliary
 * -------------------------------------------------------------------------- */
template <typename T>
void partition_k_groups_to_buffer(
    const T* values,
    const Index* indices,
    Size len,
    const int32_t* group_ids,
    Size n_groups,
    T* out_values,
    Size* out_groups,
    Size& out_total
);

/* -----------------------------------------------------------------------------
 * FUNCTION: finalize_group_stats
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute mean and variance from accumulated moments.
 *
 * PARAMETERS:
 *     count      [in]  Number of non-zero elements
 *     sum        [in]  Sum of non-zero elements
 *     sum_sq     [in]  Sum of squares of non-zero elements
 *     n_total    [in]  Total group size (including zeros)
 *     out_mean   [out] Computed mean (sum / n_total)
 *     out_var    [out] Computed variance with Bessel correction
 *     ddof       [in]  Delta degrees of freedom (default 1)
 *
 * POSTCONDITIONS:
 *     - out_mean includes zeros in denominator
 *     - out_var uses (n_total - ddof) in denominator
 *     - out_var >= 0 (clamped for numerical stability)
 *
 * ALGORITHM:
 *     - Mean: sum / n_total
 *     - Variance: adjusted for implicit zeros in sparse data
 * -------------------------------------------------------------------------- */
void finalize_group_stats(
    Size count,                             // Non-zero count
    double sum,                             // Sum
    double sum_sq,                          // Sum of squares
    Size n_total,                           // Total size
    double& out_mean,                       // Output mean
    double& out_var,                        // Output variance
    int ddof = 1                            // Degrees of freedom
);

} // namespace scl::kernel::stat::partition
