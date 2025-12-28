// =============================================================================
// FILE: scl/kernel/stat/rank_utils.h
// BRIEF: API reference for ranking utilities
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

namespace scl::kernel::stat::rank {

/* -----------------------------------------------------------------------------
 * FUNCTION: find_negative_boundary
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Find the index of the first non-negative element in a sorted array.
 *
 * PARAMETERS:
 *     arr      [in]  Pointer to sorted array
 *     n        [in]  Array length
 *
 * PRECONDITIONS:
 *     - Array must be sorted in ascending order
 *
 * POSTCONDITIONS:
 *     - Returns index k such that arr[k] >= 0 and arr[k-1] < 0
 *     - Returns 0 if arr[0] >= 0
 *     - Returns n if all elements are negative
 *
 * ALGORITHM:
 *     - Fast path: check endpoints first
 *     - n >= 32: binary search (O(log n))
 *     - n < 32: 4-way unrolled linear scan
 *
 * COMPLEXITY:
 *     Time:  O(log n) for large n, O(n) for small n
 *     Space: O(1)
 * -------------------------------------------------------------------------- */
template <typename T>
Size find_negative_boundary(const T* arr, Size n);

/* -----------------------------------------------------------------------------
 * FUNCTION: merge_with_ties
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Merge two sorted arrays while computing rank sums and tie corrections.
 *
 * PARAMETERS:
 *     a        [in]      First sorted array
 *     pa       [in,out]  Current position in array a
 *     pa_end   [in]      End position for array a
 *     b        [in]      Second sorted array
 *     pb       [in,out]  Current position in array b
 *     pb_end   [in]      End position for array b
 *     rank     [in,out]  Current rank counter
 *     R1       [in,out]  Accumulated rank sum for group 1
 *     tie_sum  [in,out]  Accumulated tie correction term
 *
 * PRECONDITIONS:
 *     - Arrays a and b are sorted in ascending order
 *     - pa <= pa_end, pb <= pb_end
 *
 * POSTCONDITIONS:
 *     - pa == pa_end and pb == pb_end after completion
 *     - R1 contains sum of ranks for elements from array a
 *     - tie_sum contains sum(t^3 - t) for all tie groups of size t
 *
 * ALGORITHM:
 *     Merge sort style iteration:
 *     1. Select minimum value from both arrays
 *     2. Count ties in both arrays at this value
 *     3. Compute average rank for tie group
 *     4. Accumulate rank sum and tie correction
 *
 * COMPLEXITY:
 *     Time:  O(pa_end - pa + pb_end - pb)
 *     Space: O(1)
 * -------------------------------------------------------------------------- */
template <typename T>
void merge_with_ties(
    const T* a, Size& pa, Size pa_end,    // First array and range
    const T* b, Size& pb, Size pb_end,    // Second array and range
    Size& rank,                            // Current rank (1-indexed)
    double& R1,                            // Rank sum for group 1
    double& tie_sum                        // Tie correction accumulator
);

/* -----------------------------------------------------------------------------
 * FUNCTION: merge_for_ks
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Merge two sorted arrays while tracking maximum ECDF difference.
 *
 * PARAMETERS:
 *     a        [in]  First sorted array
 *     na       [in]  Length of array a
 *     b        [in]  Second sorted array
 *     nb       [in]  Length of array b
 *     inv_n1   [in]  Precomputed 1/n1 for ECDF computation
 *     inv_n2   [in]  Precomputed 1/n2 for ECDF computation
 *     out_D    [out] Maximum absolute difference |F1(x) - F2(x)|
 *
 * PRECONDITIONS:
 *     - Arrays a and b are sorted in ascending order
 *     - inv_n1 and inv_n2 are valid reciprocals
 *
 * POSTCONDITIONS:
 *     - out_D contains the KS test statistic D
 *
 * ALGORITHM:
 *     Merge iteration tracking empirical CDFs:
 *     1. Select minimum value
 *     2. Advance both pointers past ties
 *     3. Update F1 = count_a / n1, F2 = count_b / n2
 *     4. Track maximum |F1 - F2|
 *
 * COMPLEXITY:
 *     Time:  O(na + nb)
 *     Space: O(1)
 * -------------------------------------------------------------------------- */
template <typename T>
void merge_for_ks(
    const T* a, Size na,                   // First array
    const T* b, Size nb,                   // Second array
    double inv_n1, double inv_n2,          // Reciprocals for ECDF
    double& out_D                          // KS statistic output
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_rank_sum_sparse
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute rank sum and tie correction for sparse data with explicit zeros.
 *
 * PARAMETERS:
 *     a           [in]  Sorted non-zero values from group 1
 *     na_nz       [in]  Number of non-zero elements in group 1
 *     n1_total    [in]  Total size of group 1 (including zeros)
 *     b           [in]  Sorted non-zero values from group 2
 *     nb_nz       [in]  Number of non-zero elements in group 2
 *     n2_total    [in]  Total size of group 2 (including zeros)
 *     out_R1      [out] Rank sum for group 1
 *     out_tie_sum [out] Tie correction term
 *
 * PRECONDITIONS:
 *     - Arrays a and b are sorted in ascending order
 *     - na_nz <= n1_total, nb_nz <= n2_total
 *
 * POSTCONDITIONS:
 *     - out_R1 contains complete rank sum for group 1
 *     - out_tie_sum contains sum(t^3 - t) for all ties including zeros
 *
 * ALGORITHM:
 *     1. Find negative boundaries in both arrays
 *     2. Merge negative values
 *     3. Handle zero values (n1_total - na_nz from group 1, etc.)
 *     4. Merge positive values
 *
 * COMPLEXITY:
 *     Time:  O(na_nz + nb_nz)
 *     Space: O(1)
 * -------------------------------------------------------------------------- */
template <typename T>
void compute_rank_sum_sparse(
    const T* a, Size na_nz, Size n1_total, // Group 1 data
    const T* b, Size nb_nz, Size n2_total, // Group 2 data
    double& out_R1,                         // Rank sum output
    double& out_tie_sum                     // Tie correction output
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_rank_sums_k_groups
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute rank sums for k groups (Kruskal-Wallis).
 *
 * PARAMETERS:
 *     sorted_values      [in]  All values sorted in ascending order
 *     group_assignments  [in]  Group ID for each sorted value (0 to k-1)
 *     n_total            [in]  Total number of values
 *     n_groups           [in]  Number of groups (k)
 *     out_rank_sums      [out] Array of size k, rank sum for each group
 *     out_tie_sum        [out] Tie correction term
 *
 * PRECONDITIONS:
 *     - sorted_values is sorted in ascending order
 *     - group_assignments[i] corresponds to sorted_values[i]
 *     - out_rank_sums has at least n_groups elements
 *
 * POSTCONDITIONS:
 *     - out_rank_sums[g] contains sum of ranks for group g
 *     - out_tie_sum contains sum(t^3 - t) for all ties
 *
 * COMPLEXITY:
 *     Time:  O(n_total)
 *     Space: O(1)
 * -------------------------------------------------------------------------- */
template <typename T>
void compute_rank_sums_k_groups(
    const T* sorted_values,                 // Sorted values
    const Size* group_assignments,          // Group for each value
    Size n_total,                           // Total count
    Size n_groups,                          // Number of groups
    double* out_rank_sums,                  // [k] rank sums per group
    double& out_tie_sum                     // Tie correction
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_u_and_pvalue
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Mann-Whitney U statistic and p-value from rank sum.
 *
 * PARAMETERS:
 *     R1       [in]  Rank sum for group 1
 *     tie_sum  [in]  Tie correction term
 *     c        [in]  Precomputed MWU constants
 *     out_u    [out] U statistic
 *     out_pval [out] Two-sided p-value
 *
 * ALGORITHM:
 *     1. U = R1 - n1*(n1+1)/2
 *     2. Compute tie-corrected variance
 *     3. Apply continuity correction
 *     4. Normal approximation for p-value
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1)
 * -------------------------------------------------------------------------- */
void compute_u_and_pvalue(
    double R1,                              // Rank sum
    double tie_sum,                         // Tie correction
    const MWUConstants& c,                  // Constants
    Real& out_u,                            // U statistic
    Real& out_pval                          // P-value
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_auroc
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute AUROC from Mann-Whitney U statistic.
 *
 * PARAMETERS:
 *     U        [in]  Mann-Whitney U statistic
 *     n1       [in]  Size of group 1
 *     n2       [in]  Size of group 2
 *
 * RETURNS:
 *     AUROC = U / (n1 * n2), or 0.5 if n1*n2 == 0
 *
 * NOTE:
 *     AUROC represents the probability that a random element from
 *     group 2 is greater than a random element from group 1.
 * -------------------------------------------------------------------------- */
Real compute_auroc(double U, double n1, double n2);

} // namespace scl::kernel::stat::rank
