// =============================================================================
// FILE: scl/kernel/stat/permutation_stat.h
// BRIEF: API reference for optimized permutation testing
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

namespace scl::kernel::stat::permutation_stat {

/* -----------------------------------------------------------------------------
 * ENUM: PermStatType
 * -----------------------------------------------------------------------------
 * VALUES:
 *     MWU      - Mann-Whitney U statistic
 *     MeanDiff - Mean difference (t-test like)
 *     KS       - Kolmogorov-Smirnov D statistic (future)
 * -------------------------------------------------------------------------- */
enum class PermStatType {
    MWU,
    MeanDiff,
    KS
};

/* -----------------------------------------------------------------------------
 * FUNCTION: batch_permutation_reuse_sort
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Batch permutation test optimized by reusing sorted data structure.
 *     Permutes group_ids instead of re-sorting data for each permutation.
 *
 * PARAMETERS:
 *     matrix         [in]  Sparse matrix (features x samples)
 *     group_ids      [in]  Binary group assignment (0 or 1)
 *     n_permutations [in]  Number of permutations (100-100000)
 *     out_p_values   [out] P-values, size = n_features
 *     stat_type      [in]  Statistic type (MWU, MeanDiff, KS)
 *     seed           [in]  Random seed for reproducibility
 *
 * PRECONDITIONS:
 *     - Both groups (0 and 1) have at least one member
 *     - Output array sized >= matrix.primary_dim()
 *     - n_permutations in [100, 100000]
 *
 * POSTCONDITIONS:
 *     - out_p_values[i] = empirical p-value for feature i
 *     - P-value = (count of |stat_perm| >= |stat_obs| + 1) / (n_perms + 1)
 *
 * ALGORITHM:
 *     For each feature in parallel:
 *     1. Extract non-zero values with indices
 *     2. Sort values ONCE (argsort preserving index mapping)
 *     3. Compute observed statistic using original group_ids
 *     4. For each permutation:
 *        a. Shuffle group_ids (O(n) Fisher-Yates)
 *        b. Recompute statistic using sorted data + shuffled groups
 *        c. Adaptive early stopping check every 100 permutations
 *     5. Compute two-sided p-value from null distribution
 *
 * KEY OPTIMIZATION:
 *     Standard approach: sort data for EACH permutation = O(P * n log n)
 *     This approach: sort ONCE, permute groups = O(n log n + P * n)
 *     Speedup factor: approximately log(n) for large n
 *
 * COMPLEXITY:
 *     Time:  O(features * (nnz * log(nnz) + n_permutations * nnz))
 *     Space: O(threads * (max_row_length + n_samples + n_permutations))
 *
 * THREAD SAFETY:
 *     Safe - parallelized over features with thread-local workspace and RNG
 *
 * THROWS:
 *     ArgumentError - if either group is empty
 *
 * NUMERICAL NOTES:
 *     - Uses Xoshiro256++ PRNG with jump() for parallel streams
 *     - Lemire's nearly divisionless bounded random
 *     - Adaptive early stopping if p < 0.001 or p > 0.5
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void batch_permutation_reuse_sort(
    const Sparse<T, IsCSR>& matrix,         // Input sparse matrix
    Array<const int32_t> group_ids,         // Binary group assignment
    Size n_permutations,                     // Number of permutations
    Array<Real> out_p_values,                // [n_features] P-values
    PermStatType stat_type = PermStatType::MWU,  // Statistic type
    uint64_t seed = 42                       // Random seed
);

/* -----------------------------------------------------------------------------
 * FUNCTION: permutation_test_single
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Single-feature permutation test with sort-reuse optimization.
 *
 * PARAMETERS:
 *     values         [in]  Feature values
 *     group_ids      [in]  Binary group assignment (0 or 1)
 *     n_permutations [in]  Number of permutations
 *     stat_type      [in]  Statistic type
 *     seed           [in]  Random seed
 *
 * PRECONDITIONS:
 *     - values.len == group_ids.len
 *     - Both groups have at least one member
 *
 * POSTCONDITIONS:
 *     - Returns empirical two-sided p-value
 *
 * ALGORITHM:
 *     Same as batch_permutation_reuse_sort but for single feature.
 *
 * COMPLEXITY:
 *     Time:  O(n * log(n) + n_permutations * n)
 *     Space: O(n + n_permutations)
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
template <typename T>
Real permutation_test_single(
    Array<const T> values,                   // Feature values
    Array<const int32_t> group_ids,          // Binary group assignment
    Size n_permutations,                     // Number of permutations
    PermStatType stat_type = PermStatType::MWU,  // Statistic type
    uint64_t seed = 42                       // Random seed
);

/* -----------------------------------------------------------------------------
 * INTERNAL: config::*
 * -----------------------------------------------------------------------------
 * CONSTANTS:
 *     DEFAULT_N_PERMUTATIONS = 1000
 *     MIN_PERMUTATIONS = 100
 *     MAX_PERMUTATIONS = 100000
 *     EARLY_CHECK_INTERVAL = 100
 *     EARLY_STOP_ALPHA = 0.001   (stop if p < this)
 *     EARLY_STOP_BETA = 0.5      (stop if p > this)
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::FastRNG
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Xoshiro256++ PRNG with jump() for parallel independent streams.
 *
 * METHODS:
 *     operator()  - Generate uint64
 *     bounded(n)  - Uniform random in [0, n) using Lemire's method
 *     jump()      - Advance state by 2^128 steps
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::shuffle_groups
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Fisher-Yates shuffle for group_ids array.
 *     4x unrolled for performance.
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::compute_U_from_sorted
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Mann-Whitney U from pre-sorted data.
 *
 * ALGORITHM:
 *     1. Walk through sorted values tracking ties
 *     2. For each value, compute average rank
 *     3. Accumulate rank sum for group 0
 *     4. U = R1 - n1*(n1+1)/2
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::compute_mean_diff_from_sorted
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute mean difference (group0 - group1) from data.
 *
 * ALGORITHM:
 *     1. Accumulate sums and counts for each group
 *     2. Return mean1 - mean2
 * -------------------------------------------------------------------------- */

} // namespace scl::kernel::stat::permutation_stat
