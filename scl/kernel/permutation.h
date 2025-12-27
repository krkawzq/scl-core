// =============================================================================
// FILE: scl/kernel/permutation.h
// BRIEF: API reference for permutation testing and multiple comparison kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.h"
#include "scl/core/sparse.h"

namespace scl::kernel::permutation {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

/* -----------------------------------------------------------------------------
 * NAMESPACE: config
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Configuration constants for permutation testing.
 *
 * CONSTANTS:
 *     DEFAULT_N_PERMUTATIONS - Default number of permutations (1000)
 *     MIN_PERMUTATIONS       - Minimum allowed permutations (100)
 *     MAX_PERMUTATIONS       - Maximum allowed permutations (100000)
 *     PARALLEL_THRESHOLD     - Min rows for parallel batch test (500)
 *
 * USAGE:
 *     More permutations increase precision but also computation time.
 *     For p-value resolution of 0.001, use at least 1000 permutations.
 * -------------------------------------------------------------------------- */

namespace config {
    constexpr Size DEFAULT_N_PERMUTATIONS = 1000;
    constexpr Size MIN_PERMUTATIONS = 100;
    constexpr Size MAX_PERMUTATIONS = 100000;
    constexpr Size PARALLEL_THRESHOLD = 500;
}

// =============================================================================
// SECTION 2: Generic Permutation Test
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: permutation_test
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Generic permutation test with user-defined test statistic.
 *
 * PARAMETERS:
 *     compute_statistic    [in] Callable: Array<const Index> -> Real
 *     labels               [in] Group labels to permute, size = n
 *     observed_statistic   [in] Observed value of test statistic
 *     n_permutations       [in] Number of permutations (default: 1000)
 *     two_sided            [in] Use two-sided test (default: true)
 *     seed                 [in] Random seed for reproducibility
 *
 * RETURNS:
 *     P-value: proportion of permuted statistics >= observed (or extreme)
 *
 * PRECONDITIONS:
 *     - labels.len > 0
 *     - compute_statistic must be callable with Array<const Index>
 *
 * POSTCONDITIONS:
 *     - Returns p-value in range [1/(n_perm+1), 1]
 *     - labels array is unchanged
 *
 * ALGORITHM:
 *     1. Copy labels to permutation buffer
 *     2. For each permutation:
 *        a. Shuffle using Fisher-Yates algorithm
 *        b. Compute test statistic on shuffled labels
 *        c. Store in null distribution
 *     3. Compute p-value from null distribution
 *
 * COMPLEXITY:
 *     Time:  O(n_permutations * (n + cost of compute_statistic))
 *     Space: O(n_permutations + n)
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 *
 * NUMERICAL NOTES:
 *     Uses (count + 1) / (n_perm + 1) to avoid zero p-values.
 * -------------------------------------------------------------------------- */
template <typename StatFunc>
Real permutation_test(
    StatFunc&& compute_statistic,      // Test statistic function
    Array<Index> labels,                // Group labels to permute
    Real observed_statistic,            // Observed test statistic
    Size n_permutations = config::DEFAULT_N_PERMUTATIONS,
    bool two_sided = true,              // Two-sided test
    uint64_t seed = 42                  // Random seed
);

// =============================================================================
// SECTION 3: Correlation Permutation Test
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: permutation_correlation_test
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Permutation test for Pearson correlation significance.
 *
 * COMPUTES:
 *     P-value for H0: rho = 0 vs H1: rho != 0 (two-sided)
 *
 * PARAMETERS:
 *     x                    [in] First variable, size = n
 *     y                    [in] Second variable, size = n
 *     observed_correlation [in] Observed Pearson correlation
 *     n_permutations       [in] Number of permutations
 *     seed                 [in] Random seed
 *
 * RETURNS:
 *     Two-sided p-value for correlation test
 *
 * PRECONDITIONS:
 *     - x.len == y.len
 *     - x.len >= 3 (need at least 3 points for correlation)
 *
 * POSTCONDITIONS:
 *     - Returns p-value in [1/(n_perm+1), 1]
 *     - x and y are unchanged
 *
 * ALGORITHM:
 *     1. Precompute mean and std of x (constant across permutations)
 *     2. For each permutation:
 *        a. Shuffle indices
 *        b. Compute correlation with permuted y
 *        c. Store in null distribution
 *     3. Compare |observed| to |null| for two-sided test
 *
 * COMPLEXITY:
 *     Time:  O(n_permutations * n)
 *     Space: O(n_permutations + n)
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 * -------------------------------------------------------------------------- */
Real permutation_correlation_test(
    Array<const Real> x,               // First variable
    Array<const Real> y,               // Second variable
    Real observed_correlation,          // Observed correlation
    Size n_permutations = config::DEFAULT_N_PERMUTATIONS,
    uint64_t seed = 42                  // Random seed
);

// =============================================================================
// SECTION 4: FDR Correction Methods
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: fdr_correction_bh
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Benjamini-Hochberg FDR correction for multiple testing.
 *
 * COMPUTES:
 *     q_i = min(p_(i) * n / i, 1), with cumulative minimum from end
 *
 * PARAMETERS:
 *     p_values  [in]  Raw p-values, size = n
 *     q_values  [out] Adjusted q-values (FDR), size = n, PRE-ALLOCATED
 *
 * PRECONDITIONS:
 *     - q_values.len >= p_values.len
 *     - p_values in [0, 1]
 *
 * POSTCONDITIONS:
 *     - q_values[i] contains FDR-adjusted p-value for test i
 *     - q_values[i] >= p_values[i]
 *     - q_values[i] in [0, 1]
 *
 * ALGORITHM:
 *     1. Sort p-values and get order
 *     2. From largest to smallest rank:
 *        adjusted = p * n / rank
 *        q = min(adjusted, cumulative_min)
 *     3. Map back to original order
 *
 * COMPLEXITY:
 *     Time:  O(n log n) for sorting
 *     Space: O(n) for sorting indices
 *
 * THREAD SAFETY:
 *     Safe - parallelizable internally
 *
 * REFERENCE:
 *     Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery
 *     rate: a practical and powerful approach to multiple testing.
 * -------------------------------------------------------------------------- */
void fdr_correction_bh(
    Array<const Real> p_values,        // Raw p-values [n]
    Array<Real> q_values               // Output FDR q-values [n], PRE-ALLOCATED
);

/* -----------------------------------------------------------------------------
 * FUNCTION: fdr_correction_by
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Benjamini-Yekutieli FDR correction for dependent tests.
 *
 * COMPUTES:
 *     q_i = min(p_(i) * c_n * n / i, 1), where c_n = sum(1/k) for k=1..n
 *
 * PARAMETERS:
 *     p_values  [in]  Raw p-values, size = n
 *     q_values  [out] Adjusted q-values, size = n, PRE-ALLOCATED
 *
 * PRECONDITIONS:
 *     - q_values.len >= p_values.len
 *     - p_values in [0, 1]
 *
 * POSTCONDITIONS:
 *     - q_values controls FDR under arbitrary dependence
 *     - More conservative than BH correction
 *
 * ALGORITHM:
 *     Same as BH, but multiplied by harmonic sum c_n = 1 + 1/2 + ... + 1/n
 *
 * COMPLEXITY:
 *     Time:  O(n log n)
 *     Space: O(n)
 *
 * REFERENCE:
 *     Benjamini, Y. and Yekutieli, D. (2001). The control of the false
 *     discovery rate in multiple testing under dependency.
 * -------------------------------------------------------------------------- */
void fdr_correction_by(
    Array<const Real> p_values,        // Raw p-values [n]
    Array<Real> q_values               // Output q-values [n], PRE-ALLOCATED
);

// =============================================================================
// SECTION 5: Family-Wise Error Rate Corrections
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: bonferroni_correction
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Bonferroni correction for multiple testing (FWER control).
 *
 * COMPUTES:
 *     adjusted_p_i = min(p_i * n, 1)
 *
 * PARAMETERS:
 *     p_values           [in]  Raw p-values, size = n
 *     adjusted_p_values  [out] Bonferroni-adjusted p-values, PRE-ALLOCATED
 *
 * PRECONDITIONS:
 *     - adjusted_p_values.len >= p_values.len
 *
 * POSTCONDITIONS:
 *     - adjusted_p_values[i] = min(p_values[i] * n, 1)
 *     - Controls FWER at alpha if rejecting when adjusted_p < alpha
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * NOTES:
 *     Very conservative - use FDR methods when many tests expected to be true.
 * -------------------------------------------------------------------------- */
void bonferroni_correction(
    Array<const Real> p_values,        // Raw p-values [n]
    Array<Real> adjusted_p_values      // Output adjusted p-values [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: holm_correction
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Holm-Bonferroni step-down correction (less conservative than Bonferroni).
 *
 * COMPUTES:
 *     adjusted_p_(i) = max(p_(j) * (n - j + 1)) for j <= i
 *
 * PARAMETERS:
 *     p_values           [in]  Raw p-values, size = n
 *     adjusted_p_values  [out] Holm-adjusted p-values, PRE-ALLOCATED
 *
 * PRECONDITIONS:
 *     - adjusted_p_values.len >= p_values.len
 *
 * POSTCONDITIONS:
 *     - Controls FWER at alpha
 *     - Uniformly more powerful than Bonferroni
 *
 * ALGORITHM:
 *     1. Sort p-values ascending
 *     2. For i = 1 to n:
 *        adjusted = p_(i) * (n - i + 1)
 *        result_(i) = max(adjusted, result_(i-1))
 *     3. Map back to original order
 *
 * COMPLEXITY:
 *     Time:  O(n log n)
 *     Space: O(n)
 *
 * REFERENCE:
 *     Holm, S. (1979). A simple sequentially rejective multiple test procedure.
 * -------------------------------------------------------------------------- */
void holm_correction(
    Array<const Real> p_values,        // Raw p-values [n]
    Array<Real> adjusted_p_values      // Output adjusted p-values [n]
);

// =============================================================================
// SECTION 6: Utility Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: count_significant
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Count number of p-values below significance threshold.
 *
 * PARAMETERS:
 *     p_values [in] P-values to test, size = n
 *     alpha    [in] Significance threshold (default: 0.05)
 *
 * RETURNS:
 *     Number of p-values < alpha
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1)
 * -------------------------------------------------------------------------- */
Size count_significant(
    Array<const Real> p_values,        // P-values [n]
    Real alpha = Real(0.05)            // Significance threshold
);

/* -----------------------------------------------------------------------------
 * FUNCTION: get_significant_indices
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Get indices of significant tests.
 *
 * PARAMETERS:
 *     p_values      [in]  P-values, size = n
 *     alpha         [in]  Significance threshold
 *     indices       [out] Output buffer for significant indices, PRE-ALLOCATED
 *     n_significant [out] Number of significant tests found
 *
 * PRECONDITIONS:
 *     - indices.len should be >= expected number of significant tests
 *
 * POSTCONDITIONS:
 *     - indices[0..n_significant-1] contain indices where p < alpha
 *     - n_significant <= indices.len (truncated if buffer too small)
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 * -------------------------------------------------------------------------- */
void get_significant_indices(
    Array<const Real> p_values,        // P-values [n]
    Real alpha,                         // Significance threshold
    Array<Index> indices,               // Output buffer for indices
    Size& n_significant                 // Number of significant found
);

// =============================================================================
// SECTION 7: Batch Permutation Test
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: batch_permutation_test
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Parallel permutation test for multiple features (rows of sparse matrix).
 *
 * COMPUTES:
 *     For each row: p-value for mean difference between groups
 *
 * PARAMETERS:
 *     matrix         [in]  CSR sparse matrix, shape (n_features x n_samples)
 *     group_labels   [in]  Group assignments (0/1), size = n_samples
 *     n_permutations [in]  Number of permutations per feature
 *     p_values       [out] Output p-values, size = n_features, PRE-ALLOCATED
 *     seed           [in]  Random seed
 *
 * PRECONDITIONS:
 *     - matrix must be CSR format (IsCSR = true)
 *     - group_labels.len >= matrix.cols()
 *     - p_values.len >= matrix.rows()
 *     - group_labels contain only 0 and 1
 *
 * POSTCONDITIONS:
 *     - p_values[i] contains two-sided p-value for row i
 *     - Rows with no non-zeros get p-value = 1.0
 *
 * ALGORITHM:
 *     Parallel over features:
 *     1. Compute observed mean difference for row
 *     2. For each permutation:
 *        a. Shuffle group labels (thread-local RNG)
 *        b. Compute permuted mean difference
 *     3. Compute two-sided p-value
 *
 * COMPLEXITY:
 *     Time:  O(n_features * n_permutations * avg_nnz_per_row)
 *     Space: O(n_threads * (n_permutations + n_samples))
 *
 * THREAD SAFETY:
 *     Safe - uses WorkspacePool for thread-local buffers
 *
 * PERFORMANCE:
 *     Uses WorkspacePool to avoid per-feature allocations.
 *     Each thread has independent RNG seeded from base seed + row index.
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void batch_permutation_test(
    const Sparse<T, IsCSR>& matrix,    // CSR input matrix (features x samples)
    Array<const Index> group_labels,    // Group labels [n_samples]
    Size n_permutations,                // Number of permutations
    Array<Real> p_values,               // Output p-values [n_features]
    uint64_t seed = 42                  // Random seed
);

} // namespace scl::kernel::permutation
