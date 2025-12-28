// =============================================================================
// FILE: scl/kernel/multiple_testing.h
// BRIEF: API reference for multiple testing correction methods
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include <vector>

namespace scl::kernel::multiple_testing {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_FDR_LEVEL = Real(0.05);
    constexpr Real DEFAULT_LAMBDA = Real(0.5);
    constexpr Real MIN_PVALUE = Real(1e-300);
    constexpr Real MAX_PVALUE = Real(1.0);
    constexpr Size SPLINE_KNOTS = 10;
    constexpr Size MIN_TESTS_FOR_STOREY = 100;
}

// =============================================================================
// FDR Correction Methods
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: benjamini_hochberg
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply Benjamini-Hochberg FDR correction to p-values.
 *
 * PARAMETERS:
 *     p_values          [in]  Input p-values [n_tests]
 *     adjusted_p_values [out] BH-adjusted p-values [n_tests]
 *     fdr_level         [in]  FDR control level (typically 0.05)
 *
 * PRECONDITIONS:
 *     - p_values.len == adjusted_p_values.len
 *     - All p-values in [0, 1]
 *
 * POSTCONDITIONS:
 *     - adjusted_p_values[i] contains BH-adjusted p-value
 *     - Adjusted values are monotonic and in [0, 1]
 *
 * ALGORITHM:
 *     1. Sort p-values in ascending order
 *     2. Compute adjusted p = p * n / rank
 *     3. Enforce monotonicity from right to left
 *     4. Map back to original order
 *
 * COMPLEXITY:
 *     Time:  O(n log n) for sorting
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
void benjamini_hochberg(
    Array<const Real> p_values,             // Input p-values [n_tests]
    Array<Real> adjusted_p_values,          // Output adjusted p-values [n_tests]
    Real fdr_level = config::DEFAULT_FDR_LEVEL // FDR control level
);

/* -----------------------------------------------------------------------------
 * FUNCTION: bonferroni
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply Bonferroni correction (multiply by number of tests).
 *
 * PARAMETERS:
 *     p_values          [in]  Input p-values [n_tests]
 *     adjusted_p_values [out] Bonferroni-adjusted p-values [n_tests]
 *
 * PRECONDITIONS:
 *     - p_values.len == adjusted_p_values.len
 *     - All p-values in [0, 1]
 *
 * POSTCONDITIONS:
 *     - adjusted_p_values[i] = min(p_values[i] * n, 1.0)
 *     - All adjusted values in [0, 1]
 *
 * ALGORITHM:
 *     1. Multiply each p-value by n (number of tests)
 *     2. Clamp to [0, 1]
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - uses SIMD-optimized operations
 * -------------------------------------------------------------------------- */
void bonferroni(
    Array<const Real> p_values,             // Input p-values [n_tests]
    Array<Real> adjusted_p_values            // Output adjusted p-values [n_tests]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: storey_qvalue
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Estimate q-values using Storey's method with pi0 estimation.
 *
 * PARAMETERS:
 *     p_values  [in]  Input p-values [n_tests]
 *     q_values  [out] Estimated q-values [n_tests]
 *     lambda    [in]  Tuning parameter for pi0 estimation (default 0.5)
 *
 * PRECONDITIONS:
 *     - p_values.len == q_values.len
 *     - All p-values in [0, 1]
 *     - lambda in (0, 1)
 *
 * POSTCONDITIONS:
 *     - q_values[i] contains estimated q-value for test i
 *     - All q-values in [0, 1]
 *
 * ALGORITHM:
 *     1. Estimate pi0 (proportion of true nulls) using lambda
 *     2. Sort p-values
 *     3. Compute q-values from right to left with monotonicity
 *
 * COMPLEXITY:
 *     Time:  O(n log n) for sorting
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
void storey_qvalue(
    Array<const Real> p_values,             // Input p-values [n_tests]
    Array<Real> q_values,                    // Output q-values [n_tests]
    Real lambda = config::DEFAULT_LAMBDA      // Pi0 estimation parameter
);

/* -----------------------------------------------------------------------------
 * FUNCTION: local_fdr
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Estimate local false discovery rate using kernel density estimation.
 *
 * PARAMETERS:
 *     p_values [in]  Input p-values [n_tests]
 *     lfdr     [out] Local FDR estimates [n_tests]
 *
 * PRECONDITIONS:
 *     - p_values.len == lfdr.len
 *     - All p-values in [0, 1]
 *
 * POSTCONDITIONS:
 *     - lfdr[i] contains local FDR estimate for test i
 *     - All lfdr values in [0, 1]
 *
 * ALGORITHM:
 *     1. Transform p-values to z-scores
 *     2. Estimate density f(z) using KDE
 *     3. Compute f0(z) (null density, standard normal)
 *     4. Estimate pi0
 *     5. Compute lfdr = pi0 * f0(z) / f(z)
 *
 * COMPLEXITY:
 *     Time:  O(n^2) for KDE estimation
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
void local_fdr(
    Array<const Real> p_values,             // Input p-values [n_tests]
    Array<Real> lfdr                          // Output local FDR [n_tests]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: empirical_fdr
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Estimate FDR using permutation-based empirical null distribution.
 *
 * PARAMETERS:
 *     observed_scores  [in]  Observed test statistics [n_tests]
 *     permuted_scores  [in]  Permuted test statistics [n_permutations][n_tests]
 *     fdr              [out] Empirical FDR estimates [n_tests]
 *
 * PRECONDITIONS:
 *     - observed_scores.len == fdr.len
 *     - All permuted_scores arrays have same length as observed_scores
 *
 * POSTCONDITIONS:
 *     - fdr[i] contains empirical FDR for test i
 *     - All fdr values in [0, 1]
 *
 * ALGORITHM:
 *     1. For each test, count permutations with score >= observed
 *     2. Compute FDR = (permutation_count + 1) / (n_permutations + 1)
 *
 * COMPLEXITY:
 *     Time:  O(n_tests * n_permutations)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over tests
 * -------------------------------------------------------------------------- */
void empirical_fdr(
    Array<const Real> observed_scores,      // Observed statistics [n_tests]
    const std::vector<Array<Real>>& permuted_scores, // Permuted statistics
    Array<Real> fdr                          // Output FDR [n_tests]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: benjamini_yekutieli
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply Benjamini-Yekutieli FDR correction (works under dependency).
 *
 * PARAMETERS:
 *     p_values          [in]  Input p-values [n_tests]
 *     adjusted_p_values [out] BY-adjusted p-values [n_tests]
 *
 * PRECONDITIONS:
 *     - p_values.len == adjusted_p_values.len
 *     - All p-values in [0, 1]
 *
 * POSTCONDITIONS:
 *     - adjusted_p_values[i] contains BY-adjusted p-value
 *     - More conservative than BH, works under arbitrary dependency
 *
 * ALGORITHM:
 *     Similar to BH but uses correction factor C(n) = sum(1/i) for i=1..n
 *
 * COMPLEXITY:
 *     Time:  O(n log n) for sorting
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
void benjamini_yekutieli(
    Array<const Real> p_values,             // Input p-values [n_tests]
    Array<Real> adjusted_p_values            // Output adjusted p-values [n_tests]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: holm_bonferroni
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply Holm-Bonferroni step-down correction.
 *
 * PARAMETERS:
 *     p_values          [in]  Input p-values [n_tests]
 *     adjusted_p_values [out] Holm-adjusted p-values [n_tests]
 *
 * PRECONDITIONS:
 *     - p_values.len == adjusted_p_values.len
 *     - All p-values in [0, 1]
 *
 * POSTCONDITIONS:
 *     - adjusted_p_values[i] contains Holm-adjusted p-value
 *     - More powerful than Bonferroni, controls FWER
 *
 * ALGORITHM:
 *     1. Sort p-values
 *     2. For rank i: adjusted = p * (n - i + 1)
 *     3. Enforce monotonicity
 *
 * COMPLEXITY:
 *     Time:  O(n log n) for sorting
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
void holm_bonferroni(
    Array<const Real> p_values,             // Input p-values [n_tests]
    Array<Real> adjusted_p_values            // Output adjusted p-values [n_tests]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: hochberg
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply Hochberg step-up correction (more powerful than Holm).
 *
 * PARAMETERS:
 *     p_values          [in]  Input p-values [n_tests]
 *     adjusted_p_values [out] Hochberg-adjusted p-values [n_tests]
 *
 * PRECONDITIONS:
 *     - p_values.len == adjusted_p_values.len
 *     - All p-values in [0, 1]
 *
 * POSTCONDITIONS:
 *     - adjusted_p_values[i] contains Hochberg-adjusted p-value
 *     - Controls FWER, more powerful than Holm
 *
 * ALGORITHM:
 *     1. Sort p-values
 *     2. For rank i: adjusted = p * (n - i + 1)
 *     3. Enforce monotonicity from right to left
 *
 * COMPLEXITY:
 *     Time:  O(n log n) for sorting
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
void hochberg(
    Array<const Real> p_values,             // Input p-values [n_tests]
    Array<Real> adjusted_p_values            // Output adjusted p-values [n_tests]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: significant_indices
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Get indices of tests with p-values below threshold.
 *
 * PARAMETERS:
 *     p_values   [in]  Input p-values [n_tests]
 *     threshold  [in]  Significance threshold
 *     out_indices [out] Output indices [n_tests] (may be partially filled)
 *     out_count  [out] Number of significant tests
 *
 * PRECONDITIONS:
 *     - out_indices has capacity >= n_tests
 *
 * POSTCONDITIONS:
 *     - out_indices[0..out_count) contains indices of significant tests
 *     - out_count <= n_tests
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
void significant_indices(
    Array<const Real> p_values,             // Input p-values [n_tests]
    Real threshold,                          // Significance threshold
    Index* out_indices,                      // Output indices [n_tests]
    Size& out_count                          // Output count
);

/* -----------------------------------------------------------------------------
 * FUNCTION: neglog10_pvalues
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute negative log10 of p-values for visualization.
 *
 * PARAMETERS:
 *     p_values [in]  Input p-values [n_tests]
 *     neglog_p [out] Negative log10 p-values [n_tests]
 *
 * PRECONDITIONS:
 *     - p_values.len == neglog_p.len
 *     - All p-values in [0, 1]
 *
 * POSTCONDITIONS:
 *     - neglog_p[i] = -log10(p_values[i])
 *     - Values are clamped to prevent overflow
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
void neglog10_pvalues(
    Array<const Real> p_values,             // Input p-values [n_tests]
    Array<Real> neglog_p                    // Output -log10(p) [n_tests]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: fisher_combine
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Combine p-values using Fisher's method.
 *
 * PARAMETERS:
 *     p_values [in]  Input p-values to combine [n]
 *
 * PRECONDITIONS:
 *     - All p-values in [0, 1]
 *
 * POSTCONDITIONS:
 *     - Returns chi-squared test statistic
 *     - Statistic follows chi2(2*n) under null
 *
 * ALGORITHM:
 *     Compute chi2 = -2 * sum(log(p_i))
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
Real fisher_combine(
    Array<const Real> p_values               // Input p-values [n]
);

} // namespace scl::kernel::multiple_testing

