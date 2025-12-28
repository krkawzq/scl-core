// =============================================================================
// FILE: scl/kernel/stat/ks.h
// BRIEF: API reference for Kolmogorov-Smirnov test
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

namespace scl::kernel::stat::ks {

/* -----------------------------------------------------------------------------
 * FUNCTION: ks_test
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute two-sample Kolmogorov-Smirnov test for each feature.
 *
 * PARAMETERS:
 *     matrix       [in]  Sparse matrix (features x samples)
 *     group_ids    [in]  Binary group assignment (0 or 1)
 *     out_D_stats  [out] KS D statistics, size = n_features
 *     out_p_values [out] P-values, size = n_features
 *
 * PRECONDITIONS:
 *     - matrix.secondary_dim() == group_ids.len
 *     - Output arrays have size >= matrix.primary_dim()
 *     - Both groups have at least one member
 *
 * POSTCONDITIONS:
 *     - out_D_stats[i] = max |F1(x) - F2(x)| for feature i
 *     - out_p_values[i] = Kolmogorov distribution p-value
 *
 * ALGORITHM:
 *     For each feature in parallel:
 *     1. Partition non-zero values by group
 *     2. Sort each group (VQSort)
 *     3. Merge sorted arrays tracking ECDF difference
 *        - Handle sparse zeros explicitly in ECDF
 *     4. Compute D = max |F1(x) - F2(x)|
 *     5. Compute p-value via Kolmogorov distribution
 *
 *     ECDF handling for sparse data:
 *     - Values < 0: contribute to ECDF before zero point
 *     - Zeros (implicit): jump at x=0
 *     - Values > 0: contribute to ECDF after zero point
 *
 * COMPLEXITY:
 *     Time:  O(features * nnz_per_feature * log(nnz_per_feature))
 *     Space: O(threads * max_row_length)
 *
 * THREAD SAFETY:
 *     Safe - parallelized over features with thread-local workspace
 *
 * THROWS:
 *     ArgumentError - if either group is empty
 *
 * NUMERICAL NOTES:
 *     - P-value uses asymptotic Kolmogorov distribution
 *     - Accurate for n1, n2 >= 25
 *     - Uses series expansion with 100-term limit
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void ks_test(
    const Sparse<T, IsCSR>& matrix,         // Input sparse matrix
    Array<const int32_t> group_ids,        // Binary group assignment
    Array<Real> out_D_stats,                // [n_features] D statistics
    Array<Real> out_p_values                // [n_features] P-values
);

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::ks_pvalue
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute KS test p-value from D statistic.
 *
 * FORMULA:
 *     n_eff = n1 * n2 / (n1 + n2)
 *     lambda = (sqrt(n_eff) + 0.12 + 0.11/sqrt(n_eff)) * D
 *     P(D > d) = 2 * sum_{k=1}^inf (-1)^{k+1} * exp(-2*k^2*lambda^2)
 *
 * ACCURACY:
 *     Asymptotically exact; accurate for n >= 25
 * -------------------------------------------------------------------------- */

/* -----------------------------------------------------------------------------
 * INTERNAL: detail::compute_ks_sparse
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute KS D statistic for sparse data with explicit zeros.
 *
 * ALGORITHM:
 *     1. Find negative boundary in each sorted array
 *     2. Process negative values: update ECDF, track max diff
 *     3. Process zeros: account for implicit zeros in each group
 *     4. Process positive values: continue ECDF, track max diff
 * -------------------------------------------------------------------------- */

} // namespace scl::kernel::stat::ks
