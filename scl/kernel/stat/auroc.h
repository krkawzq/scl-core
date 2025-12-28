// =============================================================================
// FILE: scl/kernel/stat/auroc.h
// BRIEF: API reference for AUROC computation
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

namespace scl::kernel::stat::auroc {

/* -----------------------------------------------------------------------------
 * FUNCTION: count_groups
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Count elements in each of two groups.
 *
 * PARAMETERS:
 *     group_ids  [in]  Array of group assignments (0 or 1)
 *     out_n1     [out] Count of elements with group_id == 0
 *     out_n2     [out] Count of elements with group_id == 1
 *
 * COMPLEXITY:
 *     Time:  O(n) with SIMD optimization
 *     Space: O(1)
 * -------------------------------------------------------------------------- */
void count_groups(
    Array<const int32_t> group_ids,         // Group assignments
    Size& out_n1,                           // Group 0 count
    Size& out_n2                            // Group 1 count
);

/* -----------------------------------------------------------------------------
 * FUNCTION: auroc
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute AUROC (Area Under ROC Curve) for each feature.
 *
 * PARAMETERS:
 *     matrix       [in]  Sparse matrix (features x samples or samples x features)
 *     group_ids    [in]  Binary group assignment for each sample (0 or 1)
 *     out_auroc    [out] AUROC values, size = primary_dim
 *     out_p_values [out] Two-sided p-values (MWU normal approximation)
 *
 * PRECONDITIONS:
 *     - matrix is valid CSR/CSC sparse matrix
 *     - group_ids.len == matrix.secondary_dim()
 *     - out_auroc.len >= matrix.primary_dim()
 *     - out_p_values.len >= matrix.primary_dim()
 *     - Both groups have at least one member
 *
 * POSTCONDITIONS:
 *     - out_auroc[i] = P(X_group1 > X_group0) + 0.5 * P(X_group1 == X_group0)
 *     - out_auroc[i] in [0, 1], 0.5 indicates no discrimination
 *     - out_p_values[i] = two-sided MWU p-value
 *
 * ALGORITHM:
 *     For each feature in parallel:
 *     1. Partition non-zero values by group
 *     2. Sort each group (VQSort)
 *     3. Compute rank sum using merge with tie handling
 *     4. Compute U = R1 - n1*(n1+1)/2
 *     5. AUROC = U / (n1 * n2)
 *     6. P-value via normal approximation with tie correction
 *
 * COMPLEXITY:
 *     Time:  O(features * nnz_per_row * log(nnz_per_row))
 *     Space: O(threads * max_row_length)
 *
 * THREAD SAFETY:
 *     Safe - parallelized over features, no shared mutable state
 *
 * THROWS:
 *     ArgumentError - if either group is empty
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void auroc(
    const Sparse<T, IsCSR>& matrix,         // Input sparse matrix
    Array<const int32_t> group_ids,        // Binary group assignment
    Array<Real> out_auroc,                  // [n_features] AUROC values
    Array<Real> out_p_values                // [n_features] P-values
);

/* -----------------------------------------------------------------------------
 * FUNCTION: auroc_with_fc
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute AUROC with log2 fold change.
 *
 * PARAMETERS:
 *     matrix       [in]  Sparse matrix
 *     group_ids    [in]  Binary group assignment (0 or 1)
 *     out_auroc    [out] AUROC values
 *     out_p_values [out] P-values
 *     out_log2_fc  [out] Log2 fold change (group1 / group0)
 *
 * PRECONDITIONS:
 *     - Same as auroc()
 *     - out_log2_fc.len >= matrix.primary_dim()
 *
 * POSTCONDITIONS:
 *     - out_log2_fc[i] = log2((mean_group1 + eps) / (mean_group0 + eps))
 *     - eps = 1e-9 for numerical stability
 *
 * NOTE:
 *     This is more efficient than calling auroc() and computing FC separately,
 *     as sums are accumulated during partitioning.
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void auroc_with_fc(
    const Sparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Array<Real> out_auroc,
    Array<Real> out_p_values,
    Array<Real> out_log2_fc
);

} // namespace scl::kernel::stat::auroc
