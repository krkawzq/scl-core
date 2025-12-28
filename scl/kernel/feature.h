// =============================================================================
// FILE: scl/kernel/feature.h
// BRIEF: API reference for feature statistics kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::feature {

/* -----------------------------------------------------------------------------
 * FUNCTION: standard_moments
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute mean and variance for each primary dimension (row/column) of a
 *     sparse matrix, accounting for implicit zeros.
 *
 * PARAMETERS:
 *     matrix    [in]  Sparse matrix in CSR or CSC format
 *     out_means [out] Pre-allocated buffer for means, size = primary_dim
 *     out_vars  [out] Pre-allocated buffer for variances, size = primary_dim
 *     ddof      [in]  Delta degrees of freedom for variance (typically 0 or 1)
 *
 * PRECONDITIONS:
 *     - out_means.len >= matrix.primary_dim()
 *     - out_vars.len >= matrix.primary_dim()
 *     - matrix is valid sparse format
 *
 * POSTCONDITIONS:
 *     - out_means[i] = sum(row_i) / secondary_dim
 *     - out_vars[i] = var(row_i) with specified ddof, clamped to >= 0
 *     - Matrix unchanged
 *
 * ALGORITHM:
 *     For each primary index in parallel:
 *         1. Compute fused sum and sum-of-squares using SIMD
 *         2. Calculate mean = sum / N
 *         3. Calculate variance = (sumsq - sum * mean) / (N - ddof)
 *         4. Clamp variance to non-negative
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over primary dimension, no shared mutable state
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void standard_moments(
    const Sparse<T, IsCSR>& matrix,   // Input sparse matrix
    Array<Real> out_means,             // Output means [primary_dim]
    Array<Real> out_vars,              // Output variances [primary_dim]
    int ddof                           // Degrees of freedom adjustment
);

/* -----------------------------------------------------------------------------
 * FUNCTION: clipped_moments
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute mean and variance with per-row clipping of maximum values.
 *
 * PARAMETERS:
 *     matrix    [in]  Sparse matrix
 *     clip_vals [in]  Per-row clip thresholds, size = primary_dim
 *     out_means [out] Output means buffer
 *     out_vars  [out] Output variances buffer
 *
 * PRECONDITIONS:
 *     - All buffer sizes >= primary_dim
 *     - clip_vals contains positive values
 *
 * POSTCONDITIONS:
 *     - Values clipped to min(value, clip_vals[i]) before statistics
 *     - Variance computed with ddof=1
 *
 * ALGORITHM:
 *     For each primary index in parallel:
 *         1. Load values and clip to threshold using SIMD min
 *         2. Compute fused sum and sum-of-squares
 *         3. Calculate mean and variance
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over primary dimension
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void clipped_moments(
    const Sparse<T, IsCSR>& matrix,   // Input sparse matrix
    Array<const Real> clip_vals,       // Per-row clip thresholds [primary_dim]
    Array<Real> out_means,             // Output means [primary_dim]
    Array<Real> out_vars               // Output variances [primary_dim]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: detection_rate
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute the fraction of non-zero entries per primary dimension.
 *
 * PARAMETERS:
 *     matrix    [in]  Sparse matrix
 *     out_rates [out] Output buffer for detection rates
 *
 * PRECONDITIONS:
 *     - out_rates.len >= matrix.primary_dim()
 *
 * POSTCONDITIONS:
 *     - out_rates[i] = nnz_in_row_i / secondary_dim
 *     - Values in range [0, 1]
 *
 * COMPLEXITY:
 *     Time:  O(primary_dim)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over primary dimension
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void detection_rate(
    const Sparse<T, IsCSR>& matrix,   // Input sparse matrix
    Array<Real> out_rates              // Output detection rates [primary_dim]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: dispersion
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute dispersion index (variance / mean) for each feature.
 *
 * PARAMETERS:
 *     means          [in]  Pre-computed means
 *     vars           [in]  Pre-computed variances
 *     out_dispersion [out] Output dispersion values
 *
 * PRECONDITIONS:
 *     - All arrays have same length
 *     - means and vars computed from same data
 *
 * POSTCONDITIONS:
 *     - out_dispersion[i] = vars[i] / means[i] if means[i] > epsilon
 *     - out_dispersion[i] = 0 if means[i] <= epsilon
 *
 * ALGORITHM:
 *     For each element using 4-way SIMD unroll:
 *         1. Load mean and variance vectors
 *         2. Create mask for mean > epsilon
 *         3. Compute division with masked select
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 *
 * NUMERICAL NOTES:
 *     Uses epsilon = 1e-12 to avoid division by zero
 * -------------------------------------------------------------------------- */
void dispersion(
    Array<const Real> means,           // Input means [n]
    Array<const Real> vars,            // Input variances [n]
    Array<Real> out_dispersion         // Output dispersion [n]
);

} // namespace scl::kernel::feature
