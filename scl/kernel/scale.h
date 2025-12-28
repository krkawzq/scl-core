// =============================================================================
// FILE: scl/kernel/scale.h
// BRIEF: API reference for scaling operation kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::scale {

/* -----------------------------------------------------------------------------
 * FUNCTION: standardize
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Standardize sparse matrix values in-place: (x - mean) / std, with
 *     optional clipping and zero-centering control.
 *
 * PARAMETERS:
 *     matrix      [in,out] Sparse matrix to standardize
 *     means       [in]     Per-primary-dimension means
 *     stds        [in]     Per-primary-dimension standard deviations
 *     max_value   [in]     Clip threshold (0 disables clipping)
 *     zero_center [in]     Whether to subtract mean before scaling
 *
 * PRECONDITIONS:
 *     - means.len == matrix.primary_dim()
 *     - stds.len == matrix.primary_dim()
 *     - stds[i] > 0 for rows to be processed (zero std rows are skipped)
 *
 * POSTCONDITIONS:
 *     - Each value v transformed to: (v - mean) / std (if zero_center)
 *                                 or: v / std (if not zero_center)
 *     - Results clipped to [-max_value, max_value] if max_value > 0
 *     - Rows with std = 0 are unchanged
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix values directly
 *
 * ALGORITHM:
 *     Uses 3-tier adaptive strategy based on row length:
 *         - Short (< 16): scalar loop
 *         - Medium (16-128): 4-way SIMD unroll
 *         - Long (>= 128): 8-way SIMD unroll with prefetch
 *     Branch conditions (zero_center, do_clip) lifted outside inner loops
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over primary dimension, each row processed independently
 *
 * NUMERICAL NOTES:
 *     Uses inv_sigma = 1/std to replace division with multiplication
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void standardize(
    Sparse<T, IsCSR>& matrix,          // Matrix to standardize, modified in-place
    Array<const T> means,               // Per-row means [primary_dim]
    Array<const T> stds,                // Per-row standard deviations [primary_dim]
    T max_value,                        // Clip threshold (0 = no clipping)
    bool zero_center                    // Subtract mean before scaling
);

/* -----------------------------------------------------------------------------
 * FUNCTION: scale_rows
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Multiply each primary dimension by a corresponding scale factor.
 *
 * PARAMETERS:
 *     matrix [in,out] Sparse matrix to scale
 *     scales [in]     Per-row scale factors
 *
 * PRECONDITIONS:
 *     - scales.len == matrix.primary_dim()
 *
 * POSTCONDITIONS:
 *     - Each value in row i multiplied by scales[i]
 *     - Rows with scales[i] == 1 are unchanged (early exit)
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix values directly
 *
 * ALGORITHM:
 *     For each primary index in parallel:
 *         1. Skip if scale == 1
 *         2. Use SIMD 4-way unroll with prefetch for scaling
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over primary dimension
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void scale_rows(
    Sparse<T, IsCSR>& matrix,          // Matrix to scale, modified in-place
    Array<const T> scales               // Per-row scale factors [primary_dim]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: shift_rows
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Add a constant offset to each primary dimension.
 *
 * PARAMETERS:
 *     matrix  [in,out] Sparse matrix to shift
 *     offsets [in]     Per-row offsets to add
 *
 * PRECONDITIONS:
 *     - offsets.len == matrix.primary_dim()
 *
 * POSTCONDITIONS:
 *     - Each value in row i increased by offsets[i]
 *     - Rows with offsets[i] == 0 are unchanged (early exit)
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix values directly
 *
 * ALGORITHM:
 *     For each primary index in parallel:
 *         1. Skip if offset == 0
 *         2. Use SIMD 4-way unroll with prefetch for addition
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over primary dimension
 *
 * NUMERICAL NOTES:
 *     Only modifies stored (non-zero) values. Implicit zeros remain zero.
 *     For true shift of all values including zeros, matrix must be densified.
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void shift_rows(
    Sparse<T, IsCSR>& matrix,          // Matrix to shift, modified in-place
    Array<const T> offsets              // Per-row offsets [primary_dim]
);

} // namespace scl::kernel::scale
