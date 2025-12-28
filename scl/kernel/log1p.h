// =============================================================================
// FILE: scl/kernel/log1p.h
// BRIEF: API reference for logarithmic transform kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::log1p {

// =============================================================================
// Configuration Constants
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 64;
    constexpr double INV_LN2 = 1.44269504088896340736;
    constexpr double LN2 = 0.6931471805599453;
}

// =============================================================================
// Core Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: log1p_inplace
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply log(1 + x) transform to all non-zero values in sparse matrix.
 *
 * PARAMETERS:
 *     matrix  [in,out] Sparse matrix to transform in-place
 *
 * PRECONDITIONS:
 *     - Matrix values must be >= -1 (log1p domain requirement)
 *     - For expression data: values should be non-negative counts
 *
 * POSTCONDITIONS:
 *     - All non-zero values v transformed to log(1 + v)
 *     - Matrix structure (indices, indptr) unchanged
 *     - Zero values remain zero (not stored in sparse format)
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix.values() directly
 *
 * ALGORITHM:
 *     For each row in parallel:
 *         1. 4-way SIMD unrolled processing with prefetch
 *         2. Apply scl::simd::Log1p to value vectors
 *         3. Scalar cleanup for tail elements
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows, each writes to independent memory
 *
 * NUMERICAL NOTES:
 *     - Uses SIMD Log1p for numerical stability near zero
 *     - More accurate than log(1+x) for small x values
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void log1p_inplace(
    Sparse<T, IsCSR>& matrix           // Sparse matrix, modified in-place
);

/* -----------------------------------------------------------------------------
 * FUNCTION: log2p1_inplace
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply log2(1 + x) transform to all non-zero values in sparse matrix.
 *
 * PARAMETERS:
 *     matrix  [in,out] Sparse matrix to transform in-place
 *
 * PRECONDITIONS:
 *     - Matrix values must be >= -1
 *     - For expression data: values should be non-negative counts
 *
 * POSTCONDITIONS:
 *     - All non-zero values v transformed to log2(1 + v)
 *     - Matrix structure unchanged
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix.values() directly
 *
 * ALGORITHM:
 *     For each row in parallel:
 *         1. 4-way SIMD unrolled processing with prefetch
 *         2. Apply scl::simd::Log1p then multiply by INV_LN2
 *         3. Scalar cleanup for tail elements
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows
 *
 * NUMERICAL NOTES:
 *     - Computed as log(1+x) * (1/ln(2)) for efficiency
 *     - Base-2 logarithm common in information theory applications
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void log2p1_inplace(
    Sparse<T, IsCSR>& matrix           // Sparse matrix, modified in-place
);

/* -----------------------------------------------------------------------------
 * FUNCTION: expm1_inplace
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply exp(x) - 1 transform to all non-zero values in sparse matrix.
 *
 * PARAMETERS:
 *     matrix  [in,out] Sparse matrix to transform in-place
 *
 * PRECONDITIONS:
 *     - Values should be in reasonable range to avoid overflow
 *     - Typically used to reverse log1p transform
 *
 * POSTCONDITIONS:
 *     - All non-zero values v transformed to exp(v) - 1
 *     - Matrix structure unchanged
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix.values() directly
 *
 * ALGORITHM:
 *     For each row in parallel:
 *         1. 4-way SIMD unrolled processing with prefetch
 *         2. Apply scl::simd::Expm1 to value vectors
 *         3. Scalar cleanup for tail elements
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows
 *
 * NUMERICAL NOTES:
 *     - Uses SIMD Expm1 for numerical stability near zero
 *     - Inverse of log1p: expm1(log1p(x)) = x
 *     - More accurate than exp(x)-1 for small x values
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void expm1_inplace(
    Sparse<T, IsCSR>& matrix           // Sparse matrix, modified in-place
);

} // namespace scl::kernel::log1p
