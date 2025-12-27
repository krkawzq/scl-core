// =============================================================================
// FILE: scl/kernel/algebra.h
// BRIEF: API reference for high-performance sparse linear algebra kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.h"
#include "scl/core/sparse.h"
#include "scl/core/macros.hpp"

namespace scl::kernel::algebra {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

/* -----------------------------------------------------------------------------
 * NAMESPACE: config
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Configuration constants for performance tuning.
 *
 * CONSTANTS:
 *     PREFETCH_DISTANCE      - Cache line prefetch distance (64 elements)
 *     SHORT_ROW_THRESHOLD    - Threshold for short row optimization (8)
 *     MEDIUM_ROW_THRESHOLD   - Threshold for medium row optimization (64)
 *
 * PERFORMANCE TUNING:
 *     These thresholds control adaptive dot product strategies:
 *     - Rows with nnz < SHORT_ROW_THRESHOLD: scalar loop
 *     - Rows with nnz < MEDIUM_ROW_THRESHOLD: 4-way unroll
 *     - Rows with nnz >= MEDIUM_ROW_THRESHOLD: 8-way unroll + prefetch
 *
 * RATIONALE:
 *     Short rows benefit from minimal overhead. Medium rows need some
 *     unrolling for ILP. Long rows benefit from aggressive prefetching
 *     and maximum parallelism.
 * -------------------------------------------------------------------------- */

namespace config {
    constexpr size_t PREFETCH_DISTANCE = 64;
    constexpr size_t SHORT_ROW_THRESHOLD = 8;
    constexpr size_t MEDIUM_ROW_THRESHOLD = 64;
}

// =============================================================================
// SECTION 2: Sparse Matrix-Vector Multiplication
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: spmv
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sparse matrix-vector multiplication with alpha and beta scaling.
 *
 * COMPUTES:
 *     y = alpha * A * x + beta * y
 *
 * PARAMETERS:
 *     A     [in]     Sparse matrix (Sparse<T, IsCSR>)
 *     x     [in]     Input vector (Array<const T>), size = secondary_dim(A)
 *     y     [in,out] Output vector (Array<T>), size = primary_dim(A), PRE-ALLOCATED
 *     alpha [in]     Scalar multiplier for A*x (default: T(1))
 *     beta  [in]     Scalar multiplier for y (default: T(0))
 *
 * TEMPLATE PARAMETERS:
 *     T     - Element type (typically Real)
 *     IsCSR - true for CSR (row-major), false for CSC (column-major)
 *
 * PRECONDITIONS:
 *     - y.size() >= primary_dim(A)
 *     - x.size() >= secondary_dim(A)
 *     - A must be valid sparse matrix
 *
 * POSTCONDITIONS:
 *     - y contains result: alpha * A * x + beta * y_original
 *     - A and x are unchanged
 *
 * MUTABILITY:
 *     INPLACE on y (modifies output vector)
 *
 * ALGORITHM:
 *     1. Scale y by beta (SIMD-optimized, handles beta=0, beta=1 efficiently)
 *     2. If alpha == 0, return early
 *     3. Parallel loop over primary dimension (rows for CSR, columns for CSC)
 *     4. For each row/column:
 *        a. Get values and indices arrays
 *        b. Compute dot product using adaptive strategy:
 *           - Short rows (nnz < 8): scalar loop
 *           - Medium rows (nnz < 64): 4-way unrolled loop
 *           - Long rows (nnz >= 64): 8-way unrolled loop with prefetching
 *        c. Accumulate: y[i] += alpha * dot
 *
 * COMPLEXITY:
 *     Time:  O(nnz) sequential work, parallelized across primary_dim
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over primary dimension, each thread writes to
 *     distinct elements of y. No shared mutable state.
 *
 * PERFORMANCE NOTES:
 *     - SIMD-optimized beta scaling (zero, identity, general cases)
 *     - Adaptive dot product strategy based on row/column length
 *     - 8-way unrolling for long rows reduces loop overhead
 *     - Prefetching hides memory latency for long rows
 *     - Parallel processing scales with hardware concurrency
 *
 * NUMERICAL NOTES:
 *     - Uses standard floating-point arithmetic (no special handling for NaN/Inf)
 *     - Accumulation order is non-deterministic in parallel execution
 *     - For very large matrices, consider using compensated summation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void spmv(
    const Sparse<T, IsCSR>& A,  // Sparse matrix input
    Array<const T> x,            // Input vector [secondary_dim]
    Array<T> y,                  // Output vector [primary_dim], PRE-ALLOCATED
    T alpha = T(1),              // Scalar for A*x
    T beta = T(0)                // Scalar for y
);

// =============================================================================
// SECTION 3: Convenience Wrappers
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: spmv_simple
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Simplified sparse matrix-vector multiplication: y = A * x
 *
 * PARAMETERS:
 *     A [in]     Sparse matrix (Sparse<T, IsCSR>)
 *     x [in]     Input vector (Array<const T>), size = secondary_dim(A)
 *     y [in,out] Output vector (Array<T>), size = primary_dim(A), PRE-ALLOCATED
 *
 * TEMPLATE PARAMETERS:
 *     T     - Element type (typically Real)
 *     IsCSR - true for CSR (row-major), false for CSC (column-major)
 *
 * PRECONDITIONS:
 *     - y.size() >= primary_dim(A)
 *     - x.size() >= secondary_dim(A)
 *     - A must be valid sparse matrix
 *
 * POSTCONDITIONS:
 *     - y contains A * x
 *     - Original y values are overwritten (not accumulated)
 *
 * MUTABILITY:
 *     INPLACE on y (overwrites output vector)
 *
 * EQUIVALENT TO:
 *     spmv(A, x, y, T(1), T(0))
 *
 * COMPLEXITY:
 *     Time:  O(nnz) parallelized
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - same as spmv
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void spmv_simple(
    const Sparse<T, IsCSR>& A,  // Sparse matrix input
    Array<const T> x,            // Input vector [secondary_dim]
    Array<T> y                   // Output vector [primary_dim], PRE-ALLOCATED
);

/* -----------------------------------------------------------------------------
 * FUNCTION: spmv_add
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Accumulate sparse matrix-vector product: y += A * x
 *
 * PARAMETERS:
 *     A [in]     Sparse matrix (Sparse<T, IsCSR>)
 *     x [in]     Input vector (Array<const T>), size = secondary_dim(A)
 *     y [in,out] Output vector (Array<T>), size = primary_dim(A), PRE-ALLOCATED
 *
 * TEMPLATE PARAMETERS:
 *     T     - Element type (typically Real)
 *     IsCSR - true for CSR (row-major), false for CSC (column-major)
 *
 * PRECONDITIONS:
 *     - y.size() >= primary_dim(A)
 *     - x.size() >= secondary_dim(A)
 *     - A must be valid sparse matrix
 *
 * POSTCONDITIONS:
 *     - y contains y_original + A * x
 *     - Original y values are accumulated (not overwritten)
 *
 * MUTABILITY:
 *     INPLACE on y (accumulates into output vector)
 *
 * EQUIVALENT TO:
 *     spmv(A, x, y, T(1), T(1))
 *
 * COMPLEXITY:
 *     Time:  O(nnz) parallelized
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - same as spmv
 *
 * USE CASE:
 *     Useful for accumulating contributions from multiple sparse matrices
 *     or for iterative algorithms that update y incrementally.
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void spmv_add(
    const Sparse<T, IsCSR>& A,  // Sparse matrix input
    Array<const T> x,            // Input vector [secondary_dim]
    Array<T> y                   // Output vector [primary_dim], PRE-ALLOCATED
);

/* -----------------------------------------------------------------------------
 * FUNCTION: spmv_scaled
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Scaled sparse matrix-vector multiplication: y = alpha * A * x
 *
 * PARAMETERS:
 *     A     [in]     Sparse matrix (Sparse<T, IsCSR>)
 *     x     [in]     Input vector (Array<const T>), size = secondary_dim(A)
 *     y     [in,out] Output vector (Array<T>), size = primary_dim(A), PRE-ALLOCATED
 *     alpha [in]     Scalar multiplier for A*x
 *
 * TEMPLATE PARAMETERS:
 *     T     - Element type (typically Real)
 *     IsCSR - true for CSR (row-major), false for CSC (column-major)
 *
 * PRECONDITIONS:
 *     - y.size() >= primary_dim(A)
 *     - x.size() >= secondary_dim(A)
 *     - A must be valid sparse matrix
 *
 * POSTCONDITIONS:
 *     - y contains alpha * A * x
 *     - Original y values are overwritten (not accumulated)
 *
 * MUTABILITY:
 *     INPLACE on y (overwrites output vector)
 *
 * EQUIVALENT TO:
 *     spmv(A, x, y, alpha, T(0))
 *
 * COMPLEXITY:
 *     Time:  O(nnz) parallelized
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - same as spmv
 *
 * USE CASE:
 *     Useful when you need to scale the matrix-vector product but don't
 *     need to accumulate into existing y values.
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void spmv_scaled(
    const Sparse<T, IsCSR>& A,  // Sparse matrix input
    Array<const T> x,            // Input vector [secondary_dim]
    Array<T> y,                  // Output vector [primary_dim], PRE-ALLOCATED
    T alpha                      // Scalar for A*x
);

} // namespace scl::kernel::algebra

