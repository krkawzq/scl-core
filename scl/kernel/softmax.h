// =============================================================================
// FILE: scl/kernel/softmax.h
// BRIEF: API reference for softmax operations
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::softmax {

/* -----------------------------------------------------------------------------
 * FUNCTION: softmax_inplace (dense array)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply softmax normalization in-place to a dense array.
 *
 * PARAMETERS:
 *     vals     [in,out] Pointer to values array, modified in-place
 *     len      [in]     Length of array
 *
 * PRECONDITIONS:
 *     - vals must be valid pointer if len > 0
 *     - len >= 0
 *
 * POSTCONDITIONS:
 *     - All values in [0, 1] and sum to 1.0
 *     - Original ordering preserved
 *     - For empty array (len=0): no-op
 *
 * MUTABILITY:
 *     INPLACE - modifies vals directly
 *
 * ALGORITHM:
 *     3-tier adaptive strategy based on array length:
 *     1. Short (< 16): Scalar loop
 *     2. Medium (< 128): 4-way SIMD unroll with prefetch
 *     3. Long (>= 128): 8-way SIMD unroll with 8 accumulators for ILP
 *
 *     Steps:
 *     1. Find max value for numerical stability
 *     2. Compute exp(x - max) and sum simultaneously
 *     3. Normalize by dividing each element by sum
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe for different arrays, unsafe for same array
 *
 * NUMERICAL NOTES:
 *     - Max subtraction prevents overflow in exp()
 *     - Returns uniform distribution if sum is zero
 * -------------------------------------------------------------------------- */
template <typename T>
void softmax_inplace(
    T* vals,                       // Values array, modified in-place
    Size len                       // Array length
);

/* -----------------------------------------------------------------------------
 * FUNCTION: softmax_inplace (dense array with temperature)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply softmax with temperature scaling in-place.
 *
 * PARAMETERS:
 *     vals        [in,out] Pointer to values array, modified in-place
 *     len         [in]     Length of array
 *     temperature [in]     Temperature parameter (higher = more uniform)
 *
 * PRECONDITIONS:
 *     - vals must be valid pointer if len > 0
 *     - len >= 0
 *
 * POSTCONDITIONS:
 *     - All values in [0, 1] and sum to 1.0
 *     - temperature > 0: softmax(x / temperature)
 *     - temperature <= 0: one-hot at maximum value
 *
 * MUTABILITY:
 *     INPLACE - modifies vals directly
 *
 * ALGORITHM:
 *     1. Scale all values by 1/temperature
 *     2. Apply standard softmax
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe for different arrays, unsafe for same array
 * -------------------------------------------------------------------------- */
template <typename T>
void softmax_inplace(
    T* vals,                       // Values array, modified in-place
    Size len,                      // Array length
    T temperature                  // Temperature scaling parameter
);

/* -----------------------------------------------------------------------------
 * FUNCTION: log_softmax_inplace (dense array)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply log-softmax in-place to a dense array.
 *
 * PARAMETERS:
 *     vals     [in,out] Pointer to values array, modified in-place
 *     len      [in]     Length of array
 *
 * PRECONDITIONS:
 *     - vals must be valid pointer if len > 0
 *     - len >= 0
 *
 * POSTCONDITIONS:
 *     - All values <= 0 (log probabilities)
 *     - exp(vals) sums to 1.0
 *     - For empty array (len=0): no-op
 *
 * MUTABILITY:
 *     INPLACE - modifies vals directly
 *
 * ALGORITHM:
 *     log_softmax(x) = x - max - log(sum(exp(x - max)))
 *
 *     3-tier adaptive strategy:
 *     1. Find max value
 *     2. Compute sum(exp(x - max)) with SIMD
 *     3. Subtract (max + log(sum)) from each element
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe for different arrays, unsafe for same array
 *
 * NUMERICAL NOTES:
 *     - More numerically stable than log(softmax(x))
 *     - Avoids computing explicit probabilities
 * -------------------------------------------------------------------------- */
template <typename T>
void log_softmax_inplace(
    T* vals,                       // Values array, modified in-place
    Size len                       // Array length
);

/* -----------------------------------------------------------------------------
 * FUNCTION: log_softmax_inplace (dense array with temperature)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply log-softmax with temperature scaling in-place.
 *
 * PARAMETERS:
 *     vals        [in,out] Pointer to values array, modified in-place
 *     len         [in]     Length of array
 *     temperature [in]     Temperature parameter
 *
 * PRECONDITIONS:
 *     - vals must be valid pointer if len > 0
 *     - len >= 0
 *
 * POSTCONDITIONS:
 *     - All values <= 0 (log probabilities)
 *     - temperature > 0: log_softmax(x / temperature)
 *     - temperature <= 0: 0 at max, -inf elsewhere
 *
 * MUTABILITY:
 *     INPLACE - modifies vals directly
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe for different arrays, unsafe for same array
 * -------------------------------------------------------------------------- */
template <typename T>
void log_softmax_inplace(
    T* vals,                       // Values array, modified in-place
    Size len,                      // Array length
    T temperature                  // Temperature scaling parameter
);

/* -----------------------------------------------------------------------------
 * FUNCTION: softmax_inplace (sparse matrix)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply softmax row-wise in-place to a sparse matrix.
 *
 * PARAMETERS:
 *     matrix   [in,out] Sparse matrix (CSR or CSC), values modified in-place
 *
 * PRECONDITIONS:
 *     - Matrix must be valid sparse format
 *     - Matrix values must be mutable
 *
 * POSTCONDITIONS:
 *     - Each row sums to 1.0 (considering only non-zero elements)
 *     - Matrix structure (indices, pointers) unchanged
 *     - Empty rows are unchanged
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix.values() directly
 *
 * ALGORITHM:
 *     For each row in parallel:
 *         Apply 3-tier adaptive softmax to non-zero values
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows, no shared mutable state
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void softmax_inplace(
    Sparse<T, IsCSR>& matrix       // Sparse matrix, values modified in-place
);

/* -----------------------------------------------------------------------------
 * FUNCTION: softmax_inplace (sparse matrix with temperature)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply softmax with temperature row-wise in-place to a sparse matrix.
 *
 * PARAMETERS:
 *     matrix      [in,out] Sparse matrix, values modified in-place
 *     temperature [in]     Temperature parameter
 *
 * PRECONDITIONS:
 *     - Matrix must be valid sparse format
 *     - Matrix values must be mutable
 *
 * POSTCONDITIONS:
 *     - Each row sums to 1.0 (considering only non-zero elements)
 *     - Matrix structure unchanged
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix.values() directly
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void softmax_inplace(
    Sparse<T, IsCSR>& matrix,      // Sparse matrix, values modified in-place
    T temperature                  // Temperature scaling parameter
);

/* -----------------------------------------------------------------------------
 * FUNCTION: log_softmax_inplace (sparse matrix)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply log-softmax row-wise in-place to a sparse matrix.
 *
 * PARAMETERS:
 *     matrix   [in,out] Sparse matrix, values modified in-place
 *
 * PRECONDITIONS:
 *     - Matrix must be valid sparse format
 *     - Matrix values must be mutable
 *
 * POSTCONDITIONS:
 *     - All values <= 0 (log probabilities)
 *     - Matrix structure unchanged
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix.values() directly
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void log_softmax_inplace(
    Sparse<T, IsCSR>& matrix       // Sparse matrix, values modified in-place
);

/* -----------------------------------------------------------------------------
 * FUNCTION: log_softmax_inplace (sparse matrix with temperature)
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply log-softmax with temperature row-wise in-place to a sparse matrix.
 *
 * PARAMETERS:
 *     matrix      [in,out] Sparse matrix, values modified in-place
 *     temperature [in]     Temperature parameter
 *
 * PRECONDITIONS:
 *     - Matrix must be valid sparse format
 *     - Matrix values must be mutable
 *
 * POSTCONDITIONS:
 *     - All values <= 0 (log probabilities)
 *     - Matrix structure unchanged
 *
 * MUTABILITY:
 *     INPLACE - modifies matrix.values() directly
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void log_softmax_inplace(
    Sparse<T, IsCSR>& matrix,      // Sparse matrix, values modified in-place
    T temperature                  // Temperature scaling parameter
);

} // namespace scl::kernel::softmax
