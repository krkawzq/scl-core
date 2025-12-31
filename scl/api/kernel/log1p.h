/**
 * @file scl/api/kernel/log1p.h
 * @brief SCL Log1p Transformation C-API
 *
 * This header provides C-ABI compatible functions for logarithmic transformations:
 *   - log1p(x) = log(1 + x)  : Natural logarithm of (1+x)
 *   - log2p1(x) = log2(1 + x) : Base-2 logarithm of (1+x)
 *   - expm1(x) = exp(x) - 1  : Exponential minus 1
 *
 * ## Supported Operations
 *
 * ### Sparse Matrix Operations (In-Place)
 * - `scl_log1p_sparse`: Apply log1p to sparse matrix non-zero values
 * - `scl_log2p1_sparse`: Apply log2p1 to sparse matrix non-zero values
 * - `scl_expm1_sparse`: Apply expm1 to sparse matrix non-zero values
 *
 * ### Dense Array Operations
 * - `scl_log1p_array`: Apply log1p to dense array (in-place or out-of-place)
 * - `scl_log2p1_array`: Apply log2p1 to dense array (in-place or out-of-place)
 * - `scl_expm1_array`: Apply expm1 to dense array (in-place or out-of-place)
 *
 * ## Usage Example (Sparse Matrix)
 *
 * ```c
 * scl_sparse_t matrix = scl_sparse_from_coo(...);
 *
 * // Apply log1p transformation in-place
 * int32_t err = scl_log1p_sparse(matrix);
 * if (scl_is_error(err)) {
 *     printf("Error: %s\n", scl_get_error_message());
 *     scl_sparse_destroy(matrix);
 *     return;
 * }
 *
 * scl_sparse_destroy(matrix);
 * ```
 *
 * ## Usage Example (Dense Array)
 *
 * ```c
 * double values[] = {0.0, 1.0, 2.0, 3.0};
 * double output[4];
 *
 * // Apply log1p transformation (out-of-place)
 * int32_t err = scl_log1p_array(
 *     values, 4,       // input
 *     output,          // output (can be same as input for in-place)
 *     SCL_REAL64       // precision
 * );
 *
 * if (scl_is_success(err)) {
 *     // output now contains [log(1), log(2), log(3), log(4)]
 * }
 * ```
 *
 * ## Performance Characteristics
 *
 * - Sparse matrices: Parallel processing with automatic work distribution
 * - Dense arrays: Automatic parallel processing for arrays >= 1024 elements
 * - SIMD optimization: 8-way loop unrolling with prefetching
 * - Performance: ~16 flops/cycle on AVX-512, ~8 flops/cycle on AVX2
 *
 * ## Precision Support
 *
 * Only Real32 (float) and Real64 (double) are supported.
 * Operations on other precisions will return SCL_ERROR_INVALID_TYPE.
 *
 * ## Thread Safety
 *
 * - Sparse operations modify the matrix in-place (requires external locking)
 * - Array operations are thread-safe if input/output buffers don't overlap
 * - Multiple concurrent calls with different data are safe
 *
 * ## Memory Management
 *
 * - Sparse matrices: No additional allocations (in-place transformation)
 * - Dense arrays: No additional allocations (in-place or out-of-place)
 * - User is responsible for buffer validity during operation
 */

#ifndef SCL_API_KERNEL_LOG1P_H_
#define SCL_API_KERNEL_LOG1P_H_

#include "scl/api/core/type.h"
#include "scl/api/core/error.h"
#include "scl/api/core/sparse.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * SECTION 1: Sparse Matrix Operations (In-Place)
 * ============================================================================ */

/**
 * @brief Apply log1p transformation to sparse matrix (in-place)
 *
 * Applies log1p(x) = log(1 + x) to all non-zero values in the sparse matrix.
 * Zero values remain zero (sparse structure preserved).
 *
 * @param[in,out] matrix Sparse matrix handle (modified in-place)
 * @return Error code (0 = success, non-zero = error)
 *
 * @pre matrix must be a valid sparse matrix handle
 * @pre matrix value type must be Real32 or Real64
 * @post All non-zero values are replaced with log1p(value)
 * @post Matrix dimensions and structure remain unchanged
 *
 * @note This operation is parallel-safe (uses internal thread pool)
 * @note Typical performance: 1-5 GB/s throughput
 *
 * Error codes:
 * - SCL_SUCCESS (0): Operation succeeded
 * - SCL_ERROR_NULL_POINTER: matrix is null
 * - SCL_ERROR_INVALID_TYPE: matrix value type not supported (not Real32/64)
 */
int32_t scl_log1p_sparse(scl_sparse_t matrix);

/**
 * @brief Apply log2p1 transformation to sparse matrix (in-place)
 *
 * Applies log2p1(x) = log2(1 + x) to all non-zero values in the sparse matrix.
 *
 * @param[in,out] matrix Sparse matrix handle (modified in-place)
 * @return Error code (0 = success, non-zero = error)
 *
 * @pre matrix must be a valid sparse matrix handle
 * @pre matrix value type must be Real32 or Real64
 *
 * @note log2p1(x) = log(1+x) / ln(2) = log1p(x) * 1.44269504...
 */
int32_t scl_log2p1_sparse(scl_sparse_t matrix);

/**
 * @brief Apply expm1 transformation to sparse matrix (in-place)
 *
 * Applies expm1(x) = exp(x) - 1 to all non-zero values in the sparse matrix.
 *
 * @param[in,out] matrix Sparse matrix handle (modified in-place)
 * @return Error code (0 = success, non-zero = error)
 *
 * @pre matrix must be a valid sparse matrix handle
 * @pre matrix value type must be Real32 or Real64
 *
 * @warning This operation can densify the matrix if applied after log1p
 *          because exp(log(1+x)) - 1 = x (recovers original values)
 */
int32_t scl_expm1_sparse(scl_sparse_t matrix);

/* ============================================================================
 * SECTION 2: Dense Array Operations
 * ============================================================================ */

/**
 * @brief Apply log1p transformation to dense array
 *
 * Applies log1p(x) = log(1 + x) to all elements in the array.
 *
 * @param[in] input Input array pointer (read-only if output != input)
 * @param[in] size Number of elements in input array
 * @param[out] output Output array pointer (can be same as input for in-place)
 * @param[in] precision Precision type (SCL_REAL32 or SCL_REAL64)
 * @return Error code (0 = success, non-zero = error)
 *
 * @pre input must point to valid memory of size elements
 * @pre output must point to valid memory of size elements
 * @pre input and output may alias (in-place) or be separate (out-of-place)
 * @pre precision must be SCL_REAL32 or SCL_REAL64
 *
 * @post output[i] = log1p(input[i]) for i in [0, size)
 *
 * @note In-place example: scl_log1p_array(data, n, data, SCL_REAL64)
 * @note Out-of-place example: scl_log1p_array(input, n, output, SCL_REAL64)
 * @note Automatically uses parallel processing for size >= 1024
 *
 * Error codes:
 * - SCL_SUCCESS (0): Operation succeeded
 * - SCL_ERROR_NULL_POINTER: input or output is null
 * - SCL_ERROR_INVALID_ARGUMENT: size is 0
 * - SCL_ERROR_INVALID_TYPE: precision not supported (not Real32/64)
 */
int32_t scl_log1p_array(
    const void* input,
    int64_t size,
    void* output,
    scl_value_type_t precision
);

/**
 * @brief Apply log2p1 transformation to dense array
 *
 * Applies log2p1(x) = log2(1 + x) to all elements in the array.
 *
 * @param[in] input Input array pointer
 * @param[in] size Number of elements
 * @param[out] output Output array pointer (can be same as input)
 * @param[in] precision Precision type (SCL_REAL32 or SCL_REAL64)
 * @return Error code (0 = success, non-zero = error)
 *
 * @note All preconditions and postconditions same as scl_log1p_array
 */
int32_t scl_log2p1_array(
    const void* input,
    int64_t size,
    void* output,
    scl_value_type_t precision
);

/**
 * @brief Apply expm1 transformation to dense array
 *
 * Applies expm1(x) = exp(x) - 1 to all elements in the array.
 *
 * @param[in] input Input array pointer
 * @param[in] size Number of elements
 * @param[out] output Output array pointer (can be same as input)
 * @param[in] precision Precision type (SCL_REAL32 or SCL_REAL64)
 * @return Error code (0 = success, non-zero = error)
 *
 * @note All preconditions and postconditions same as scl_log1p_array
 * @note Numerically stable for small x values (avoids catastrophic cancellation)
 */
int32_t scl_expm1_array(
    const void* input,
    int64_t size,
    void* output,
    scl_value_type_t precision
);

#ifdef __cplusplus
}
#endif

#endif  // SCL_API_KERNEL_LOG1P_H_
