#pragma once

// =============================================================================
// WARNING: UNSAFE API - ABI UNSTABLE - ADVANCED USE ONLY
// =============================================================================
//
// This header exposes internal memory layouts for zero-overhead access.
//
// RISKS OF USING THIS API:
//   1. ABI Unstable: struct layouts may change between versions
//   2. Memory Safety: bypasses registry safety checks
//   3. Lifetime Management: caller responsible for pointer validity
//   4. No Error Checking: incorrect usage causes undefined behavior
//   5. No Portability: assumes specific memory layout
//
// ONLY USE THIS API IF:
//   - You need absolute maximum performance
//   - You understand the internal memory model
//   - You can handle pointer lifecycle manually
//   - You can rebuild when library version changes
//
// FOR MOST USE CASES: Use the safe API in sparse.h and dense.h
//
// VERSION COMPATIBILITY:
//   - This API is NOT covered by semantic versioning guarantees
//   - Breaking changes may occur in minor or patch releases
//   - Always check struct layouts after library updates
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Raw Sparse Matrix Layout (Internal Structure Exposed)
// =============================================================================

/// @brief Raw sparse matrix layout (CSR or CSC)
/// @warning ABI UNSTABLE - struct may change without notice
/// @details Exposes internal block-allocated sparse matrix structure
///
/// MEMORY LAYOUT:
///   - data_ptrs[i] points to values for row i (CSR) or column i (CSC)
///   - indices_ptrs[i] points to column (CSR) or row (CSC) indices
///   - lengths[i] = number of non-zeros in row/column i
///   - For traditional CSR/CSC, use contiguous strategy
///
/// LIFETIME RULES:
///   - data_ptrs, indices_ptrs, lengths: managed by registry
///   - Individual row/column arrays: may be aliases (shared with views)
///   - Caller MUST NOT free any pointers directly
///   - Use scl_sparse_destroy() for proper cleanup
typedef struct scl_sparse_raw {
    void** data_ptrs;           ///< Array of pointers to row/col values [pdim]
    void** indices_ptrs;        ///< Array of pointers to row/col indices [pdim]
    scl_index_t* lengths;       ///< Length of each row/col [pdim]
    scl_index_t rows;           ///< Number of rows
    scl_index_t cols;           ///< Number of columns
    scl_index_t nnz;            ///< Total number of non-zeros
    scl_bool_t is_csr;          ///< SCL_TRUE for CSR, SCL_FALSE for CSC
} scl_sparse_raw_t;

/// @brief Get raw internal structure (read-only, does not transfer ownership)
/// @param[in] matrix Matrix handle (non-null)
/// @param[out] out Raw structure (non-null)
/// @return SCL_OK on success, error code otherwise
/// @warning Pointers in out become invalid when matrix is destroyed
/// @warning DO NOT modify pointed-to data (undefined behavior)
/// @warning DO NOT free any pointers (use scl_sparse_destroy)
scl_error_t scl_sparse_unsafe_get_raw(
    scl_sparse_t matrix,
    scl_sparse_raw_t* out
);

/// @brief Create matrix from raw structure (EXTREMELY DANGEROUS)
/// @param[in] raw Raw structure with pre-allocated arrays (non-null)
/// @param[out] out Matrix handle (non-null)
/// @return SCL_OK on success, error code otherwise
/// @warning CALLER MUST register all pointers with registry first
/// @warning Incorrect usage WILL cause memory corruption
/// @warning For experts only - prefer safe API
/// @note This function does NOT register pointers - that's caller's job
scl_error_t scl_sparse_unsafe_from_raw(
    const scl_sparse_raw_t* raw,
    scl_sparse_t* out
);

// =============================================================================
// Raw Dense Matrix Layout
// =============================================================================

/// @brief Raw dense matrix layout (row-major, pure view)
/// @warning ABI UNSTABLE - struct may change without notice
///
/// MEMORY LAYOUT:
///   - Row-major: data[i * stride + j] = element at (i, j)
///   - For contiguous: stride == cols
///   - For views: stride may be > cols (skip padding)
///
/// LIFETIME RULES:
///   - DenseView is always a view - never owns data
///   - Caller MUST ensure data pointer remains valid
///   - Use scl_dense_destroy() to free the handle (not the data)
typedef struct scl_dense_raw {
    scl_real_t* data;           ///< Row-major data pointer (external, not owned)
    scl_index_t rows;           ///< Number of rows
    scl_index_t cols;           ///< Number of columns
    scl_index_t stride;         ///< Row stride (>= cols)
} scl_dense_raw_t;

/// @brief Get raw internal structure (read-only, does not transfer ownership)
/// @param[in] matrix Matrix handle (non-null)
/// @param[out] out Raw structure (non-null)
/// @return SCL_OK on success, error code otherwise
/// @warning data pointer lifetime managed by caller (DenseView never owns data)
scl_error_t scl_dense_unsafe_get_raw(
    scl_dense_t matrix,
    scl_dense_raw_t* out
);

/// @brief Create matrix view from raw structure
/// @param[in] raw Raw structure (non-null)
/// @param[out] out Matrix handle (non-null)
/// @return SCL_OK on success, error code otherwise
/// @warning DenseView never owns data - caller manages data lifetime
/// @warning data pointer must remain valid for matrix lifetime
scl_error_t scl_dense_unsafe_from_raw(
    const scl_dense_raw_t* raw,
    scl_dense_t* out
);

// =============================================================================
// Direct Memory Access (Zero Overhead, Maximum Risk)
// =============================================================================

/// @brief Get row data pointer for CSR matrix (zero-overhead)
/// @param[in] matrix CSR matrix handle (non-null, must be CSR)
/// @param[in] row Row index [0, rows)
/// @param[out] data Row data pointer (non-null)
/// @param[out] indices Row indices pointer (non-null)
/// @param[out] length Number of non-zeros in row (non-null)
/// @return SCL_OK on success, error code otherwise
/// @warning NO bounds checking in release builds
/// @warning Pointers invalid after matrix destruction
/// @warning DO NOT modify if matrix is shared (aliased)
scl_error_t scl_sparse_unsafe_get_row(
    scl_sparse_t matrix,
    scl_index_t row,
    scl_real_t** data,
    scl_index_t** indices,
    scl_index_t* length
);

/// @brief Get column data pointer for CSC matrix (zero-overhead)
/// @param[in] matrix CSC matrix handle (non-null, must be CSC)
/// @param[in] col Column index [0, cols)
/// @param[out] data Column data pointer (non-null)
/// @param[out] indices Column indices pointer (non-null)
/// @param[out] length Number of non-zeros in column (non-null)
/// @return SCL_OK on success, error code otherwise
/// @warning NO bounds checking in release builds
/// @warning Pointers invalid after matrix destruction
scl_error_t scl_sparse_unsafe_get_col(
    scl_sparse_t matrix,
    scl_index_t col,
    scl_real_t** data,
    scl_index_t** indices,
    scl_index_t* length
);

/// @brief Get mutable row pointer for dense matrix (zero-overhead)
/// @param[in] matrix Dense matrix handle (non-null)
/// @param[in] row Row index [0, rows)
/// @param[out] data Row data pointer (non-null)
/// @return SCL_OK on success, error code otherwise
/// @warning NO bounds checking in release builds
/// @warning Use stride for element access: data[j] = element at (row, j)
/// @warning Modifications affect original matrix
scl_error_t scl_dense_unsafe_get_row(
    scl_dense_t matrix,
    scl_index_t row,
    scl_real_t** data
);

// =============================================================================
// Registry Integration (For Advanced Users)
// =============================================================================

/// @brief Register external buffer with registry (for ownership transfer)
/// @param[in] ptr Pointer to buffer
/// @param[in] size Buffer size in bytes
/// @return Buffer ID on success, 0 on failure
/// @warning Buffer must be allocated with new[] (will use delete[])
/// @note Use this to integrate externally-allocated memory with SCL
scl_size_t scl_unsafe_register_buffer(void* ptr, scl_size_t size);

/// @brief Create alias for shared memory (for zero-copy slicing)
/// @param[in] ptr Alias pointer (points somewhere within parent buffer)
/// @param[in] buffer_id Parent buffer ID from scl_unsafe_register_buffer
/// @return SCL_OK on success, error code otherwise
/// @note Alias shares lifetime with parent buffer via reference counting
/// @note ptr address implicitly encodes offset - no separate offset needed
scl_error_t scl_unsafe_create_alias(
    void* ptr,
    scl_size_t buffer_id
);

/// @brief Unregister pointer from registry (manual cleanup)
/// @param[in] ptr Pointer to unregister
/// @return SCL_OK on success, error code otherwise
/// @warning Only use if you know what you're doing
/// @warning Incorrect usage causes memory leaks or double-frees
scl_error_t scl_unsafe_unregister(void* ptr);

#ifdef __cplusplus
}
#endif
