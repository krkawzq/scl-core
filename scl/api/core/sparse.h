/**
 * @file scl/api/core/sparse.h
 * @brief SCL Sparse Matrix C-API
 *
 * This header provides FILE-style operations for sparse matrices:
 *   - Creation and destruction
 *   - Property queries (dimensions, nnz, type info)
 *   - Data access (row/column data, element lookup)
 *   - Operations (transpose, scale, slice, clone)
 *
 * ## Usage Example
 *
 * ```c
 * // Create a sparse matrix from COO format
 * int64_t rows[] = {0, 1, 2};
 * int64_t cols[] = {1, 0, 2};
 * double vals[] = {1.0, 2.0, 3.0};
 *
 * scl_sparse_t mat = scl_sparse_from_coo(
 *     3, 3,           // dimensions
 *     rows, cols, vals, 3,  // COO data
 *     SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR
 * );
 *
 * if (!SCL_IS_VALID_SPARSE(mat)) {
 *     printf("Error: %s\n", scl_get_error_message());
 *     return;
 * }
 *
 * // Query properties
 * int64_t nnz = scl_sparse_nnz(mat);
 * printf("NNZ: %lld\n", (long long)nnz);
 *
 * // Cleanup
 * scl_sparse_destroy(mat);
 * ```
 *
 * ## Memory Management
 *
 * - All `scl_sparse_*` creation functions return a new handle or SCL_NULL_SPARSE on error
 * - Handles must be destroyed with `scl_sparse_destroy`
 * - Some operations (slice, row_select) return handles that share data with the original
 *   - These shared handles must still be destroyed independently
 *   - Shared data is reference-counted internally
 *
 * ## Thread Safety
 *
 * - Read operations are thread-safe (nnz, rows, cols, at, exists)
 * - Write operations (scale) require external synchronization
 * - Creation and destruction are not thread-safe for the same handle
 */

#ifndef SCL_API_CORE_SPARSE_H_
#define SCL_API_CORE_SPARSE_H_

#include "type.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * SECTION 1: Creation Functions
 * ============================================================================ */

/**
 * @brief Create an empty sparse matrix (all zeros)
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param value_type Value type (Real/Int/Uint + precision)
 * @param index_type Index precision (SCL_INDEX32 or SCL_INDEX64)
 * @param layout Storage layout (SCL_LAYOUT_CSR or SCL_LAYOUT_CSC)
 * @return New sparse handle, or SCL_NULL_SPARSE on error
 */
scl_sparse_t scl_sparse_zeros(
    int64_t rows,
    int64_t cols,
    scl_value_type_t value_type,
    scl_index_type_t index_type,
    scl_layout_t layout
);

/**
 * @brief Create an identity matrix
 *
 * @param n Matrix dimension (n x n)
 * @param value_type Value type (Real/Int/Uint + precision)
 * @param index_type Index precision
 * @return New sparse CSR handle, or SCL_NULL_SPARSE on error
 */
scl_sparse_t scl_sparse_identity(
    int64_t n,
    scl_value_type_t value_type,
    scl_index_type_t index_type
);

/**
 * @brief Create sparse matrix from COO (Coordinate) format
 *
 * Converts COO data to CSR/CSC format. Indices are sorted automatically.
 * Duplicate entries are summed.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param row_indices Row indices array (length = nnz)
 * @param col_indices Column indices array (length = nnz)
 * @param values Value array (length = nnz)
 * @param nnz Number of non-zero elements
 * @param real_type Value precision
 * @param index_type Index precision
 * @param layout Output layout (CSR or CSC)
 * @return New sparse handle, or SCL_NULL_SPARSE on error
 *
 * @note Index arrays are interpreted according to index_type
 * @note Values array is interpreted according to real_type
 */
scl_sparse_t scl_sparse_from_coo(
    int64_t rows,
    int64_t cols,
    const void* row_indices,
    const void* col_indices,
    const void* values,
    int64_t nnz,
    scl_value_type_t value_type,
    scl_index_type_t index_type,
    scl_layout_t layout
);

/**
 * @brief Create sparse matrix from CSR arrays
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param row_ptrs Row pointer array (length = rows + 1, always int64_t)
 * @param col_indices Column indices array
 * @param values Value array
 * @param real_type Value precision
 * @param index_type Index precision for col_indices
 * @return New CSR sparse handle, or SCL_NULL_SPARSE on error
 *
 * @note Data is copied; caller retains ownership of input arrays
 */
scl_sparse_t scl_sparse_from_csr(
    int64_t rows,
    int64_t cols,
    const int64_t* row_ptrs,
    const void* col_indices,
    const void* values,
    scl_value_type_t value_type,
    scl_index_type_t index_type
);

/**
 * @brief Create sparse matrix from CSC arrays
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param col_ptrs Column pointer array (length = cols + 1, always int64_t)
 * @param row_indices Row indices array
 * @param values Value array
 * @param real_type Value precision
 * @param index_type Index precision for row_indices
 * @return New CSC sparse handle, or SCL_NULL_SPARSE on error
 */
scl_sparse_t scl_sparse_from_csc(
    int64_t rows,
    int64_t cols,
    const int64_t* col_ptrs,
    const void* row_indices,
    const void* values,
    scl_value_type_t value_type,
    scl_index_type_t index_type
);

/**
 * @brief Create sparse matrix from dense array
 *
 * Non-zero elements (|x| > tolerance) are extracted.
 *
 * @param rows Number of rows
 * @param cols Number of columns
 * @param data Dense data array (row-major, length = rows * cols)
 * @param real_type Value precision
 * @param index_type Index precision
 * @param layout Output layout
 * @param tolerance Values with |x| <= tolerance are treated as zero (use 0.0 for exact)
 * @return New sparse handle, or SCL_NULL_SPARSE on error
 */
scl_sparse_t scl_sparse_from_dense(
    int64_t rows,
    int64_t cols,
    const void* data,
    scl_value_type_t value_type,
    scl_index_type_t index_type,
    scl_layout_t layout,
    double tolerance
);

/* ============================================================================
 * SECTION 2: Destruction
 * ============================================================================ */

/**
 * @brief Destroy a sparse matrix handle
 *
 * Releases all resources associated with the handle.
 * Safe to call with SCL_NULL_SPARSE (no-op).
 *
 * @param handle Sparse handle to destroy
 */
void scl_sparse_destroy(scl_sparse_t handle);

/* ============================================================================
 * SECTION 3: Property Queries
 * ============================================================================ */

/**
 * @brief Get number of rows
 * @param handle Sparse handle
 * @return Number of rows, or 0 if handle is null
 */
int64_t scl_sparse_rows(scl_sparse_t handle);

/**
 * @brief Get number of columns
 * @param handle Sparse handle
 * @return Number of columns, or 0 if handle is null
 */
int64_t scl_sparse_cols(scl_sparse_t handle);

/**
 * @brief Get total number of non-zero elements
 * @param handle Sparse handle
 * @return NNZ count, or 0 if handle is null
 */
int64_t scl_sparse_nnz(scl_sparse_t handle);

/**
 * @brief Get matrix density (nnz / (rows * cols))
 * @param handle Sparse handle
 * @return Density in [0, 1], or 0.0 if handle is null
 */
double scl_sparse_density(scl_sparse_t handle);

/**
 * @brief Get sparsity (1 - density)
 * @param handle Sparse handle
 * @return Sparsity in [0, 1], or 1.0 if handle is null
 */
double scl_sparse_sparsity(scl_sparse_t handle);

/**
 * @brief Check if matrix is empty (zero dimensions or no NNZ)
 * @param handle Sparse handle
 * @return 1 if empty, 0 otherwise
 */
int32_t scl_sparse_is_empty(scl_sparse_t handle);

/**
 * @brief Get value type (Real/Int/Uint + precision)
 * @param handle Sparse handle
 * @return Value type enumeration (SCL_REAL32, SCL_INT32, SCL_UINT8, etc.)
 */
scl_value_type_t scl_sparse_value_type(scl_sparse_t handle);

/**
 * @brief Get value precision type (legacy, backward compatibility)
 * @param handle Sparse handle
 * @return Real type enumeration (same as scl_sparse_value_type for Real types)
 */
scl_real_type_t scl_sparse_real_type(scl_sparse_t handle);

/**
 * @brief Get index precision type
 * @param handle Sparse handle
 * @return Index type enumeration
 */
scl_index_type_t scl_sparse_index_type(scl_sparse_t handle);

/**
 * @brief Get storage layout
 * @param handle Sparse handle
 * @return Layout enumeration (CSR or CSC)
 */
scl_layout_t scl_sparse_layout(scl_sparse_t handle);

/* ============================================================================
 * SECTION 4: Row/Column Data Access
 * ============================================================================ */

/**
 * @brief Get number of non-zeros in a specific row (CSR) or column (CSC)
 *
 * @param handle Sparse handle
 * @param idx Row index (CSR) or column index (CSC)
 * @return Number of non-zeros, or -1 on error
 */
int64_t scl_sparse_primary_length(scl_sparse_t handle, int64_t idx);

/**
 * @brief Get row data for CSR matrix
 *
 * @param handle Sparse handle (must be CSR)
 * @param row Row index
 * @param[out] values Pointer to receive values array (read-only)
 * @param[out] indices Pointer to receive column indices array (read-only)
 * @param[out] length Pointer to receive row length
 * @return Error code (0 = success)
 *
 * @note Returned pointers are valid until handle is destroyed or modified
 */
int32_t scl_sparse_row_data(
    scl_sparse_t handle,
    int64_t row,
    const void** values,
    const void** indices,
    int64_t* length
);

/**
 * @brief Get column data for CSC matrix
 *
 * @param handle Sparse handle (must be CSC)
 * @param col Column index
 * @param[out] values Pointer to receive values array (read-only)
 * @param[out] indices Pointer to receive row indices array (read-only)
 * @param[out] length Pointer to receive column length
 * @return Error code (0 = success)
 */
int32_t scl_sparse_col_data(
    scl_sparse_t handle,
    int64_t col,
    const void** values,
    const void** indices,
    int64_t* length
);

/* ============================================================================
 * SECTION 5: Element Access
 * ============================================================================ */

/**
 * @brief Get element value at (row, col)
 *
 * Uses binary search. O(log(nnz_per_row)) for CSR, O(log(nnz_per_col)) for CSC.
 *
 * @param handle Sparse handle
 * @param row Row index
 * @param col Column index
 * @param[out] value Pointer to receive value (type matches real_type)
 * @return Error code (0 = success, element found; error if out of bounds)
 *
 * @note Returns 0.0 for non-existent elements (structural zeros)
 */
int32_t scl_sparse_at(
    scl_sparse_t handle,
    int64_t row,
    int64_t col,
    void* value
);

/**
 * @brief Get element value as double (convenience function)
 *
 * @param handle Sparse handle
 * @param row Row index
 * @param col Column index
 * @return Element value as double, or 0.0 if not found/error
 */
double scl_sparse_get(scl_sparse_t handle, int64_t row, int64_t col);

/**
 * @brief Check if element exists at (row, col)
 *
 * @param handle Sparse handle
 * @param row Row index
 * @param col Column index
 * @return 1 if element exists (non-zero), 0 otherwise
 */
int32_t scl_sparse_exists(scl_sparse_t handle, int64_t row, int64_t col);

/* ============================================================================
 * SECTION 6: Clone and Transform
 * ============================================================================ */

/**
 * @brief Create a deep copy of sparse matrix
 *
 * @param handle Source sparse handle
 * @return New independent copy, or SCL_NULL_SPARSE on error
 */
scl_sparse_t scl_sparse_clone(scl_sparse_t handle);

/**
 * @brief Create a deep copy with specific buffer strategy
 *
 * @param handle Source sparse handle
 * @param config Buffer allocation configuration
 * @return New copy with specified memory layout, or SCL_NULL_SPARSE on error
 */
scl_sparse_t scl_sparse_clone_with_strategy(
    scl_sparse_t handle,
    scl_buffer_config_t config
);

/**
 * @brief Transpose matrix (CSR <-> CSC conversion)
 *
 * @param handle Source sparse handle
 * @return New transposed matrix (layout swapped), or SCL_NULL_SPARSE on error
 *
 * @note CSR input produces CSC output and vice versa
 */
scl_sparse_t scl_sparse_transpose(scl_sparse_t handle);

/* ============================================================================
 * SECTION 7: In-Place Operations
 * ============================================================================ */

/**
 * @brief Scale all values by a scalar (in-place)
 *
 * @param handle Sparse handle
 * @param scalar Scale factor
 * @return Error code (0 = success)
 */
int32_t scl_sparse_scale(scl_sparse_t handle, double scalar);

/**
 * @brief Sort indices within each row/column (in-place)
 *
 * Ensures indices are in ascending order (required for efficient lookup).
 *
 * @param handle Sparse handle
 * @return Error code (0 = success)
 *
 * @note Most creation functions already produce sorted output
 */
int32_t scl_sparse_sort_indices(scl_sparse_t handle);

/**
 * @brief Check if indices are sorted
 *
 * @param handle Sparse handle
 * @return 1 if sorted, 0 if not sorted or error
 */
int32_t scl_sparse_is_sorted(scl_sparse_t handle);

/* ============================================================================
 * SECTION 8: Slicing Operations (Zero-Copy When Possible)
 * ============================================================================ */

/**
 * @brief Slice rows from CSR matrix
 *
 * Creates a view that shares data with original (zero-copy for CSR).
 *
 * @param handle Sparse CSR handle
 * @param start Start row (inclusive)
 * @param end End row (exclusive)
 * @return Sliced matrix, or SCL_NULL_SPARSE on error
 *
 * @note For CSC, this requires filtering and copying data
 */
scl_sparse_t scl_sparse_row_slice(
    scl_sparse_t handle,
    int64_t start,
    int64_t end
);

/**
 * @brief Slice columns from CSC matrix
 *
 * Creates a view that shares data with original (zero-copy for CSC).
 *
 * @param handle Sparse CSC handle
 * @param start Start column (inclusive)
 * @param end End column (exclusive)
 * @return Sliced matrix, or SCL_NULL_SPARSE on error
 *
 * @note For CSR, this requires filtering and copying data
 */
scl_sparse_t scl_sparse_col_slice(
    scl_sparse_t handle,
    int64_t start,
    int64_t end
);

/**
 * @brief Select specific rows by indices
 *
 * @param handle Sparse handle
 * @param row_indices Array of row indices to select
 * @param count Number of indices
 * @return Matrix with selected rows, or SCL_NULL_SPARSE on error
 */
scl_sparse_t scl_sparse_row_select(
    scl_sparse_t handle,
    const int64_t* row_indices,
    int64_t count
);

/**
 * @brief Select specific columns by indices
 *
 * @param handle Sparse handle
 * @param col_indices Array of column indices to select
 * @param count Number of indices
 * @return Matrix with selected columns, or SCL_NULL_SPARSE on error
 */
scl_sparse_t scl_sparse_col_select(
    scl_sparse_t handle,
    const int64_t* col_indices,
    int64_t count
);

/* ============================================================================
 * SECTION 9: Export Functions
 * ============================================================================ */

/**
 * @brief Export to dense array (row-major)
 *
 * @param handle Sparse handle
 * @param[out] data Output buffer (must be pre-allocated: rows * cols * sizeof(real_type))
 * @return Error code (0 = success)
 */
int32_t scl_sparse_to_dense(scl_sparse_t handle, void* data);

/**
 * @brief Get required buffer size for dense export
 *
 * @param handle Sparse handle
 * @return Buffer size in bytes
 */
size_t scl_sparse_dense_buffer_size(scl_sparse_t handle);

/**
 * @brief Export to COO format
 *
 * @param handle Sparse handle
 * @param[out] row_indices Output row indices (must be pre-allocated: nnz * index_size)
 * @param[out] col_indices Output column indices (must be pre-allocated: nnz * index_size)
 * @param[out] values Output values (must be pre-allocated: nnz * real_size)
 * @return Error code (0 = success)
 */
int32_t scl_sparse_to_coo(
    scl_sparse_t handle,
    void* row_indices,
    void* col_indices,
    void* values
);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* SCL_API_CORE_SPARSE_H_ */

