#pragma once

// =============================================================================
// FILE: scl/binding/c_api/core/dense.h
// BRIEF: C API for dense matrix view operations (row-major layout)
// =============================================================================
//
// DESIGN PHILOSOPHY:
//   - This library does NOT allocate dense matrices internally
//   - Dense matrices are always views into caller-managed memory
//   - Caller is responsible for data allocation and lifetime
//   - This matches common usage: NumPy/PyTorch allocates, SCL operates
//
// MEMORY LAYOUT:
//   - Row-major (C-style) layout: A[i][j] = data[i * stride + j]
//   - Compatible with NumPy C-contiguous arrays
//   - Supports non-contiguous views via stride parameter
//
// LIFETIME MANAGEMENT:
//   - scl_dense_wrap(): Zero-copy view (caller manages data lifetime)
//   - scl_dense_destroy(): Releases view handle only (NOT the data)
//   - Data must remain valid while view is in use
//
// THREAD SAFETY:
//   - All functions are thread-safe via thread-local error reporting
//   - Matrix data is NOT thread-safe (caller must synchronize)
//   - Read-only operations safe for concurrent reads
//
// ERROR HANDLING:
//   - All functions return scl_error_t (SCL_OK on success)
//   - Use scl_get_last_error() for error details
//   - NULL arguments always return SCL_ERROR_NULL_POINTER
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Lifecycle Management
// =============================================================================

/// @brief Create view wrapping external data (zero-copy, does not own data)
/// @param[out] out Output view handle (non-null)
/// @param[in] rows Number of rows (> 0)
/// @param[in] cols Number of columns (> 0)
/// @param[in,out] data Mutable data pointer (non-null, caller manages lifetime)
/// @param[in] stride Row stride (>= cols, usually equals cols for contiguous)
/// @return SCL_OK on success, error code otherwise
/// @note Does NOT copy or own data - caller must keep data alive
/// @note Stride allows non-contiguous sub-matrices
/// @note Caller must call scl_dense_destroy() when done (frees handle only)
SCL_EXPORT scl_error_t scl_dense_wrap(
    scl_dense_t* out,
    scl_index_t rows,
    scl_index_t cols,
    scl_real_t* data,
    scl_index_t stride
);

/// @brief Destroy dense view and release handle
/// @param[in,out] matrix Pointer to view handle (may be null)
/// @return SCL_OK on success, error code otherwise
/// @note Sets *matrix to NULL after destruction
/// @note Safe to call with NULL or already-destroyed handle
/// @note Does NOT free the underlying data (view only)
SCL_EXPORT scl_error_t scl_dense_destroy(scl_dense_t* matrix);

// =============================================================================
// Property Queries
// =============================================================================

/// @brief Get number of rows
/// @param[in] matrix View handle (non-null)
/// @param[out] out Number of rows (non-null)
/// @return SCL_OK on success, error code otherwise
SCL_EXPORT scl_error_t scl_dense_rows(scl_dense_t matrix, scl_index_t* out);

/// @brief Get number of columns
/// @param[in] matrix View handle (non-null)
/// @param[out] out Number of columns (non-null)
/// @return SCL_OK on success, error code otherwise
SCL_EXPORT scl_error_t scl_dense_cols(scl_dense_t matrix, scl_index_t* out);

/// @brief Get row stride
/// @param[in] matrix View handle (non-null)
/// @param[out] out Row stride (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note Stride is distance between consecutive rows
/// @note For contiguous matrices: stride == cols
SCL_EXPORT scl_error_t scl_dense_stride(scl_dense_t matrix, scl_index_t* out);

/// @brief Get total number of elements
/// @param[in] matrix View handle (non-null)
/// @param[out] out Total elements = rows * cols (non-null)
/// @return SCL_OK on success, error code otherwise
SCL_EXPORT scl_error_t scl_dense_size(scl_dense_t matrix, scl_size_t* out);

/// @brief Check if view is in valid state
/// @param[in] matrix View handle (non-null)
/// @param[out] out SCL_TRUE if valid, SCL_FALSE otherwise (non-null)
/// @return SCL_OK on success, error code otherwise
SCL_EXPORT scl_error_t scl_dense_is_valid(scl_dense_t matrix, scl_bool_t* out);

/// @brief Check if view is contiguous (stride == cols)
/// @param[in] matrix View handle (non-null)
/// @param[out] out SCL_TRUE if contiguous, SCL_FALSE otherwise (non-null)
/// @return SCL_OK on success, error code otherwise
SCL_EXPORT scl_error_t scl_dense_is_contiguous(scl_dense_t matrix, scl_bool_t* out);

// =============================================================================
// Data Access
// =============================================================================

/// @brief Get pointer to internal data
/// @param[in] matrix View handle (non-null)
/// @param[out] out Pointer to data (non-null)
/// @param[out] size Total accessible elements = rows * stride (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note Returned pointer valid while underlying data alive
/// @note For non-contiguous views, use stride to access elements correctly
SCL_EXPORT scl_error_t scl_dense_get_data(
    scl_dense_t matrix,
    const scl_real_t** out,
    scl_size_t* size
);

/// @brief Get element at (row, col)
/// @param[in] matrix View handle (non-null)
/// @param[in] row Row index [0, rows)
/// @param[in] col Column index [0, cols)
/// @param[out] out Element value (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note Returns SCL_ERROR_INDEX_OUT_OF_BOUNDS if indices invalid
SCL_EXPORT scl_error_t scl_dense_get(
    scl_dense_t matrix,
    scl_index_t row,
    scl_index_t col,
    scl_real_t* out
);

/// @brief Set element at (row, col)
/// @param[in] matrix View handle (non-null)
/// @param[in] row Row index [0, rows)
/// @param[in] col Column index [0, cols)
/// @param[in] value New element value
/// @return SCL_OK on success, error code otherwise
/// @note Returns SCL_ERROR_INDEX_OUT_OF_BOUNDS if indices invalid
SCL_EXPORT scl_error_t scl_dense_set(
    scl_dense_t matrix,
    scl_index_t row,
    scl_index_t col,
    scl_real_t value
);

/// @brief Export to caller-allocated buffer (copies data)
/// @param[in] matrix View handle (non-null)
/// @param[out] data Caller-allocated buffer [rows * cols] (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note Always copies data in row-major contiguous format
/// @note Caller responsible for allocating rows * cols elements
SCL_EXPORT scl_error_t scl_dense_export(
    scl_dense_t matrix,
    scl_real_t* data
);

// =============================================================================
// In-Place Operations
// =============================================================================

/// @brief Fill view with scalar value
/// @param[in] matrix View handle (non-null)
/// @param[in] value Fill value
/// @return SCL_OK on success, error code otherwise
/// @note Modifies underlying data in-place
SCL_EXPORT scl_error_t scl_dense_fill(
    scl_dense_t matrix,
    scl_real_t value
);

#ifdef __cplusplus
}
#endif
