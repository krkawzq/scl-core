#pragma once

// =============================================================================
// FILE: scl/binding/c_api/core/sparse.h
// BRIEF: C API for sparse matrix operations (CSR/CSC formats)
// =============================================================================
//
// DESIGN PHILOSOPHY:
//   - Support both CSR (Compressed Sparse Row) and CSC (Compressed Sparse Column)
//   - Block-allocated storage for efficient memory management
//   - Zero-copy slicing via registry-based reference counting
//   - Compatible with traditional CSR/CSC for interoperability
//
// STORAGE FORMATS:
//   CSR (Compressed Sparse Row):
//     - Efficient for row-wise operations and matrix-vector multiplication
//     - Optimal for: iterating rows, row slicing, sparse A * dense x
//   
//   CSC (Compressed Sparse Column):
//     - Efficient for column-wise operations
//     - Optimal for: iterating columns, column slicing, dense x * sparse A
//
// BLOCK ALLOCATION:
//   - Traditional CSR/CSC uses single contiguous arrays
//   - SCL uses block allocation to enable:
//     * Efficient append/grow operations
//     * Zero-copy row/column slicing
//     * Better memory reuse for large matrices
//   - Use SCL_BLOCK_STRATEGY_CONTIGUOUS for traditional format
//
// MEMORY SEMANTICS:
//   - All matrices managed by internal Registry
//   - Slice/view operations share underlying data (zero-copy)
//   - Reference counting ensures safe shared access
//   - Use scl_sparse_clone() for independent copy
//
// THREAD SAFETY:
//   - All API functions are thread-safe (thread-local error reporting)
//   - Matrix data is NOT thread-safe (caller must synchronize)
//   - Multiple threads can read concurrently (if matrix immutable)
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Block Allocation Strategy
// =============================================================================

/// @brief Block allocation strategy for sparse matrices
typedef enum {
    /// @brief Single contiguous block (traditional CSR/CSC format)
    /// @details Compatible with SciPy, NumPy, and other libraries
    SCL_BLOCK_STRATEGY_CONTIGUOUS = 0,
    
    /// @brief Small blocks (1K-16K elements per block)
    /// @details Good for matrices with frequent modifications
    SCL_BLOCK_STRATEGY_SMALL = 1,
    
    /// @brief Large blocks (64K-1M elements per block)
    /// @details Good for large read-only matrices
    SCL_BLOCK_STRATEGY_LARGE = 2,
    
    /// @brief Adaptive strategy (auto-tune based on matrix properties)
    /// @details Recommended default for most use cases
    SCL_BLOCK_STRATEGY_ADAPTIVE = 3
} scl_block_strategy_t;

// =============================================================================
// Layout Information
// =============================================================================

/// @brief Memory layout information for sparse matrices
typedef struct {
    scl_index_t data_block_count;    ///< Number of data blocks
    scl_index_t index_block_count;   ///< Number of index blocks
    scl_size_t data_bytes;            ///< Total bytes for values
    scl_size_t index_bytes;           ///< Total bytes for indices
    scl_size_t metadata_bytes;        ///< Total bytes for metadata
    scl_bool_t is_contiguous;         ///< Single contiguous block?
    scl_bool_t is_traditional_format; ///< Compatible with traditional CSR/CSC?
} scl_sparse_layout_info_t;

// =============================================================================
// Lifecycle Management
// =============================================================================

/// @brief Create sparse matrix from traditional CSR/CSC format (deep copy)
/// @param[out] out Output matrix handle (non-null)
/// @param[in] rows Number of rows (> 0)
/// @param[in] cols Number of columns (> 0)
/// @param[in] nnz Number of non-zeros (>= 0)
/// @param[in] indptr Offset array [rows+1] for CSR or [cols+1] for CSC (non-null)
/// @param[in] indices Column (CSR) or row (CSC) indices [nnz] (non-null)
/// @param[in] data Non-zero values [nnz] (non-null)
/// @param[in] is_csr SCL_TRUE for CSR, SCL_FALSE for CSC
/// @return SCL_OK on success, error code otherwise
/// @note Copies all data (safe for temporary inputs)
/// @note Caller must call scl_sparse_destroy() when done
scl_error_t scl_sparse_create(
    scl_sparse_t* out,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    const scl_index_t* indptr,
    const scl_index_t* indices,
    const scl_real_t* data,
    scl_bool_t is_csr
);

/// @brief Create sparse matrix with specific block strategy
/// @param[out] out Output matrix handle (non-null)
/// @param[in] rows Number of rows (> 0)
/// @param[in] cols Number of columns (> 0)
/// @param[in] nnz Number of non-zeros (>= 0)
/// @param[in] indptr Offset array (non-null)
/// @param[in] indices Index array (non-null)
/// @param[in] data Value array (non-null)
/// @param[in] is_csr SCL_TRUE for CSR, SCL_FALSE for CSC
/// @param[in] strategy Block allocation strategy
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_sparse_create_with_strategy(
    scl_sparse_t* out,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    const scl_index_t* indptr,
    const scl_index_t* indices,
    const scl_real_t* data,
    scl_bool_t is_csr,
    scl_block_strategy_t strategy
);

/// @brief Wrap external data as zero-copy view (does not copy)
/// @param[out] out Output matrix handle (non-null)
/// @param[in] rows Number of rows (> 0)
/// @param[in] cols Number of columns (> 0)
/// @param[in] nnz Number of non-zeros (>= 0)
/// @param[in,out] indptr Mutable offset array (caller manages lifetime)
/// @param[in,out] indices Mutable index array (caller manages lifetime)
/// @param[in,out] data Mutable value array (caller manages lifetime)
/// @param[in] is_csr SCL_TRUE for CSR, SCL_FALSE for CSC
/// @return SCL_OK on success, error code otherwise
/// @note Does NOT copy data - caller must keep arrays alive
/// @note Use scl_sparse_wrap_and_own() to transfer ownership
scl_error_t scl_sparse_wrap(
    scl_sparse_t* out,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_index_t* indptr,
    scl_index_t* indices,
    scl_real_t* data,
    scl_bool_t is_csr
);

/// @brief Wrap and transfer ownership to SCL (for NumPy integration)
/// @param[out] out Output matrix handle (non-null)
/// @param[in] rows Number of rows (> 0)
/// @param[in] cols Number of columns (> 0)
/// @param[in] nnz Number of non-zeros (>= 0)
/// @param[in,out] indptr Offset array (ownership transferred)
/// @param[in,out] indices Index array (ownership transferred)
/// @param[in,out] data Value array (ownership transferred)
/// @param[in] is_csr SCL_TRUE for CSR, SCL_FALSE for CSC
/// @return SCL_OK on success, error code otherwise
/// @note SCL takes ownership and will free arrays on destroy
/// @note Arrays must be allocated with new[] (will use delete[])
scl_error_t scl_sparse_wrap_and_own(
    scl_sparse_t* out,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_index_t* indptr,
    scl_index_t* indices,
    scl_real_t* data,
    scl_bool_t is_csr
);

/// @brief Clone sparse matrix (deep copy)
/// @param[in] src Source matrix (non-null)
/// @param[out] out Output matrix (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note Creates independent copy with separate memory
/// @note Preserves block structure of source
scl_error_t scl_sparse_clone(scl_sparse_t src, scl_sparse_t* out);

/// @brief Destroy sparse matrix and release resources
/// @param[in,out] matrix Pointer to matrix handle (may be null)
/// @return SCL_OK on success, error code otherwise
/// @note Sets *matrix to NULL after destruction
/// @note Safe to call with NULL or already-destroyed handle
scl_error_t scl_sparse_destroy(scl_sparse_t* matrix);

// =============================================================================
// Property Queries
// =============================================================================

/// @brief Get number of rows
scl_error_t scl_sparse_rows(scl_sparse_t matrix, scl_index_t* out);

/// @brief Get number of columns
scl_error_t scl_sparse_cols(scl_sparse_t matrix, scl_index_t* out);

/// @brief Get number of non-zeros
scl_error_t scl_sparse_nnz(scl_sparse_t matrix, scl_index_t* out);

/// @brief Check if matrix is in CSR format
scl_error_t scl_sparse_is_csr(scl_sparse_t matrix, scl_bool_t* out);

/// @brief Check if matrix is in CSC format
scl_error_t scl_sparse_is_csc(scl_sparse_t matrix, scl_bool_t* out);

/// @brief Check if matrix is in valid state
scl_error_t scl_sparse_is_valid(scl_sparse_t matrix, scl_bool_t* out);

/// @brief Check if matrix is contiguous (single block)
scl_error_t scl_sparse_is_contiguous(scl_sparse_t matrix, scl_bool_t* out);

/// @brief Get layout information
scl_error_t scl_sparse_layout_info(
    scl_sparse_t matrix,
    scl_sparse_layout_info_t* info
);

// =============================================================================
// Data Export
// =============================================================================

/// @brief Export to traditional CSR/CSC format (copies data)
/// @param[in] matrix Matrix handle (non-null)
/// @param[out] indptr Caller-allocated offset array [primary_dim+1]
/// @param[out] indices Caller-allocated index array [nnz]
/// @param[out] data Caller-allocated value array [nnz]
/// @return SCL_OK on success, error code otherwise
/// @note Caller responsible for allocating arrays
/// @note primary_dim = rows (CSR) or cols (CSC)
scl_error_t scl_sparse_export(
    scl_sparse_t matrix,
    scl_index_t* indptr,
    scl_index_t* indices,
    scl_real_t* data
);

/// @brief Get read-only pointer to lengths array (zero-copy)
/// @param[in] matrix Matrix handle (non-null)
/// @param[out] lengths Pointer to internal lengths array (nnz per row/col)
/// @param[out] lengths_size Size of lengths array (= rows for CSR, cols for CSC)
/// @return SCL_OK on success, error code otherwise
/// @note Sparse stores lengths[], not traditional indptr[]. Use scl_sparse_export()
///       if you need indptr format.
/// @note Pointers valid while matrix alive
SCL_EXPORT scl_error_t scl_sparse_get_lengths(
    scl_sparse_t matrix,
    const scl_index_t** lengths,
    scl_size_t* lengths_size
);

/// @brief Get read-only pointer to index array (zero-copy)
/// @param[in] matrix Matrix handle (non-null, must be contiguous)
/// @param[out] indices Pointer to internal index array
/// @param[out] indices_size Size of index array (= nnz)
/// @return SCL_OK on success, error code otherwise
/// @note Only valid if matrix is contiguous
/// @note Pointers valid while matrix alive
SCL_EXPORT scl_error_t scl_sparse_get_indices(
    scl_sparse_t matrix,
    const scl_index_t** indices,
    scl_size_t* indices_size
);

/// @brief Get read-only pointer to value array (zero-copy)
/// @param[in] matrix Matrix handle (non-null, must be contiguous)
/// @param[out] data Pointer to internal value array
/// @param[out] data_size Size of value array (= nnz)
/// @return SCL_OK on success, error code otherwise
/// @note Only valid if matrix is contiguous
/// @note Pointers valid while matrix alive
SCL_EXPORT scl_error_t scl_sparse_get_data(
    scl_sparse_t matrix,
    const scl_real_t** data,
    scl_size_t* data_size
);

// =============================================================================
// Format Conversion
// =============================================================================

/// @brief Transpose matrix (CSR <-> CSC)
/// @param[in] src Source matrix (non-null)
/// @param[out] out Transposed matrix (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note CSR becomes CSC and vice versa
scl_error_t scl_sparse_transpose(scl_sparse_t src, scl_sparse_t* out);

/// @brief Convert to contiguous storage (traditional format)
/// @param[in] src Source matrix (non-null)
/// @param[out] out Contiguous matrix (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note Result compatible with SciPy/NumPy sparse matrices
scl_error_t scl_sparse_to_contiguous(scl_sparse_t src, scl_sparse_t* out);

// =============================================================================
// COO Format Support
// =============================================================================

/// @brief Create sparse matrix from COO (Coordinate) format
/// @param[out] out Output matrix (non-null)
/// @param[in] rows Number of rows (> 0)
/// @param[in] cols Number of columns (> 0)
/// @param[in] nnz Number of non-zeros (>= 0)
/// @param[in] row_indices Row indices [nnz]
/// @param[in] col_indices Column indices [nnz]
/// @param[in] values Values [nnz]
/// @param[in] is_csr Output format: SCL_TRUE=CSR, SCL_FALSE=CSC
/// @param[in] strategy Block allocation strategy
/// @return SCL_OK on success, error code otherwise
/// @note COO format: (row[i], col[i], value[i]) tuples
/// @note Input may have duplicate entries (will be summed)
scl_error_t scl_sparse_from_coo(
    scl_sparse_t* out,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    const scl_index_t* row_indices,
    const scl_index_t* col_indices,
    const scl_real_t* values,
    scl_bool_t is_csr,
    scl_block_strategy_t strategy
);

/// @brief Export to COO format (allocates new arrays)
/// @param[in] matrix Matrix handle (non-null)
/// @param[out] row_indices Row indices array [nnz] (SCL allocates)
/// @param[out] col_indices Column indices array [nnz] (SCL allocates)
/// @param[out] values Values array [nnz] (SCL allocates)
/// @param[out] nnz Number of non-zeros
/// @return SCL_OK on success, error code otherwise
/// @note SCL allocates arrays - caller must free (implementation-defined)
scl_error_t scl_sparse_to_coo(
    scl_sparse_t matrix,
    scl_index_t** row_indices,
    scl_index_t** col_indices,
    scl_real_t** values,
    scl_index_t* nnz
);

// =============================================================================
// Slicing Operations
// =============================================================================

/// @brief Row range slice (zero-copy view)
/// @param[in] src Source CSR matrix (non-null)
/// @param[in] start Start row index [0, rows)
/// @param[in] end End row index (start, rows]
/// @param[out] out Slice handle (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note Only for CSR matrices (error for CSC)
/// @note Zero-copy: shares data with source
scl_error_t scl_sparse_row_range_view(
    scl_sparse_t src,
    scl_index_t start,
    scl_index_t end,
    scl_sparse_t* out
);

/// @brief Row range slice with copy
/// @param[in] src Source CSR matrix (non-null)
/// @param[in] start Start row index
/// @param[in] end End row index
/// @param[in] strategy Block allocation strategy
/// @param[out] out Slice handle (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note Creates independent copy
scl_error_t scl_sparse_row_range_copy(
    scl_sparse_t src,
    scl_index_t start,
    scl_index_t end,
    scl_block_strategy_t strategy,
    scl_sparse_t* out
);

/// @brief Row slice (arbitrary indices, zero-copy if possible)
/// @param[in] src Source CSR matrix (non-null)
/// @param[in] row_indices Array of row indices to select
/// @param[in] n_rows Number of rows to select
/// @param[out] out Slice handle (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_sparse_slice_rows(
    scl_sparse_t src,
    const scl_index_t* row_indices,
    scl_size_t n_rows,
    scl_sparse_t* out
);

/// @brief Row slice with copy
scl_error_t scl_sparse_row_slice_copy(
    scl_sparse_t src,
    const scl_index_t* row_indices,
    scl_size_t n_rows,
    scl_block_strategy_t strategy,
    scl_sparse_t* out
);

/// @brief Column slice (arbitrary indices)
/// @param[in] src Source matrix (non-null)
/// @param[in] col_indices Array of column indices to select
/// @param[in] n_cols Number of columns to select
/// @param[in] strategy Block allocation strategy
/// @param[out] out Slice handle (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note Efficient for CSC, requires reconstruction for CSR
scl_error_t scl_sparse_col_slice(
    scl_sparse_t src,
    const scl_index_t* col_indices,
    scl_size_t n_cols,
    scl_block_strategy_t strategy,
    scl_sparse_t* out
);

/// @brief Column slice for CSC matrices (zero-copy)
scl_error_t scl_sparse_slice_cols(
    scl_sparse_t src,
    const scl_index_t* col_indices,
    scl_size_t n_cols,
    scl_sparse_t* out
);

// =============================================================================
// Batch Operations
// =============================================================================

/// @brief Vertical stack (concatenate CSR matrices by rows)
/// @param[in] matrices Array of CSR matrices (all non-null)
/// @param[in] n_matrices Number of matrices (> 0)
/// @param[in] strategy Block allocation strategy
/// @param[out] out Stacked matrix (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note All matrices must have same number of columns
/// @note All matrices must be CSR format
scl_error_t scl_sparse_vstack(
    const scl_sparse_t* matrices,
    scl_size_t n_matrices,
    scl_block_strategy_t strategy,
    scl_sparse_t* out
);

/// @brief Horizontal stack (concatenate CSC matrices by columns)
/// @param[in] matrices Array of CSC matrices (all non-null)
/// @param[in] n_matrices Number of matrices (> 0)
/// @param[in] strategy Block allocation strategy
/// @param[out] out Stacked matrix (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note All matrices must have same number of rows
/// @note All matrices must be CSC format
scl_error_t scl_sparse_hstack(
    const scl_sparse_t* matrices,
    scl_size_t n_matrices,
    scl_block_strategy_t strategy,
    scl_sparse_t* out
);

#ifdef __cplusplus
}
#endif
