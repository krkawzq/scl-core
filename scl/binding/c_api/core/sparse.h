#pragma once

// =============================================================================
// FILE: scl/binding/c_api/core/sparse.h
// BRIEF: Safe C API for sparse matrix operations
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Lifecycle Management
// =============================================================================

// Create sparse matrix from traditional CSR/CSC format (copies data)
scl_error_t scl_sparse_create(
    scl_sparse_t* out,              // Output handle
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    const scl_index_t* indptr,      // [rows+1] for CSR, [cols+1] for CSC
    const scl_index_t* indices,     // [nnz]
    const scl_real_t* data,         // [nnz]
    int is_csr                      // 1 = CSR, 0 = CSC
);

// Create sparse matrix by wrapping external data (zero-copy, caller manages lifetime)
scl_error_t scl_sparse_wrap(
    scl_sparse_t* out,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_index_t* indptr,            // Mutable, caller owns
    scl_index_t* indices,
    scl_real_t* data,
    int is_csr
);

// Clone (deep copy)
scl_error_t scl_sparse_clone(scl_sparse_t src, scl_sparse_t* out);

// Destroy and release memory
scl_error_t scl_sparse_destroy(scl_sparse_t* matrix);

// =============================================================================
// Property Queries
// =============================================================================

scl_error_t scl_sparse_rows(scl_sparse_t matrix, scl_index_t* out);
scl_error_t scl_sparse_cols(scl_sparse_t matrix, scl_index_t* out);
scl_error_t scl_sparse_nnz(scl_sparse_t matrix, scl_index_t* out);
scl_error_t scl_sparse_is_csr(scl_sparse_t matrix, int* out);
scl_error_t scl_sparse_is_valid(scl_sparse_t matrix, int* out);

// =============================================================================
// Data Export (to traditional format)
// =============================================================================

// Get pointers to internal data (read-only, valid while matrix alive)
// Note: Only valid if matrix is in contiguous storage
scl_error_t scl_sparse_get_indptr(
    scl_sparse_t matrix,
    const scl_index_t** out,
    scl_size_t* size
);

scl_error_t scl_sparse_get_indices(
    scl_sparse_t matrix,
    const scl_index_t** out,
    scl_size_t* size
);

scl_error_t scl_sparse_get_data(
    scl_sparse_t matrix,
    const scl_real_t** out,
    scl_size_t* size
);

// Export to caller-allocated buffers (copies data)
scl_error_t scl_sparse_export(
    scl_sparse_t matrix,
    scl_index_t* indptr,            // Caller-allocated [primary_dim+1]
    scl_index_t* indices,           // Caller-allocated [nnz]
    scl_real_t* data                // Caller-allocated [nnz]
);

// =============================================================================
// Format Conversion
// =============================================================================

// Convert CSR <-> CSC
scl_error_t scl_sparse_transpose(scl_sparse_t src, scl_sparse_t* out);

// Convert to contiguous storage
scl_error_t scl_sparse_to_contiguous(scl_sparse_t src, scl_sparse_t* out);

// =============================================================================
// Advanced Lifecycle Management
// =============================================================================

// Create sparse matrix by wrapping external data with registry registration
// Unlike scl_sparse_wrap, this registers the data with registry for automatic
// lifecycle management. Use this when you want scl-core to manage the memory
// but the data was allocated externally (e.g., from NumPy).
// The arrays must have been allocated with new[] (will be deleted with delete[])
scl_error_t scl_sparse_wrap_and_own(
    scl_sparse_t* out,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    scl_index_t* indptr,            // Will be owned by scl-core
    scl_index_t* indices,           // Will be owned by scl-core
    scl_real_t* data,               // Will be owned by scl-core
    int is_csr
);

// Create a zero-copy slice/view of a sparse matrix
// The slice shares data with the original - modifications affect both
// Slice is reference-counted and safe even if original is destroyed first
scl_error_t scl_sparse_slice_rows(
    scl_sparse_t src,
    const scl_index_t* row_indices, // Array of row indices to select
    scl_size_t n_rows,              // Number of rows to select
    scl_sparse_t* out               // Output slice handle
);

scl_error_t scl_sparse_slice_cols(
    scl_sparse_t src,
    const scl_index_t* col_indices, // Array of column indices to select  
    scl_size_t n_cols,              // Number of columns to select
    scl_sparse_t* out               // Output slice handle
);

// =============================================================================
// Block Allocation Strategy
// =============================================================================

// Block allocation strategy options
typedef enum {
    SCL_BLOCK_STRATEGY_CONTIGUOUS = 0,  // Single block (traditional CSR/CSC)
    SCL_BLOCK_STRATEGY_SMALL = 1,        // Small blocks (1K-16K elements)
    SCL_BLOCK_STRATEGY_LARGE = 2,        // Large blocks (64K-1M elements)
    SCL_BLOCK_STRATEGY_ADAPTIVE = 3      // Auto-tune (default)
} scl_block_strategy_t;

// Create sparse matrix with specific block strategy
scl_error_t scl_sparse_create_with_strategy(
    scl_sparse_t* out,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    const scl_index_t* indptr,
    const scl_index_t* indices,
    const scl_real_t* data,
    int is_csr,
    scl_block_strategy_t strategy
);

// =============================================================================
// COO Format Support
// =============================================================================

// Create sparse matrix from COO (Coordinate) format
scl_error_t scl_sparse_from_coo(
    scl_sparse_t* out,
    scl_index_t rows,
    scl_index_t cols,
    scl_index_t nnz,
    const scl_index_t* row_indices,  // [nnz]
    const scl_index_t* col_indices,  // [nnz]
    const scl_real_t* values,        // [nnz]
    int is_csr,                      // Output format: 1=CSR, 0=CSC
    scl_block_strategy_t strategy
);

// Export to COO format (allocates new arrays)
scl_error_t scl_sparse_to_coo(
    scl_sparse_t matrix,
    scl_index_t** row_indices,       // Output [nnz], allocated by scl-core
    scl_index_t** col_indices,       // Output [nnz], allocated by scl-core
    scl_real_t** values,             // Output [nnz], allocated by scl-core
    scl_index_t* nnz                 // Output nnz
);

// =============================================================================
// Layout Information
// =============================================================================

// Layout information structure
typedef struct {
    scl_index_t data_block_count;
    scl_index_t index_block_count;
    scl_size_t data_bytes;
    scl_size_t index_bytes;
    scl_size_t metadata_bytes;
    int is_contiguous;
    int is_traditional_format;
} scl_sparse_layout_info_t;

// Query layout information
scl_error_t scl_sparse_layout_info(
    scl_sparse_t matrix,
    scl_sparse_layout_info_t* info
);

// Check if matrix is contiguous
scl_error_t scl_sparse_is_contiguous(
    scl_sparse_t matrix,
    int* out
);

// =============================================================================
// Advanced Slicing
// =============================================================================

// Row range slice (contiguous range) - zero-copy view
scl_error_t scl_sparse_row_range_view(
    scl_sparse_t src,
    scl_index_t start,
    scl_index_t end,
    scl_sparse_t* out
);

// Row range slice with copy (independent memory)
scl_error_t scl_sparse_row_range_copy(
    scl_sparse_t src,
    scl_index_t start,
    scl_index_t end,
    scl_block_strategy_t strategy,
    scl_sparse_t* out
);

// Row slice with copy (for arbitrary indices)
scl_error_t scl_sparse_row_slice_copy(
    scl_sparse_t src,
    const scl_index_t* row_indices,
    scl_size_t n_rows,
    scl_block_strategy_t strategy,
    scl_sparse_t* out
);

// Column slice (requires reconstruction for CSR)
scl_error_t scl_sparse_col_slice(
    scl_sparse_t src,
    const scl_index_t* col_indices,
    scl_size_t n_cols,
    scl_block_strategy_t strategy,
    scl_sparse_t* out
);

// =============================================================================
// Batch Operations
// =============================================================================

// Vertical stack (CSR matrices only)
scl_error_t scl_sparse_vstack(
    const scl_sparse_t* matrices,    // Array of CSR matrices
    scl_size_t n_matrices,
    scl_block_strategy_t strategy,
    scl_sparse_t* out
);

// Horizontal stack (CSC matrices only)
scl_error_t scl_sparse_hstack(
    const scl_sparse_t* matrices,    // Array of CSC matrices
    scl_size_t n_matrices,
    scl_block_strategy_t strategy,
    scl_sparse_t* out
);

#ifdef __cplusplus
}
#endif
