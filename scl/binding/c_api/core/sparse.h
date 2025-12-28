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

#ifdef __cplusplus
}
#endif
