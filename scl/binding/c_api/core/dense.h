#pragma once

// =============================================================================
// FILE: scl/binding/c_api/core/dense.h
// BRIEF: Safe C API for dense matrix operations
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Lifecycle Management
// =============================================================================

// Create dense matrix from row-major array (copies data)
scl_error_t scl_dense_create(
    scl_dense_t* out,
    scl_index_t rows,
    scl_index_t cols,
    const scl_real_t* data        // Row-major [rows * cols]
);

// Create dense matrix by wrapping external data (zero-copy, caller manages lifetime)
scl_error_t scl_dense_wrap(
    scl_dense_t* out,
    scl_index_t rows,
    scl_index_t cols,
    scl_real_t* data,             // Mutable, caller owns
    scl_index_t stride            // Row stride (usually == cols)
);

// Clone (deep copy)
scl_error_t scl_dense_clone(scl_dense_t src, scl_dense_t* out);

// Destroy and release memory
scl_error_t scl_dense_destroy(scl_dense_t* matrix);

// =============================================================================
// Property Queries
// =============================================================================

scl_error_t scl_dense_rows(scl_dense_t matrix, scl_index_t* out);
scl_error_t scl_dense_cols(scl_dense_t matrix, scl_index_t* out);
scl_error_t scl_dense_stride(scl_dense_t matrix, scl_index_t* out);
scl_error_t scl_dense_is_valid(scl_dense_t matrix, int* out);

// =============================================================================
// Data Access
// =============================================================================

// Get pointer to internal data (read-only, valid while matrix alive)
scl_error_t scl_dense_get_data(
    scl_dense_t matrix,
    const scl_real_t** out,
    scl_size_t* size              // Total size in elements
);

// Export to caller-allocated buffer (copies data)
scl_error_t scl_dense_export(
    scl_dense_t matrix,
    scl_real_t* data              // Caller-allocated [rows * cols]
);

// =============================================================================
// Conversion
// =============================================================================

// Convert to sparse matrix (CSR or CSC)
scl_error_t scl_dense_to_sparse(
    scl_dense_t src,
    scl_sparse_t* out,
    int is_csr,                   // 1 = CSR, 0 = CSC
    scl_real_t epsilon            // Threshold for zero (e.g., 1e-10)
);

#ifdef __cplusplus
}
#endif
