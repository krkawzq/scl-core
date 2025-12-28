#pragma once

// =============================================================================
// FILE: scl/binding/c_api/kernels/merge.h
// BRIEF: C API for matrix merging operations
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "../core.h"

// =============================================================================
// Vertical Stack (vstack)
// =============================================================================

scl_error_t scl_merge_vstack(
    scl_sparse_matrix_t matrix1,      // First matrix (CSR)
    scl_sparse_matrix_t matrix2,      // Second matrix (CSR)
    scl_sparse_matrix_t* result       // Output: stacked matrix
);

// =============================================================================
// Horizontal Stack (hstack)
// =============================================================================

scl_error_t scl_merge_hstack(
    scl_sparse_matrix_t matrix1,      // First matrix (CSR)
    scl_sparse_matrix_t matrix2,      // Second matrix (CSR)
    scl_sparse_matrix_t* result       // Output: stacked matrix
);

#ifdef __cplusplus
}
#endif
