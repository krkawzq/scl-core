#pragma once

// =============================================================================
// FILE: scl/binding/c_api/sparse_kernel/sparse_kernel.h
// BRIEF: C API for sparse matrix statistics and utilities
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Primary Statistics (Rows for CSR, Cols for CSC)
// =============================================================================

scl_error_t scl_sparse_kernel_primary_sums(
    scl_sparse_t matrix,
    scl_real_t* output,
    scl_size_t primary_dim
);

scl_error_t scl_sparse_kernel_primary_means(
    scl_sparse_t matrix,
    scl_real_t* output,
    scl_size_t primary_dim
);

scl_error_t scl_sparse_kernel_primary_variances(
    scl_sparse_t matrix,
    scl_real_t* output,
    scl_size_t primary_dim,
    int ddof
);

scl_error_t scl_sparse_kernel_primary_nnz(
    scl_sparse_t matrix,
    scl_index_t* output,
    scl_size_t primary_dim
);

// =============================================================================
// Matrix Operations
// =============================================================================

scl_error_t scl_sparse_kernel_eliminate_zeros(
    scl_sparse_t matrix,
    scl_sparse_t* out_matrix,
    scl_real_t tolerance
);

scl_error_t scl_sparse_kernel_prune(
    scl_sparse_t matrix,
    scl_sparse_t* out_matrix,
    scl_real_t threshold,
    int keep_structure
);

#ifdef __cplusplus
}
#endif
