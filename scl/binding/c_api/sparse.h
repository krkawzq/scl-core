#pragma once

// =============================================================================
// FILE: scl/binding/c_api/sparse.h
// BRIEF: C API for sparse matrix statistics
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "scl/binding/c_api/core_types.h"

// Primary dimension sums
scl_error_t scl_sparse_primary_sums(
    scl_sparse_matrix_t matrix,
    scl_real_t* output,
    scl_size_t n_rows
);

// Primary dimension means
scl_error_t scl_sparse_primary_means(
    scl_sparse_matrix_t matrix,
    scl_real_t* output,
    scl_size_t n_rows
);

// Primary dimension variances
scl_error_t scl_sparse_primary_variances(
    scl_sparse_matrix_t matrix,
    scl_real_t* output,
    scl_size_t n_rows,
    int ddof
);

// Primary dimension non-zero counts
scl_error_t scl_sparse_primary_nnz(
    scl_sparse_matrix_t matrix,
    scl_index_t* output,
    scl_size_t n_rows
);

#ifdef __cplusplus
}
#endif
