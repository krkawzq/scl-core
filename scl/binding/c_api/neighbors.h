#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "core_types.h"

// =============================================================================
// FILE: scl/binding/c_api/neighbors.h
// BRIEF: C API declarations for K-Nearest Neighbors
// =============================================================================

// Compute squared norms for each row/column of sparse matrix
scl_error_t scl_compute_norms(
    scl_sparse_matrix_t matrix,        // Sparse matrix (CSR or CSC)
    scl_real_t* norms_sq               // Output squared norms [primary_dim]
);

// Compute K-nearest neighbors using Euclidean distance
scl_error_t scl_knn(
    scl_sparse_matrix_t matrix,        // Sparse matrix (CSR or CSC)
    const scl_real_t* norms_sq,        // Precomputed squared norms [primary_dim]
    scl_size_t k,                      // Number of neighbors
    scl_index_t* out_indices,          // Output neighbor indices [primary_dim * k]
    scl_real_t* out_distances           // Output distances [primary_dim * k]
);

#ifdef __cplusplus
}
#endif
