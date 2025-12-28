#pragma once

// =============================================================================
// FILE: scl/binding/c_api/neighbors/neighbors.h
// BRIEF: C API for K-nearest neighbors computation
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// K-Nearest Neighbors
// =============================================================================

// Compute squared norms for each row/column (helper function)
scl_error_t scl_neighbors_compute_norms(
    scl_sparse_t matrix,                // Expression matrix (samples x features, CSR)
    scl_real_t* norms_sq                 // Output: squared norms [n_samples]
);

// Compute KNN with precomputed norms (faster for repeated calls)
scl_error_t scl_knn(
    scl_sparse_t matrix,
    const scl_real_t* norms_sq,          // Precomputed squared norms [n_samples]
    scl_size_t n_samples,
    scl_size_t k,                        // Number of neighbors
    scl_index_t* out_indices,            // Output: neighbor indices [n_samples * k]
    scl_real_t* out_distances            // Output: neighbor distances [n_samples * k]
);

// Compute KNN without precomputed norms (computes norms internally)
scl_error_t scl_knn_simple(
    scl_sparse_t matrix,
    scl_size_t n_samples,
    scl_size_t k,
    scl_index_t* out_indices,
    scl_real_t* out_distances
);

#ifdef __cplusplus
}
#endif
