#pragma once

// =============================================================================
// FILE: scl/binding/c_api/bbknn.h
// BRIEF: C API for Batch Balanced KNN
// =============================================================================

#include "scl/binding/c_api/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Batch Balanced KNN
// =============================================================================

// Compute BBKNN (Batch Balanced KNN)
// matrix: Sparse matrix handle (CSR format)
// batch_labels: Batch labels for each sample [n_samples]
// n_batches: Number of batches
// k: Number of neighbors per batch
// out_indices: Output neighbor indices [n_samples * n_batches * k]
// out_distances: Output distances [n_samples * n_batches * k]
// Returns: Error code
scl_error_t scl_bbknn(
    scl_sparse_matrix_t matrix,
    const int32_t* batch_labels,
    scl_size_t n_batches,
    scl_size_t k,
    scl_index_t* out_indices,
    scl_real_t* out_distances
);

#ifdef __cplusplus
}
#endif
