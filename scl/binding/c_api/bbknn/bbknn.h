#pragma once

// =============================================================================
// FILE: scl/binding/c_api/bbknn/bbknn.h
// BRIEF: C API for Batch Balanced KNN (BBKNN)
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Batch Balanced KNN
// =============================================================================

// Compute BBKNN: for each cell, find k nearest neighbors within each batch
// Output: out_indices [n_cells * n_batches * k], out_distances [n_cells * n_batches * k]
scl_error_t scl_bbknn(
    scl_sparse_t matrix,              // Expression matrix (cells x genes, CSR)
    const int32_t* batch_labels,      // Batch labels [n_cells]
    scl_size_t n_cells,
    scl_size_t n_batches,
    scl_size_t k,                     // Number of neighbors per batch
    scl_index_t* out_indices,         // Output: neighbor indices [n_cells * n_batches * k]
    scl_real_t* out_distances         // Output: neighbor distances [n_cells * n_batches * k]
);

// Compute BBKNN with precomputed norms (faster for repeated calls)
scl_error_t scl_bbknn_with_norms(
    scl_sparse_t matrix,
    const int32_t* batch_labels,
    scl_size_t n_cells,
    scl_size_t n_batches,
    scl_size_t k,
    const scl_real_t* norms_sq,       // Precomputed squared norms [n_cells]
    scl_index_t* out_indices,
    scl_real_t* out_distances
);

#ifdef __cplusplus
}
#endif
