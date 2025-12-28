#pragma once

// =============================================================================
// FILE: scl/binding/c_api/impute/impute.h
// BRIEF: C API for imputation
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// KNN imputation
scl_error_t scl_impute_knn(
    scl_sparse_t matrix,
    const scl_index_t* knn_indices,
    const scl_real_t* knn_distances,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t k_neighbors,
    scl_real_t* X_imputed,
    scl_real_t bandwidth,
    scl_real_t threshold
);

// Weighted KNN imputation
scl_error_t scl_impute_knn_weighted(
    scl_sparse_t matrix,
    const scl_index_t* knn_indices,
    const scl_real_t* knn_distances,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t k_neighbors,
    scl_real_t* X_imputed,
    scl_real_t bandwidth,
    scl_real_t threshold
);

// Diffusion imputation
scl_error_t scl_impute_diffusion(
    scl_sparse_t matrix,
    scl_sparse_t transition_matrix,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_steps,
    scl_real_t* X_imputed
);

// MAGIC imputation
scl_error_t scl_impute_magic(
    scl_sparse_t matrix,
    scl_sparse_t affinity_matrix,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t t,
    scl_real_t* X_imputed
);

#ifdef __cplusplus
}
#endif
