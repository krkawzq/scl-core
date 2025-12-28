#pragma once

// =============================================================================
// FILE: scl/binding/c_api/velocity.h
// BRIEF: C API for RNA velocity analysis
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "scl/binding/c_api/core_types.h"

// Velocity model types
typedef enum {
    SCL_VELOCITY_STEADY_STATE = 0,
    SCL_VELOCITY_DYNAMICAL = 1,
    SCL_VELOCITY_STOCHASTIC = 2
} scl_velocity_model_t;

// Fit kinetics parameters
scl_error_t scl_velocity_fit_kinetics(
    scl_sparse_matrix_t spliced,
    scl_sparse_matrix_t unspliced,
    scl_real_t* gamma,
    scl_real_t* r2,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_velocity_model_t model
);

// Compute velocity
scl_error_t scl_velocity_compute_velocity(
    scl_sparse_matrix_t spliced,
    scl_sparse_matrix_t unspliced,
    const scl_real_t* gamma,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* velocity_out
);

// Compute splice ratio
scl_error_t scl_velocity_splice_ratio(
    scl_sparse_matrix_t spliced,
    scl_sparse_matrix_t unspliced,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* ratio_out
);

// Compute velocity graph (transition probabilities)
scl_error_t scl_velocity_velocity_graph(
    const scl_real_t* velocity,
    const scl_real_t* expression,
    scl_sparse_matrix_t knn,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* transition_probs,
    scl_index_t k_neighbors
);

#ifdef __cplusplus
}
#endif
