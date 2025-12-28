#pragma once

// =============================================================================
// FILE: scl/binding/c_api/velocity/velocity.h
// BRIEF: C API for RNA velocity analysis
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Velocity Models
// =============================================================================

typedef enum {
    SCL_VELOCITY_STEADY_STATE = 0,
    SCL_VELOCITY_DYNAMICAL = 1,
    SCL_VELOCITY_STOCHASTIC = 2
} scl_velocity_model_t;

// =============================================================================
// Velocity Functions
// =============================================================================

// Fit gene kinetics (gamma and R2)
scl_error_t scl_velocity_fit_kinetics(
    scl_sparse_t spliced,
    scl_sparse_t unspliced,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* gamma,                // [n_genes] output
    scl_real_t* r2,                   // [n_genes] output
    scl_velocity_model_t model
);

// Compute velocity (dS/dt = U - gamma * S)
scl_error_t scl_velocity_compute(
    scl_sparse_t spliced,
    scl_sparse_t unspliced,
    const scl_real_t* gamma,          // [n_genes]
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* velocity_out           // [n_cells * n_genes] row-major output
);

// Compute splice ratio (U / (S + eps))
scl_error_t scl_velocity_splice_ratio(
    scl_sparse_t spliced,
    scl_sparse_t unspliced,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* ratio_out             // [n_cells * n_genes] row-major output
);

// Compute velocity graph (transition probabilities)
scl_error_t scl_velocity_graph(
    const scl_real_t* velocity,        // [n_cells * n_genes] row-major
    const scl_real_t* expression,       // [n_cells * n_genes] row-major
    scl_sparse_t knn,                  // KNN adjacency matrix
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* transition_probs,      // [n_cells * k_neighbors] row-major output
    scl_index_t k_neighbors
);

#ifdef __cplusplus
}
#endif
