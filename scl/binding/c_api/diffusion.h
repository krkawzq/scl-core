#pragma once

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// FILE: scl/binding/c_api/diffusion.h
// BRIEF: C API for diffusion operations
// =============================================================================

// Compute transition matrix (row-stochastic)
scl_error_t scl_diffusion_compute_transition_matrix(
    scl_sparse_matrix_t adjacency,
    scl_real_t* transition_values,
    int symmetric
);

// Diffusion on dense vector
scl_error_t scl_diffusion_diffuse_vector(
    scl_sparse_matrix_t transition,
    scl_real_t* x,
    scl_size_t n_nodes,
    scl_index_t n_steps
);

// Diffusion on dense matrix
scl_error_t scl_diffusion_diffuse_matrix(
    scl_sparse_matrix_t transition,
    scl_real_t* X,
    scl_index_t n_nodes,
    scl_index_t n_features,
    scl_index_t n_steps
);

// Diffusion pseudotime (DPT)
scl_error_t scl_diffusion_compute_dpt(
    scl_sparse_matrix_t transition,
    scl_index_t root_cell,
    scl_real_t* pseudotime,
    scl_size_t n_nodes,
    scl_index_t max_iter,
    scl_real_t tol
);

// Multi-root DPT
scl_error_t scl_diffusion_compute_dpt_multi_root(
    scl_sparse_matrix_t transition,
    const scl_index_t* root_cells,
    scl_size_t n_roots,
    scl_real_t* pseudotime,
    scl_size_t n_nodes,
    scl_index_t max_iter
);

// Random walk with restart
scl_error_t scl_diffusion_random_walk_with_restart(
    scl_sparse_matrix_t transition,
    const scl_index_t* seed_nodes,
    scl_size_t n_seeds,
    scl_real_t* scores,
    scl_size_t n_nodes,
    scl_real_t alpha,
    scl_index_t max_iter,
    scl_real_t tol
);

// Diffusion map embedding
scl_error_t scl_diffusion_diffusion_map_embedding(
    scl_sparse_matrix_t transition,
    scl_real_t* embedding,
    scl_size_t n_nodes,
    scl_index_t n_components,
    scl_index_t n_iter
);

// Heat kernel signature
scl_error_t scl_diffusion_heat_kernel_signature(
    scl_sparse_matrix_t transition,
    scl_real_t* signature,
    scl_size_t n_nodes,
    scl_real_t t,
    scl_index_t n_steps
);

// MAGIC-style imputation
scl_error_t scl_diffusion_magic_impute(
    scl_sparse_matrix_t transition,
    scl_real_t* X,
    scl_index_t n_nodes,
    scl_index_t n_features,
    scl_index_t t
);

// Diffusion distance
scl_error_t scl_diffusion_diffusion_distance(
    scl_sparse_matrix_t transition,
    scl_real_t* distances,
    scl_size_t n_nodes,
    scl_index_t n_steps
);

// Personalized PageRank
scl_error_t scl_diffusion_personalized_pagerank(
    scl_sparse_matrix_t transition,
    scl_index_t seed_node,
    scl_real_t* scores,
    scl_size_t n_nodes,
    scl_real_t alpha,
    scl_index_t max_iter,
    scl_real_t tol
);

// Lazy random walk
scl_error_t scl_diffusion_lazy_random_walk(
    scl_sparse_matrix_t transition,
    scl_real_t* x,
    scl_size_t n_nodes,
    scl_index_t n_steps,
    scl_real_t laziness
);

#ifdef __cplusplus
}
#endif
