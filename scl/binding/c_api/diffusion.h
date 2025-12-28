#pragma once

// =============================================================================
// FILE: scl/binding/c_api/diffusion/diffusion.h
// BRIEF: C API for diffusion operations
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compute transition matrix from adjacency matrix (row-stochastic)
scl_error_t scl_diffusion_compute_transition_matrix(
    scl_sparse_t adjacency,
    int symmetric
);

// Diffuse a vector through the transition matrix
scl_error_t scl_diffusion_diffuse_vector(
    scl_sparse_t transition,
    scl_real_t* x,
    scl_size_t n_nodes,
    scl_index_t n_steps
);

// Diffuse a dense matrix through the transition matrix
scl_error_t scl_diffusion_diffuse_matrix(
    scl_sparse_t transition,
    scl_real_t* X,
    scl_index_t n_nodes,
    scl_index_t n_features,
    scl_index_t n_steps
);

// Compute diffusion pseudotime from a root cell
scl_error_t scl_diffusion_compute_dpt(
    scl_sparse_t transition,
    scl_index_t root_cell,
    scl_real_t* pseudotime,
    scl_size_t n_nodes,
    scl_index_t max_iter,
    scl_real_t tol
);

// Compute DPT from multiple root cells
scl_error_t scl_diffusion_compute_dpt_multi_root(
    scl_sparse_t transition,
    const scl_index_t* root_cells,
    scl_size_t n_roots,
    scl_real_t* pseudotime,
    scl_size_t n_nodes,
    scl_index_t max_iter
);

// Random walk with restart from seed nodes
scl_error_t scl_diffusion_random_walk_with_restart(
    scl_sparse_t transition,
    const scl_index_t* seed_nodes,
    scl_size_t n_seeds,
    scl_real_t* scores,
    scl_size_t n_nodes,
    scl_real_t alpha,
    scl_index_t max_iter,
    scl_real_t tol
);

// Personalized PageRank
scl_error_t scl_diffusion_personalized_pagerank(
    scl_sparse_t transition,
    scl_index_t seed_node,
    scl_real_t* scores,
    scl_size_t n_nodes,
    scl_real_t alpha,
    scl_index_t max_iter,
    scl_real_t tol
);

// Compute diffusion map embedding
scl_error_t scl_diffusion_diffusion_map_embedding(
    scl_sparse_t transition,
    scl_real_t* embedding,
    scl_index_t n_nodes,
    scl_index_t n_components,
    scl_index_t n_iter
);

// Compute heat kernel signature
scl_error_t scl_diffusion_heat_kernel_signature(
    scl_sparse_t transition,
    scl_real_t* signature,
    scl_size_t n_nodes,
    scl_real_t t,
    scl_index_t n_steps
);

// MAGIC-style imputation
scl_error_t scl_diffusion_magic_impute(
    scl_sparse_t transition,
    scl_real_t* X,
    scl_index_t n_nodes,
    scl_index_t n_features,
    scl_index_t t
);

// Compute pairwise diffusion distances
scl_error_t scl_diffusion_diffusion_distance(
    scl_sparse_t transition,
    scl_real_t* distances,
    scl_size_t n_nodes,
    scl_index_t n_steps
);

// Lazy random walk
scl_error_t scl_diffusion_lazy_random_walk(
    scl_sparse_t transition,
    scl_real_t* x,
    scl_size_t n_nodes,
    scl_index_t n_steps,
    scl_real_t laziness
);

#ifdef __cplusplus
}
#endif
