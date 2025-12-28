#pragma once

// =============================================================================
// FILE: scl/binding/c_api/leiden.h
// BRIEF: C API for Leiden clustering
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "scl/binding/c_api/core_types.h"

// Leiden clustering
scl_error_t scl_leiden_cluster(
    const scl_sparse_matrix_t* adjacency,
    scl_index_t* labels,
    scl_real_t resolution,
    scl_index_t max_iter,
    uint64_t seed
);

// Multi-level Leiden clustering
scl_error_t scl_leiden_cluster_multilevel(
    const scl_sparse_matrix_t* adjacency,
    scl_index_t* labels,
    scl_real_t resolution,
    scl_index_t max_levels,
    uint64_t seed
);

// Compute modularity score
scl_error_t scl_leiden_compute_modularity(
    const scl_sparse_matrix_t* adjacency,
    const scl_index_t* labels,
    scl_real_t* modularity,
    scl_real_t resolution
);

// Get community sizes
scl_error_t scl_leiden_community_sizes(
    const scl_index_t* labels,
    scl_index_t* sizes,
    scl_size_t n_nodes,
    scl_index_t* n_communities
);

// Sort communities by size (descending)
scl_error_t scl_leiden_sort_communities_by_size(
    scl_index_t* labels,
    scl_size_t n_nodes,
    scl_index_t n_communities
);

#ifdef __cplusplus
}
#endif
