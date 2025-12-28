#pragma once

// =============================================================================
// FILE: scl/binding/c_api/leiden/leiden.h
// BRIEF: C API for Leiden clustering
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Leiden Clustering
// =============================================================================

scl_error_t scl_leiden_cluster(
    scl_sparse_t adjacency,
    scl_index_t* partition,
    scl_size_t n_nodes,
    scl_real_t resolution,
    scl_index_t max_iter,
    scl_index_t* out_n_communities
);

scl_error_t scl_leiden_cluster_multilevel(
    scl_sparse_t adjacency,
    scl_index_t* partition,
    scl_size_t n_nodes,
    scl_real_t resolution,
    scl_index_t max_iter,
    scl_index_t* out_n_communities
);

scl_error_t scl_leiden_compute_modularity(
    scl_sparse_t adjacency,
    const scl_index_t* partition,
    scl_size_t n_nodes,
    scl_real_t resolution,
    scl_real_t* out_modularity
);

#ifdef __cplusplus
}
#endif
