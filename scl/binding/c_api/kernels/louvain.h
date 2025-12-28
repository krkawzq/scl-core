#pragma once

// =============================================================================
// FILE: scl/binding/c_api/kernels/louvain.h
// BRIEF: C API for Louvain clustering
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "../core.h"

// =============================================================================
// Louvain Clustering
// =============================================================================

scl_error_t scl_louvain_cluster(
    scl_sparse_matrix_t adjacency,  // CSR sparse adjacency matrix
    scl_index_t* labels,             // Output: cluster labels [n_nodes]
    scl_index_t n_nodes,             // Number of nodes
    scl_real_t resolution,           // Resolution parameter (default: 1.0)
    scl_index_t max_iter             // Maximum iterations (default: 100)
);

scl_error_t scl_louvain_compute_modularity(
    scl_sparse_matrix_t adjacency,   // CSR sparse adjacency matrix
    const scl_index_t* labels,       // Cluster labels [n_nodes]
    scl_index_t n_nodes,             // Number of nodes
    scl_real_t resolution,           // Resolution parameter
    scl_real_t* modularity           // Output: modularity score
);

scl_error_t scl_louvain_community_sizes(
    const scl_index_t* labels,       // Cluster labels [n_nodes]
    scl_index_t n_nodes,             // Number of nodes
    scl_index_t* sizes,              // Output: sizes per community [n_communities]
    scl_index_t* n_communities       // Output: number of communities
);

scl_error_t scl_louvain_get_community_members(
    const scl_index_t* labels,       // Cluster labels [n_nodes]
    scl_index_t n_nodes,             // Number of nodes
    scl_index_t community,           // Target community ID
    scl_index_t* members,            // Output: member node indices [n_members]
    scl_index_t* n_members           // Output: number of members
);

#ifdef __cplusplus
}
#endif
