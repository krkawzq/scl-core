#pragma once

// =============================================================================
// FILE: scl/binding/c_api/louvain.h
// BRIEF: C API for Louvain clustering algorithm
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Louvain Clustering
// =============================================================================

scl_error_t scl_louvain_clustering(
    scl_sparse_t adjacency,            // Sparse adjacency matrix
    scl_index_t* labels,               // Output [n_nodes] - community assignments
    scl_size_t n_nodes,
    scl_real_t resolution,             // Resolution parameter (default 1.0)
    scl_index_t max_iter               // Maximum iterations (default 100)
);

// =============================================================================
// Compute Modularity Score
// =============================================================================

scl_error_t scl_louvain_compute_modularity(
    scl_sparse_t adjacency,
    const scl_index_t* labels,
    scl_size_t n_nodes,
    scl_real_t resolution,
    scl_real_t* modularity             // Output
);

// =============================================================================
// Get Community Sizes
// =============================================================================

scl_error_t scl_louvain_community_sizes(
    const scl_index_t* labels,
    scl_size_t n_nodes,
    scl_index_t* sizes,                // Output [n_communities]
    scl_size_t sizes_size,
    scl_index_t* n_communities         // Output: number of communities
);

// =============================================================================
// Get Nodes in Specific Community
// =============================================================================

scl_error_t scl_louvain_get_community_members(
    const scl_index_t* labels,
    scl_size_t n_nodes,
    scl_index_t community,
    scl_index_t* members,              // Output [n_members]
    scl_size_t members_size,
    scl_index_t* n_members             // Output: number of members
);

#ifdef __cplusplus
}
#endif
