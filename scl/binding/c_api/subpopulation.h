#pragma once

// =============================================================================
// FILE: scl/binding/c_api/subpopulation.h
// BRIEF: C API for subpopulation analysis and cluster refinement
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Recursive Sub-clustering
// =============================================================================

scl_error_t scl_subpopulation_subclustering(
    scl_sparse_t expression,
    const scl_index_t* parent_labels,
    scl_size_t n_cells,
    scl_index_t parent_cluster,
    scl_size_t n_subclusters,
    scl_index_t* subcluster_labels,    // Output [n_cells]
    uint64_t seed
);

// =============================================================================
// Cluster Stability via Bootstrap
// =============================================================================

scl_error_t scl_subpopulation_cluster_stability(
    scl_sparse_t expression,
    const scl_index_t* original_labels,
    scl_size_t n_cells,
    scl_size_t n_bootstraps,
    scl_real_t* stability_scores,       // Output [n_clusters]
    scl_size_t stability_scores_size,
    uint64_t seed
);

// =============================================================================
// Cluster Purity
// =============================================================================

scl_error_t scl_subpopulation_cluster_purity(
    const scl_index_t* cluster_labels,
    const scl_index_t* true_labels,
    scl_size_t n_cells,
    scl_real_t* purity_per_cluster,    // Output [n_clusters]
    scl_size_t purity_size
);

// =============================================================================
// Rare Cell Detection
// =============================================================================

scl_error_t scl_subpopulation_rare_cell_detection(
    scl_sparse_t expression,
    scl_sparse_t neighbors,
    scl_real_t* rarity_scores          // Output [n_cells]
);

#ifdef __cplusplus
}
#endif
