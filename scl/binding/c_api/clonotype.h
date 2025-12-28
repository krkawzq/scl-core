#pragma once

// =============================================================================
// FILE: scl/binding/c_api/clonotype.h
// BRIEF: C API for TCR/BCR clonal analysis
// =============================================================================

#include "scl/binding/c_api/types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Clone Size Distribution
// =============================================================================

scl_error_t scl_clone_size_distribution(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_size_t* clone_sizes,
    scl_size_t* n_clones,
    scl_size_t max_clones
);

// =============================================================================
// Clonal Diversity Indices
// =============================================================================

scl_error_t scl_clonal_diversity(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_real_t* shannon_diversity,
    scl_real_t* simpson_diversity,
    scl_real_t* gini_index
);

// =============================================================================
// Clone Dynamics
// =============================================================================

scl_error_t scl_clone_dynamics(
    const scl_index_t* clone_ids_t1,
    scl_size_t n_cells_t1,
    const scl_index_t* clone_ids_t2,
    scl_size_t n_cells_t2,
    scl_real_t* expansion_rates,
    scl_size_t* n_clones,
    scl_size_t max_clones
);

// =============================================================================
// Shared Clonotypes
// =============================================================================

scl_error_t scl_shared_clonotypes(
    const scl_index_t* clone_ids_sample1,
    scl_size_t n_cells_sample1,
    const scl_index_t* clone_ids_sample2,
    scl_size_t n_cells_sample2,
    scl_index_t* shared_clones,
    scl_size_t* n_shared,
    scl_real_t* jaccard_index,
    scl_size_t max_shared
);

// =============================================================================
// Clone Phenotype
// =============================================================================

scl_error_t scl_clone_phenotype(
    scl_sparse_matrix_t expression,
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_real_t* clone_profiles,
    scl_size_t* n_clones,
    scl_size_t max_clones
);

// =============================================================================
// Clonality Score
// =============================================================================

scl_error_t scl_clonality_score(
    const scl_index_t* clone_ids,
    const scl_index_t* cluster_labels,
    scl_size_t n_cells,
    scl_real_t* clonality_per_cluster,
    scl_size_t n_clusters
);

// =============================================================================
// Repertoire Overlap
// =============================================================================

scl_error_t scl_repertoire_overlap_morisita(
    const scl_index_t* clone_ids_1,
    scl_size_t n_cells_1,
    const scl_index_t* clone_ids_2,
    scl_size_t n_cells_2,
    scl_real_t* overlap_index
);

// =============================================================================
// Diversity Per Cluster
// =============================================================================

scl_error_t scl_diversity_per_cluster(
    const scl_index_t* clone_ids,
    const scl_index_t* cluster_labels,
    scl_size_t n_cells,
    scl_real_t* shannon_per_cluster,
    scl_real_t* simpson_per_cluster,
    scl_size_t n_clusters
);

// =============================================================================
// Clone Transition Matrix
// =============================================================================

scl_error_t scl_clone_transition_matrix(
    const scl_index_t* clone_ids,
    const scl_index_t* cluster_labels,
    scl_size_t n_cells,
    scl_real_t* transition_matrix,
    scl_size_t n_clusters
);

// =============================================================================
// Rarefaction Diversity
// =============================================================================

scl_error_t scl_rarefaction_diversity(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_size_t subsample_size,
    scl_size_t n_iterations,
    scl_real_t* mean_diversity,
    scl_real_t* std_diversity,
    uint64_t seed
);

// =============================================================================
// Clone Expansion Detection
// =============================================================================

scl_error_t scl_detect_expanded_clones(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_size_t expansion_threshold,
    scl_index_t* expanded_clones,
    scl_size_t* n_expanded,
    scl_size_t max_expanded
);

// =============================================================================
// Clone Size Statistics
// =============================================================================

scl_error_t scl_clone_size_statistics(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_real_t* mean_size,
    scl_real_t* median_size,
    scl_real_t* max_size,
    scl_size_t* n_singletons,
    scl_size_t* n_clones
);

#ifdef __cplusplus
}
#endif
