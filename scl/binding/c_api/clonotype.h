#pragma once

// =============================================================================
// FILE: scl/binding/c_api/clonotype/clonotype.h
// BRIEF: C API for TCR/BCR clonal analysis
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Clone Size Distribution
// =============================================================================

scl_error_t scl_clone_size_distribution(
    const scl_index_t* clone_ids,     // Clone IDs [n_cells]
    scl_size_t n_cells,
    scl_size_t* clone_sizes,          // Output: size of each clone [max_clones]
    scl_size_t* n_clones,             // Output: number of non-empty clones
    scl_size_t max_clones              // Maximum clone ID + 1
);

// =============================================================================
// Clonal Diversity
// =============================================================================

scl_error_t scl_clonal_diversity(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_real_t* shannon_diversity,     // Output
    scl_real_t* simpson_diversity,     // Output
    scl_real_t* gini_index             // Output
);

// =============================================================================
// Clone Dynamics
// =============================================================================

scl_error_t scl_clone_dynamics(
    const scl_index_t* clone_ids_t1,  // Timepoint 1 [n_cells_t1]
    scl_size_t n_cells_t1,
    const scl_index_t* clone_ids_t2,   // Timepoint 2 [n_cells_t2]
    scl_size_t n_cells_t2,
    scl_real_t* expansion_rates,       // Output: log2 fold change [max_clones]
    scl_size_t* n_clones,              // Output: number of clones with data
    scl_size_t max_clones
);

// =============================================================================
// Shared Clonotypes
// =============================================================================

scl_error_t scl_shared_clonotypes(
    const scl_index_t* clone_ids_sample1,  // [n_cells_1]
    scl_size_t n_cells_1,
    const scl_index_t* clone_ids_sample2, // [n_cells_2]
    scl_size_t n_cells_2,
    scl_index_t* shared_clones,            // Output: shared clone IDs [max_shared]
    scl_size_t* n_shared,                  // Output: number of shared clones
    scl_real_t* jaccard_index,             // Output: Jaccard similarity
    scl_size_t max_shared
);

// =============================================================================
// Clone Phenotype
// =============================================================================

scl_error_t scl_clone_phenotype(
    scl_sparse_t expression,           // Expression matrix (cells x genes, CSR)
    const scl_index_t* clone_ids,      // [n_cells]
    scl_size_t n_cells,
    scl_size_t n_genes,
    scl_real_t* clone_profiles,        // Output: mean expression [n_clones * n_genes]
    scl_size_t* n_clones,              // Output: number of clones
    scl_size_t max_clones
);

// =============================================================================
// Clonality Score
// =============================================================================

scl_error_t scl_clonality_score(
    const scl_index_t* clone_ids,
    const scl_index_t* cluster_labels, // Cluster assignments [n_cells]
    scl_size_t n_cells,
    scl_real_t* clonality_per_cluster  // Output: clonality score [n_clusters]
);

// =============================================================================
// Repertoire Overlap
// =============================================================================

scl_error_t scl_repertoire_overlap_morisita(
    const scl_index_t* clone_ids_1,
    scl_size_t n_cells_1,
    const scl_index_t* clone_ids_2,
    scl_size_t n_cells_2,
    scl_real_t* overlap_index          // Output: Morisita-Horn index
);

// =============================================================================
// Diversity Per Cluster
// =============================================================================

scl_error_t scl_diversity_per_cluster(
    const scl_index_t* clone_ids,
    const scl_index_t* cluster_labels,
    scl_size_t n_cells,
    scl_real_t* shannon_per_cluster,   // Output [n_clusters]
    scl_real_t* simpson_per_cluster    // Output [n_clusters]
);

// =============================================================================
// Clone Transition Matrix
// =============================================================================

scl_error_t scl_clone_transition_matrix(
    const scl_index_t* clone_ids,
    const scl_index_t* cluster_labels,
    scl_size_t n_cells,
    scl_real_t* transition_matrix,      // Output: [n_clusters * n_clusters]
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
    scl_real_t* mean_diversity,        // Output
    scl_real_t* std_diversity,         // Output
    uint64_t seed
);

// =============================================================================
// Expanded Clone Detection
// =============================================================================

scl_error_t scl_detect_expanded_clones(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_size_t expansion_threshold,     // Minimum size to be considered expanded
    scl_index_t* expanded_clones,       // Output: expanded clone IDs [max_expanded]
    scl_size_t* n_expanded,             // Output: number of expanded clones
    scl_size_t max_expanded
);

// =============================================================================
// Clone Size Statistics
// =============================================================================

scl_error_t scl_clone_size_statistics(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_real_t* mean_size,              // Output
    scl_real_t* median_size,            // Output
    scl_real_t* max_size,                // Output
    scl_size_t* n_singletons,            // Output: clones with size 1
    scl_size_t* n_clones                 // Output: total number of clones
);

#ifdef __cplusplus
}
#endif
