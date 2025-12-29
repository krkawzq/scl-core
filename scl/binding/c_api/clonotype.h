#pragma once

// =============================================================================
// FILE: scl/binding/c_api/clonotype.h
// BRIEF: C API for TCR/BCR clonal analysis
// =============================================================================
//
// PURPOSE:
//   - Analyze T-cell receptor (TCR) and B-cell receptor (BCR) repertoires
//   - Quantify clonal expansion and diversity
//   - Track clone dynamics over time
//   - Assess repertoire overlap between samples
//
// CLONOTYPE REPRESENTATION:
//   - Each cell has a clone ID (integer >= 0)
//   - Clone ID = -1 means no clonotype assigned
//   - Clones with same ID share identical TCR/BCR sequence
//
// DIVERSITY METRICS:
//   - Shannon entropy: H = -Σ p_i * log(p_i)
//   - Simpson diversity: D = 1 - Σ p_i^2
//   - Gini coefficient: Inequality in clone sizes
//
// APPLICATIONS:
//   - Clonal expansion in immune responses
//   - Repertoire diversity in aging/disease
//   - Clone tracking in longitudinal studies
//   - Phenotype association with clonotypes
//
// THREAD SAFETY:
//   - All operations are thread-safe
//   - Read-only operations can be called concurrently
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Clone Size Distribution
// =============================================================================

/// @brief Compute size distribution of clones
/// @param[in] clone_ids Clone IDs [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[out] clone_sizes Size of each clone [max_clones] (non-null)
/// @param[out] n_clones Number of non-empty clones (non-null)
/// @param[in] max_clones Maximum possible clone ID + 1
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_clonotype_size_distribution(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_size_t* clone_sizes,
    scl_size_t* n_clones,
    scl_size_t max_clones
);

// =============================================================================
// Clonal Diversity
// =============================================================================

/// @brief Compute clonal diversity indices
/// @param[in] clone_ids Clone IDs [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[out] shannon_diversity Shannon entropy (non-null)
/// @param[out] simpson_diversity Simpson diversity index (non-null)
/// @param[out] gini_index Gini coefficient (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_clonotype_diversity(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_real_t* shannon_diversity,
    scl_real_t* simpson_diversity,
    scl_real_t* gini_index
);

// =============================================================================
// Clone Dynamics (Temporal)
// =============================================================================

/// @brief Compute clone expansion rates between timepoints
/// @param[in] clone_ids_t1 Clone IDs at timepoint 1 [n_cells_t1] (non-null)
/// @param[in] n_cells_t1 Number of cells at timepoint 1
/// @param[in] clone_ids_t2 Clone IDs at timepoint 2 [n_cells_t2] (non-null)
/// @param[in] n_cells_t2 Number of cells at timepoint 2
/// @param[out] expansion_rates Log2 fold change [n_clones] (non-null)
/// @param[out] n_clones Number of clones tracked (non-null)
/// @param[in] max_clones Maximum clones to track
/// @return SCL_OK on success, error code otherwise
/// @note expansion_rate = log2((size_t2 + 1) / (size_t1 + 1))
scl_error_t scl_clonotype_dynamics(
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

/// @brief Find shared clonotypes between two samples
/// @param[in] clone_ids_sample1 Clone IDs from sample 1 [n_cells_1] (non-null)
/// @param[in] n_cells_1 Number of cells in sample 1
/// @param[in] clone_ids_sample2 Clone IDs from sample 2 [n_cells_2] (non-null)
/// @param[in] n_cells_2 Number of cells in sample 2
/// @param[out] shared_clones Shared clone IDs [max_shared] (non-null)
/// @param[out] n_shared Number of shared clones (non-null)
/// @param[out] jaccard_index Jaccard similarity [0, 1] (non-null)
/// @param[in] max_shared Maximum shared clones to return
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_clonotype_shared(
    const scl_index_t* clone_ids_sample1,
    scl_size_t n_cells_1,
    const scl_index_t* clone_ids_sample2,
    scl_size_t n_cells_2,
    scl_index_t* shared_clones,
    scl_size_t* n_shared,
    scl_real_t* jaccard_index,
    scl_size_t max_shared
);

// =============================================================================
// Clone Phenotype (Expression Profile)
// =============================================================================

/// @brief Compute mean expression profile per clone
/// @param[in] expression Expression matrix (cells x genes) (non-null)
/// @param[in] clone_ids Clone IDs [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] n_genes Number of genes
/// @param[out] clone_profiles Mean expression [n_clones * n_genes] (non-null)
/// @param[out] n_clones Number of clones found (non-null)
/// @param[in] max_clones Maximum clones to process
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_clonotype_phenotype(
    scl_sparse_t expression,
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_size_t n_genes,
    scl_real_t* clone_profiles,
    scl_size_t* n_clones,
    scl_size_t max_clones
);

// =============================================================================
// Clonality Score (Per Cluster)
// =============================================================================

/// @brief Compute clonality score for each cluster
/// @param[in] clone_ids Clone IDs [n_cells] (non-null)
/// @param[in] cluster_labels Cluster assignments [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[out] clonality_per_cluster Clonality scores [n_clusters] (non-null)
/// @param[in] n_clusters Number of clusters
/// @return SCL_OK on success, error code otherwise
/// @note Clonality = 1 - normalized_entropy (higher = more clonal)
scl_error_t scl_clonotype_clonality_score(
    const scl_index_t* clone_ids,
    const scl_index_t* cluster_labels,
    scl_size_t n_cells,
    scl_real_t* clonality_per_cluster,
    scl_size_t n_clusters
);

// =============================================================================
// Repertoire Overlap
// =============================================================================

/// @brief Compute Morisita-Horn overlap index between repertoires
/// @param[in] clone_ids_1 Clone IDs from repertoire 1 [n_cells_1] (non-null)
/// @param[in] n_cells_1 Number of cells in repertoire 1
/// @param[in] clone_ids_2 Clone IDs from repertoire 2 [n_cells_2] (non-null)
/// @param[in] n_cells_2 Number of cells in repertoire 2
/// @param[out] overlap_index Morisita-Horn index [0, 1] (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note Index = 1 means identical repertoires, 0 means no overlap
scl_error_t scl_clonotype_repertoire_overlap(
    const scl_index_t* clone_ids_1,
    scl_size_t n_cells_1,
    const scl_index_t* clone_ids_2,
    scl_size_t n_cells_2,
    scl_real_t* overlap_index
);

// =============================================================================
// Diversity Per Cluster
// =============================================================================

/// @brief Compute diversity indices per cluster
/// @param[in] clone_ids Clone IDs [n_cells] (non-null)
/// @param[in] cluster_labels Cluster assignments [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[out] shannon_per_cluster Shannon diversity [n_clusters] (non-null)
/// @param[out] simpson_per_cluster Simpson diversity [n_clusters] (non-null)
/// @param[in] n_clusters Number of clusters
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_clonotype_diversity_per_cluster(
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

/// @brief Compute clone transition probabilities between clusters
/// @param[in] clone_ids Clone IDs [n_cells] (non-null)
/// @param[in] cluster_labels Cluster assignments [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[out] transition_matrix Transition matrix [n_clusters * n_clusters] (non-null)
/// @param[in] n_clusters Number of clusters
/// @return SCL_OK on success, error code otherwise
/// @note transition_matrix[i*n_clusters + j] = P(clone in cluster i appears in cluster j)
scl_error_t scl_clonotype_transition_matrix(
    const scl_index_t* clone_ids,
    const scl_index_t* cluster_labels,
    scl_size_t n_cells,
    scl_real_t* transition_matrix,
    scl_size_t n_clusters
);

// =============================================================================
// Rarefaction Analysis
// =============================================================================

/// @brief Estimate diversity at subsampled sizes (rarefaction curve)
/// @param[in] clone_ids Clone IDs [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] subsample_size Subsample size for rarefaction
/// @param[in] n_iterations Number of bootstrap iterations
/// @param[out] mean_diversity Mean Shannon diversity (non-null)
/// @param[out] std_diversity Standard deviation (non-null)
/// @param[in] seed Random seed
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_clonotype_rarefaction(
    const scl_index_t* clone_ids,
    scl_size_t n_cells,
    scl_size_t subsample_size,
    scl_size_t n_iterations,
    scl_real_t* mean_diversity,
    scl_real_t* std_diversity,
    uint64_t seed
);

// =============================================================================
// Expanded Clone Detection
// =============================================================================

/// @brief Detect expanded clones above size threshold
/// @param[in] clone_ids Clone IDs [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] expansion_threshold Minimum size for expansion
/// @param[out] expanded_clones Expanded clone IDs [max_expanded] (non-null)
/// @param[out] n_expanded Number of expanded clones (non-null)
/// @param[in] max_expanded Maximum expanded clones to return
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_clonotype_detect_expanded(
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

/// @brief Compute clone size statistics
/// @param[in] clone_ids Clone IDs [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[out] mean_size Mean clone size (non-null)
/// @param[out] median_size Median clone size (non-null)
/// @param[out] max_size Maximum clone size (non-null)
/// @param[out] n_singletons Number of clones with size 1 (non-null)
/// @param[out] n_clones Total number of clones (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_clonotype_size_statistics(
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
