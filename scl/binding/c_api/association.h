#pragma once

// =============================================================================
// FILE: scl/binding/c_api/association.h
// BRIEF: C API for feature association analysis across modalities
// =============================================================================
//
// APPLICATIONS:
//   - Gene-peak correlation (RNA + ATAC)
//   - Cis-regulatory element identification
//   - Enhancer-gene linking
//   - Multi-modal feature coupling
//   - Peak-to-gene activity scores
//
// USE CASES:
//   - Multi-omic integration (scRNA + scATAC)
//   - Regulatory network inference
//   - Feature selection across modalities
//   - Activity-based gene scores from accessibility
//
// THREAD SAFETY:
//   - All operations are thread-safe
//   - Automatic parallelization for large feature sets
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Gene-Peak Correlation
// =============================================================================

/// @brief Compute correlations between all gene-peak pairs
/// @param[in] rna_expression RNA expression matrix (non-null)
/// @param[in] atac_accessibility ATAC accessibility matrix (non-null)
/// @param[out] gene_indices Gene indices for significant pairs (non-null)
/// @param[out] peak_indices Peak indices for significant pairs (non-null)
/// @param[out] correlations Correlation values (non-null)
/// @param[out] n_correlations Number of significant correlations (non-null)
/// @param[in] min_correlation Minimum absolute correlation threshold
/// @return SCL_OK on success, error code otherwise
/// @note Caller must pre-allocate arrays with sufficient size
scl_error_t scl_association_gene_peak_correlation(
    scl_sparse_t rna_expression,
    scl_sparse_t atac_accessibility,
    scl_index_t* gene_indices,
    scl_index_t* peak_indices,
    scl_real_t* correlations,
    scl_size_t* n_correlations,
    scl_real_t min_correlation
);

// =============================================================================
// Cis-Regulatory Associations
// =============================================================================

/// @brief Compute cis-regulatory associations for specific gene-peak pairs
/// @param[in] rna_expression RNA expression matrix (non-null)
/// @param[in] atac_accessibility ATAC accessibility matrix (non-null)
/// @param[in] gene_indices Gene indices to test [n_pairs] (non-null)
/// @param[in] peak_indices Peak indices to test [n_pairs] (non-null)
/// @param[in] n_pairs Number of pairs to test
/// @param[out] correlations Pearson correlations [n_pairs] (non-null)
/// @param[out] p_values Statistical p-values [n_pairs] (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_association_cis_regulatory(
    scl_sparse_t rna_expression,
    scl_sparse_t atac_accessibility,
    const scl_index_t* gene_indices,
    const scl_index_t* peak_indices,
    scl_size_t n_pairs,
    scl_real_t* correlations,
    scl_real_t* p_values
);

// =============================================================================
// Enhancer-Gene Links
// =============================================================================

/// @brief Identify enhancer-gene links via correlation
/// @param[in] rna RNA expression matrix (non-null)
/// @param[in] atac ATAC accessibility matrix (non-null)
/// @param[in] correlation_threshold Minimum correlation for link
/// @param[out] link_genes Gene indices [n_links] (non-null)
/// @param[out] link_peaks Peak indices [n_links] (non-null)
/// @param[out] link_correlations Correlation values [n_links] (non-null)
/// @param[out] n_links Number of links found (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note Only positive correlations are returned
scl_error_t scl_association_enhancer_gene_link(
    scl_sparse_t rna,
    scl_sparse_t atac,
    scl_real_t correlation_threshold,
    scl_index_t* link_genes,
    scl_index_t* link_peaks,
    scl_real_t* link_correlations,
    scl_size_t* n_links
);

// =============================================================================
// Multi-modal Neighbors
// =============================================================================

/// @brief Compute nearest neighbors using combined distance from two modalities
/// @param[in] modality1 First modality data (non-null)
/// @param[in] modality2 Second modality data (non-null)
/// @param[in] weight1 Weight for modality 1 distance
/// @param[in] weight2 Weight for modality 2 distance
/// @param[in] k Number of neighbors
/// @param[out] neighbor_indices Neighbor indices [n_cells * k] (non-null)
/// @param[out] neighbor_distances Neighbor distances [n_cells * k] (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_association_multimodal_neighbors(
    scl_sparse_t modality1,
    scl_sparse_t modality2,
    scl_real_t weight1,
    scl_real_t weight2,
    scl_index_t k,
    scl_index_t* neighbor_indices,
    scl_real_t* neighbor_distances
);

// =============================================================================
// Feature Coupling
// =============================================================================

/// @brief Find coupled features across modalities (Spearman correlation)
/// @param[in] modality1 First modality data (non-null)
/// @param[in] modality2 Second modality data (non-null)
/// @param[out] feature1_indices Feature indices from modality1 (non-null)
/// @param[out] feature2_indices Feature indices from modality2 (non-null)
/// @param[out] coupling_scores Coupling scores (non-null)
/// @param[out] n_couplings Number of significant couplings (non-null)
/// @param[in] min_score Minimum coupling score threshold
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_association_feature_coupling(
    scl_sparse_t modality1,
    scl_sparse_t modality2,
    scl_index_t* feature1_indices,
    scl_index_t* feature2_indices,
    scl_real_t* coupling_scores,
    scl_size_t* n_couplings,
    scl_real_t min_score
);

// =============================================================================
// Peak-to-Gene Activity
// =============================================================================

/// @brief Compute gene activity scores from peak accessibility
/// @param[in] atac ATAC accessibility matrix (non-null)
/// @param[in] peak_to_gene_map Peak to gene mapping [n_peaks] (non-null)
/// @param[in] n_peaks Number of peaks
/// @param[in] n_genes Number of genes
/// @param[out] gene_activity Gene activity matrix [n_cells * n_genes] (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note peak_to_gene_map[i] = gene index for peak i, or -1 if no mapping
scl_error_t scl_association_peak_to_gene_activity(
    scl_sparse_t atac,
    const scl_index_t* peak_to_gene_map,
    scl_size_t n_peaks,
    scl_size_t n_genes,
    scl_real_t* gene_activity
);

/// @brief Compute correlation for specific feature pair in cell subset
/// @param[in] data1 First modality data (non-null)
/// @param[in] feature1 Feature index in data1
/// @param[in] data2 Second modality data (non-null)
/// @param[in] feature2 Feature index in data2
/// @param[in] cell_indices Cell subset indices [n_subset] (non-null)
/// @param[in] n_subset Number of cells in subset
/// @param[out] correlation Output correlation value (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_association_correlation_in_subset(
    scl_sparse_t data1,
    scl_index_t feature1,
    scl_sparse_t data2,
    scl_index_t feature2,
    const scl_index_t* cell_indices,
    scl_size_t n_subset,
    scl_real_t* correlation
);

#ifdef __cplusplus
}
#endif
