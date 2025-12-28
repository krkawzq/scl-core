#pragma once

// =============================================================================
// FILE: scl/binding/c_api/association.h
// BRIEF: C API for feature association analysis across modalities
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Gene-Peak Correlation
// =============================================================================

scl_error_t scl_association_gene_peak_correlation(
    scl_sparse_t rna_expression,
    scl_sparse_t atac_accessibility,
    scl_index_t* gene_indices,          // Output [n_correlations]
    scl_index_t* peak_indices,          // Output [n_correlations]
    scl_real_t* correlations,           // Output [n_correlations]
    scl_size_t* n_correlations,         // Output: number of correlations found
    scl_real_t min_correlation
);

// =============================================================================
// Cis-Regulatory Associations
// =============================================================================

scl_error_t scl_association_cis_regulatory(
    scl_sparse_t rna_expression,
    scl_sparse_t atac_accessibility,
    const scl_index_t* gene_indices,
    const scl_index_t* peak_indices,
    scl_size_t n_pairs,
    scl_real_t* correlations,           // Output [n_pairs]
    scl_real_t* p_values               // Output [n_pairs]
);

// =============================================================================
// Enhancer-Gene Links
// =============================================================================

scl_error_t scl_association_enhancer_gene_link(
    scl_sparse_t rna,
    scl_sparse_t atac,
    scl_real_t correlation_threshold,
    scl_index_t* link_genes,            // Output [n_links]
    scl_index_t* link_peaks,            // Output [n_links]
    scl_real_t* link_correlations,      // Output [n_links]
    scl_size_t* n_links                 // Output: number of links found
);

// =============================================================================
// Multi-modal Neighbors
// =============================================================================

scl_error_t scl_association_multimodal_neighbors(
    scl_sparse_t modality1,
    scl_sparse_t modality2,
    scl_real_t weight1,
    scl_real_t weight2,
    scl_index_t k,
    scl_index_t* neighbor_indices,      // Output [n_cells * k]
    scl_real_t* neighbor_distances     // Output [n_cells * k]
);

// =============================================================================
// Feature Coupling
// =============================================================================

scl_error_t scl_association_feature_coupling(
    scl_sparse_t modality1,
    scl_sparse_t modality2,
    scl_index_t* feature1_indices,      // Output [n_couplings]
    scl_index_t* feature2_indices,      // Output [n_couplings]
    scl_real_t* coupling_scores,        // Output [n_couplings]
    scl_size_t* n_couplings,            // Output: number of couplings found
    scl_real_t min_score
);

// =============================================================================
// Peak-to-Gene Activity Score
// =============================================================================

scl_error_t scl_association_peak_to_gene_activity(
    scl_sparse_t atac,
    const scl_index_t* peak_to_gene_map,  // For each peak, which gene it maps to (-1 for none)
    scl_size_t n_peaks,
    scl_size_t n_genes,
    scl_real_t* gene_activity            // Output [n_cells * n_genes]
);

#ifdef __cplusplus
}
#endif
