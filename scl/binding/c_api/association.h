#pragma once

// =============================================================================
// FILE: scl/binding/c_api/association.h
// BRIEF: C API for feature association analysis across modalities
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "core.h"

// =============================================================================
// Gene-Peak Correlation
// =============================================================================

scl_error_t scl_gene_peak_correlation_f32_csr(
    scl_sparse_matrix_t rna_expression,
    scl_sparse_matrix_t atac_accessibility,
    scl_index_t* gene_indices,       // Output [max_correlations]
    scl_index_t* peak_indices,       // Output [max_correlations]
    scl_real_t* correlations,        // Output [max_correlations]
    scl_size_t max_correlations,
    scl_size_t* n_correlations,       // Output: actual number
    scl_real_t min_correlation
);

scl_error_t scl_gene_peak_correlation_f64_csr(
    scl_sparse_matrix_t rna_expression,
    scl_sparse_matrix_t atac_accessibility,
    scl_index_t* gene_indices,
    scl_index_t* peak_indices,
    double* correlations,
    scl_size_t max_correlations,
    scl_size_t* n_correlations,
    double min_correlation
);

// =============================================================================
// Enhancer-Gene Links
// =============================================================================

scl_error_t scl_enhancer_gene_link_f32_csr(
    scl_sparse_matrix_t rna,
    scl_sparse_matrix_t atac,
    scl_real_t correlation_threshold,
    scl_index_t* link_genes,          // Output [max_links]
    scl_index_t* link_peaks,         // Output [max_links]
    scl_real_t* link_correlations,   // Output [max_links]
    scl_size_t max_links,
    scl_size_t* n_links              // Output: actual number
);

scl_error_t scl_enhancer_gene_link_f64_csr(
    scl_sparse_matrix_t rna,
    scl_sparse_matrix_t atac,
    double correlation_threshold,
    scl_index_t* link_genes,
    scl_index_t* link_peaks,
    double* link_correlations,
    scl_size_t max_links,
    scl_size_t* n_links
);

// =============================================================================
// Multi-modal Neighbors
// =============================================================================

scl_error_t scl_multimodal_neighbors_f32_csr(
    scl_sparse_matrix_t modality1,
    scl_sparse_matrix_t modality2,
    scl_real_t weight1,
    scl_real_t weight2,
    scl_index_t k,
    scl_index_t* neighbor_indices,   // Output [n_cells * k]
    scl_real_t* neighbor_distances   // Output [n_cells * k]
);

scl_error_t scl_multimodal_neighbors_f64_csr(
    scl_sparse_matrix_t modality1,
    scl_sparse_matrix_t modality2,
    double weight1,
    double weight2,
    scl_index_t k,
    scl_index_t* neighbor_indices,
    double* neighbor_distances
);

#ifdef __cplusplus
}
#endif
