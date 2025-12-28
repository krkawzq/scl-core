#pragma once

// =============================================================================
// FILE: scl/binding/c_api/annotation.h
// BRIEF: C API for cell type annotation from reference
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "core.h"

// =============================================================================
// Reference Mapping (KNN-based Transfer)
// =============================================================================

scl_error_t scl_reference_mapping_f32_csr(
    scl_sparse_matrix_t query_expression,
    scl_sparse_matrix_t reference_expression,
    const scl_index_t* reference_labels,
    scl_size_t n_ref,
    scl_sparse_matrix_t query_to_ref_neighbors,
    scl_index_t n_query,
    scl_index_t n_types,
    scl_index_t* query_labels,        // Output [n_query]
    scl_real_t* confidence_scores     // Output [n_query]
);

scl_error_t scl_reference_mapping_f64_csr(
    scl_sparse_matrix_t query_expression,
    scl_sparse_matrix_t reference_expression,
    const scl_index_t* reference_labels,
    scl_size_t n_ref,
    scl_sparse_matrix_t query_to_ref_neighbors,
    scl_index_t n_query,
    scl_index_t n_types,
    scl_index_t* query_labels,
    double* confidence_scores
);

// =============================================================================
// Correlation-Based Assignment (SingleR-style)
// =============================================================================

scl_error_t scl_correlation_assignment_f32_csr(
    scl_sparse_matrix_t query_expression,
    scl_sparse_matrix_t reference_profiles,  // n_types x n_genes
    scl_index_t n_query,
    scl_index_t n_types,
    scl_index_t n_genes,
    scl_index_t* assigned_labels,     // Output [n_query]
    scl_real_t* correlation_scores,   // Output [n_query]
    scl_real_t* all_correlations      // Optional: [n_query * n_types], can be NULL
);

scl_error_t scl_correlation_assignment_f64_csr(
    scl_sparse_matrix_t query_expression,
    scl_sparse_matrix_t reference_profiles,
    scl_index_t n_query,
    scl_index_t n_types,
    scl_index_t n_genes,
    scl_index_t* assigned_labels,
    double* correlation_scores,
    double* all_correlations
);

// =============================================================================
// Build Reference Profiles (Mean per Cell Type)
// =============================================================================

scl_error_t scl_build_reference_profiles_f32_csr(
    scl_sparse_matrix_t expression,
    const scl_index_t* labels,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_types,
    scl_real_t* profiles             // Output [n_types * n_genes]
);

scl_error_t scl_build_reference_profiles_f64_csr(
    scl_sparse_matrix_t expression,
    const scl_index_t* labels,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_types,
    double* profiles
);

// =============================================================================
// Marker Gene Score (scType-style)
// =============================================================================

scl_error_t scl_marker_gene_score_f32_csr(
    scl_sparse_matrix_t expression,
    const scl_index_t* const* marker_genes,  // Array of marker gene arrays per type
    const scl_index_t* marker_counts,         // Number of markers per type
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_types,
    scl_real_t* scores,                      // Output [n_cells * n_types]
    int normalize                             // 1 to normalize, 0 otherwise
);

scl_error_t scl_marker_gene_score_f64_csr(
    scl_sparse_matrix_t expression,
    const scl_index_t* const* marker_genes,
    const scl_index_t* marker_counts,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_types,
    double* scores,
    int normalize
);

#ifdef __cplusplus
}
#endif
