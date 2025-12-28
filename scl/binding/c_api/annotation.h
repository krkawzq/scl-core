#pragma once

// =============================================================================
// FILE: scl/binding/c_api/annotation.h
// BRIEF: C API for cell type annotation from reference
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Reference Mapping (KNN-based Transfer)
// =============================================================================

scl_error_t scl_annotation_reference_mapping(
    scl_sparse_t query_expression,
    scl_sparse_t reference_expression,
    const scl_index_t* reference_labels,
    scl_size_t n_ref,
    scl_sparse_t query_to_ref_neighbors,
    scl_index_t n_query,
    scl_index_t n_types,
    scl_index_t* query_labels,         // Output [n_query]
    scl_real_t* confidence_scores      // Output [n_query]
);

// =============================================================================
// Correlation-Based Assignment (SingleR-style)
// =============================================================================

scl_error_t scl_annotation_correlation_assignment(
    scl_sparse_t query_expression,
    scl_sparse_t reference_profiles,    // n_types x n_genes
    scl_index_t n_query,
    scl_index_t n_types,
    scl_index_t n_genes,
    scl_index_t* assigned_labels,      // Output [n_query]
    scl_real_t* correlation_scores,   // Output [n_query]
    scl_real_t* all_correlations       // Optional: [n_query * n_types], can be NULL
);

// =============================================================================
// Build Reference Profiles (Mean per Cell Type)
// =============================================================================

scl_error_t scl_annotation_build_reference_profiles(
    scl_sparse_t expression,
    const scl_index_t* labels,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_types,
    scl_real_t* profiles               // Output [n_types * n_genes]
);

// =============================================================================
// Marker Gene Score (scType-style)
// =============================================================================

scl_error_t scl_annotation_marker_gene_score(
    scl_sparse_t expression,
    const scl_index_t* const* marker_genes,  // Array of marker gene arrays per type
    const scl_index_t* marker_counts,        // Number of markers per type
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_types,
    scl_real_t* scores,                      // Output [n_cells * n_types]
    int normalize                            // 1 = normalize, 0 = don't
);

// =============================================================================
// Assign from Marker Scores
// =============================================================================

scl_error_t scl_annotation_assign_from_marker_scores(
    const scl_real_t* scores,          // [n_cells * n_types]
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_index_t* labels,               // Output [n_cells]
    scl_real_t* confidence             // Output [n_cells]
);

// =============================================================================
// Consensus Annotation (Multiple Methods)
// =============================================================================

scl_error_t scl_annotation_consensus_annotation(
    const scl_index_t* const* predictions,   // Array of prediction arrays
    const scl_real_t* const* confidences,     // Array of confidence arrays (optional)
    scl_index_t n_methods,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_index_t* consensus_labels,            // Output [n_cells]
    scl_real_t* consensus_confidence          // Output [n_cells]
);

// =============================================================================
// Detect Novel Cell Types
// =============================================================================

scl_error_t scl_annotation_detect_novel_types(
    scl_sparse_t query_expression,
    const scl_real_t* confidence_scores,
    scl_index_t n_query,
    scl_real_t threshold,
    int* is_novel                        // Output [n_query], 1 = novel, 0 = not
);

// =============================================================================
// Label Propagation from Neighbors
// =============================================================================

scl_error_t scl_annotation_label_propagation(
    scl_sparse_t neighbor_graph,
    const scl_index_t* initial_labels,   // -1 for unlabeled
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_index_t max_iter,
    scl_index_t* final_labels,           // Output [n_cells]
    scl_real_t* label_confidence         // Output [n_cells]
);

// =============================================================================
// Annotation Quality Metrics
// =============================================================================

scl_error_t scl_annotation_quality_metrics(
    const scl_index_t* predicted_labels,
    const scl_index_t* true_labels,      // Ground truth
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* accuracy,                // Output
    scl_real_t* macro_f1,                // Output
    scl_real_t* per_class_f1             // Optional: [n_types], can be NULL
);

#ifdef __cplusplus
}
#endif
