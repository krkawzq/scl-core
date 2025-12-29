#pragma once

// =============================================================================
// FILE: scl/binding/c_api/annotation.h
// BRIEF: C API for cell type annotation from reference
// =============================================================================
//
// METHODS:
//   - KNN voting: Transfer labels via nearest neighbors
//   - Correlation-based: SingleR-style correlation assignment
//   - Marker genes: scType-style marker scoring
//   - Consensus: Combine multiple annotation methods
//   - Novel detection: Identify cells not matching reference
//
// WORKFLOW:
//   1. Build reference profiles from labeled data
//   2. Compute KNN or correlations between query and reference
//   3. Transfer labels via voting or max correlation
//   4. Detect novel cell types (optional)
//   5. Evaluate annotation quality (if ground truth available)
//
// THREAD SAFETY:
//   - All operations are thread-safe
//   - Automatic parallelization for large datasets
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Reference Mapping (KNN-based Transfer)
// =============================================================================

/// @brief Transfer cell type labels via KNN voting
/// @param[in] query_expression Query expression matrix (non-null)
/// @param[in] reference_expression Reference expression matrix (non-null)
/// @param[in] reference_labels Reference cell type labels [n_ref] (non-null)
/// @param[in] n_ref Number of reference cells
/// @param[in] query_to_ref_neighbors KNN graph: query -> reference (non-null)
/// @param[in] n_query Number of query cells
/// @param[in] n_types Number of cell types
/// @param[out] query_labels Assigned labels [n_query] (non-null)
/// @param[out] confidence_scores Confidence scores [n_query] (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_annotation_reference_mapping(
    scl_sparse_t query_expression,
    scl_sparse_t reference_expression,
    const scl_index_t* reference_labels,
    scl_size_t n_ref,
    scl_sparse_t query_to_ref_neighbors,
    scl_index_t n_query,
    scl_index_t n_types,
    scl_index_t* query_labels,
    scl_real_t* confidence_scores
);

// =============================================================================
// Correlation-Based Assignment (SingleR-style)
// =============================================================================

/// @brief Assign cell types via correlation with reference profiles
/// @param[in] query_expression Query expression matrix (non-null)
/// @param[in] reference_profiles Reference type profiles [n_types x n_genes] (non-null)
/// @param[in] n_query Number of query cells
/// @param[in] n_types Number of cell types
/// @param[in] n_genes Number of genes
/// @param[out] assigned_labels Assigned labels [n_query] (non-null)
/// @param[out] correlation_scores Max correlation per cell [n_query] (non-null)
/// @param[out] all_correlations Optional: all correlations [n_query * n_types] (nullable)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_annotation_correlation_assignment(
    scl_sparse_t query_expression,
    scl_sparse_t reference_profiles,
    scl_index_t n_query,
    scl_index_t n_types,
    scl_index_t n_genes,
    scl_index_t* assigned_labels,
    scl_real_t* correlation_scores,
    scl_real_t* all_correlations
);

// =============================================================================
// Build Reference Profiles (Mean per Cell Type)
// =============================================================================

/// @brief Compute mean expression profile for each cell type
/// @param[in] expression Expression matrix (non-null)
/// @param[in] labels Cell type labels [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] n_genes Number of genes
/// @param[in] n_types Number of cell types
/// @param[out] profiles Output profiles [n_types * n_genes] (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_annotation_build_reference_profiles(
    scl_sparse_t expression,
    const scl_index_t* labels,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_types,
    scl_real_t* profiles
);

// =============================================================================
// Marker Gene Score (scType-style)
// =============================================================================

/// @brief Score cells based on marker gene expression
/// @param[in] expression Expression matrix (non-null)
/// @param[in] marker_genes Array of marker gene arrays per type (non-null)
/// @param[in] marker_counts Number of markers per type [n_types] (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] n_genes Number of genes
/// @param[in] n_types Number of cell types
/// @param[out] scores Output scores [n_cells * n_types] (non-null)
/// @param[in] normalize SCL_TRUE to apply softmax normalization
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_annotation_marker_gene_score(
    scl_sparse_t expression,
    const scl_index_t* const* marker_genes,
    const scl_index_t* marker_counts,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_types,
    scl_real_t* scores,
    scl_bool_t normalize
);

/// @brief Assign labels from marker scores
/// @param[in] scores Marker scores [n_cells * n_types] (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] n_types Number of cell types
/// @param[out] labels Assigned labels [n_cells] (non-null)
/// @param[out] confidence Max score per cell [n_cells] (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_annotation_assign_from_marker_scores(
    const scl_real_t* scores,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_index_t* labels,
    scl_real_t* confidence
);

// =============================================================================
// Consensus Annotation (Multiple Methods)
// =============================================================================

/// @brief Combine predictions from multiple annotation methods
/// @param[in] predictions Array of prediction arrays [n_methods] (non-null)
/// @param[in] confidences Array of confidence arrays (nullable)
/// @param[in] n_methods Number of methods
/// @param[in] n_cells Number of cells
/// @param[in] n_types Number of cell types
/// @param[out] consensus_labels Consensus labels [n_cells] (non-null)
/// @param[out] consensus_confidence Consensus confidence [n_cells] (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_annotation_consensus_annotation(
    const scl_index_t* const* predictions,
    const scl_real_t* const* confidences,
    scl_index_t n_methods,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_index_t* consensus_labels,
    scl_real_t* consensus_confidence
);

// =============================================================================
// Detect Novel Cell Types
// =============================================================================

/// @brief Detect cells that don't match reference types
/// @param[in] query_expression Query expression matrix (non-null)
/// @param[in] confidence_scores Annotation confidence [n_query] (non-null)
/// @param[in] n_query Number of query cells
/// @param[in] threshold Confidence threshold (cells below are novel)
/// @param[out] is_novel Novel flags [n_query] (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_annotation_detect_novel_types(
    scl_sparse_t query_expression,
    const scl_real_t* confidence_scores,
    scl_index_t n_query,
    scl_real_t threshold,
    scl_bool_t* is_novel
);

/// @brief Detect novel types by distance to assigned profile
/// @param[in] query_expression Query expression matrix (non-null)
/// @param[in] reference_profiles Reference profiles [n_types * n_genes] (non-null)
/// @param[in] assigned_labels Assigned labels [n_query] (non-null)
/// @param[in] n_query Number of query cells
/// @param[in] n_types Number of cell types
/// @param[in] n_genes Number of genes
/// @param[in] distance_threshold Distance threshold
/// @param[out] is_novel Novel flags [n_query] (non-null)
/// @param[out] distance_to_assigned Optional: distances [n_query] (nullable)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_annotation_detect_novel_by_distance(
    scl_sparse_t query_expression,
    const scl_real_t* reference_profiles,
    const scl_index_t* assigned_labels,
    scl_index_t n_query,
    scl_index_t n_types,
    scl_index_t n_genes,
    scl_real_t distance_threshold,
    scl_bool_t* is_novel,
    scl_real_t* distance_to_assigned
);

// =============================================================================
// Label Propagation
// =============================================================================

/// @brief Propagate labels through neighbor graph
/// @param[in] neighbor_graph KNN graph (non-null)
/// @param[in] initial_labels Initial labels, -1 for unlabeled [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] n_types Number of cell types
/// @param[in] max_iter Maximum iterations
/// @param[out] final_labels Output labels [n_cells] (non-null)
/// @param[out] label_confidence Confidence scores [n_cells] (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_annotation_label_propagation(
    scl_sparse_t neighbor_graph,
    const scl_index_t* initial_labels,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_index_t max_iter,
    scl_index_t* final_labels,
    scl_real_t* label_confidence
);

// =============================================================================
// Quality Metrics
// =============================================================================

/// @brief Compute annotation quality metrics
/// @param[in] predicted_labels Predicted labels [n_cells] (non-null)
/// @param[in] true_labels Ground truth labels [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] n_types Number of cell types
/// @param[out] accuracy Overall accuracy (non-null)
/// @param[out] macro_f1 Macro-averaged F1 score (non-null)
/// @param[out] per_class_f1 Optional: per-class F1 [n_types] (nullable)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_annotation_quality_metrics(
    const scl_index_t* predicted_labels,
    const scl_index_t* true_labels,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* accuracy,
    scl_real_t* macro_f1,
    scl_real_t* per_class_f1
);

/// @brief Compute confusion matrix
/// @param[in] predicted_labels Predicted labels [n_cells] (non-null)
/// @param[in] true_labels Ground truth labels [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] n_types Number of cell types
/// @param[out] confusion Confusion matrix [n_types * n_types] (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_annotation_confusion_matrix(
    const scl_index_t* predicted_labels,
    const scl_index_t* true_labels,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_index_t* confusion
);

/// @brief Compute entropy-based annotation uncertainty
/// @param[in] type_probabilities Type probabilities [n_cells * n_types] (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] n_types Number of cell types
/// @param[out] entropy Entropy values [n_cells] (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_annotation_entropy(
    const scl_real_t* type_probabilities,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* entropy
);

#ifdef __cplusplus
}
#endif
