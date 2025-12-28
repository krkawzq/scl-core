#pragma once

// =============================================================================
// FILE: scl/binding/c_api/alignment.h
// BRIEF: C API for multi-modal data alignment and batch integration
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Mutual Nearest Neighbors (MNN) Pairs
// =============================================================================

scl_error_t scl_alignment_mnn_pairs(
    scl_sparse_t data1,
    scl_sparse_t data2,
    scl_index_t k,
    scl_index_t* mnn_cell1,            // Output [n_pairs]
    scl_index_t* mnn_cell2,            // Output [n_pairs]
    scl_size_t* n_pairs                // Output: number of pairs found
);

// =============================================================================
// Anchor Finding (Seurat-style)
// =============================================================================

scl_error_t scl_alignment_find_anchors(
    scl_sparse_t data1,
    scl_sparse_t data2,
    scl_index_t k,
    scl_index_t* anchor_cell1,        // Output [n_anchors]
    scl_index_t* anchor_cell2,        // Output [n_anchors]
    scl_real_t* anchor_scores,        // Output [n_anchors]
    scl_size_t* n_anchors              // Output: number of anchors found
);

// =============================================================================
// Label Transfer via Anchors
// =============================================================================

scl_error_t scl_alignment_transfer_labels(
    const scl_index_t* anchor_cell1,
    const scl_index_t* anchor_cell2,
    const scl_real_t* anchor_weights,
    scl_size_t n_anchors,
    const scl_index_t* source_labels,
    scl_size_t n_source,
    scl_size_t n_target,
    scl_index_t* target_labels,        // Output [n_target]
    scl_real_t* transfer_confidence   // Output [n_target]
);

// =============================================================================
// Integration Quality Score
// =============================================================================

scl_error_t scl_alignment_integration_score(
    scl_sparse_t integrated_data,
    const scl_index_t* batch_labels,
    scl_size_t n_cells,
    scl_sparse_t neighbors,
    scl_real_t* score                  // Output
);

// =============================================================================
// Batch Mixing Metric
// =============================================================================

scl_error_t scl_alignment_batch_mixing(
    const scl_index_t* batch_labels,
    scl_size_t n_cells,
    scl_sparse_t neighbors,
    scl_real_t* mixing_scores          // Output [n_cells]
);

// =============================================================================
// Compute Correction Vectors (MNN-based)
// =============================================================================

scl_error_t scl_alignment_compute_correction_vectors(
    scl_sparse_t data1,
    scl_sparse_t data2,
    const scl_index_t* mnn_cell1,
    const scl_index_t* mnn_cell2,
    scl_size_t n_pairs,
    scl_real_t* correction_vectors,    // Output [n2 * n_features]
    scl_size_t n_features
);

// =============================================================================
// Smooth Correction Vectors using Gaussian Kernel
// =============================================================================

scl_error_t scl_alignment_smooth_correction_vectors(
    scl_sparse_t data2,
    scl_real_t* correction_vectors,    // In/out [n2 * n_features]
    scl_size_t n_features,
    scl_real_t sigma
);

// =============================================================================
// Canonical Correlation Analysis (CCA) for multimodal alignment
// =============================================================================

scl_error_t scl_alignment_cca_projection(
    scl_sparse_t data1,
    scl_sparse_t data2,
    scl_size_t n_components,
    scl_real_t* projection1,            // Output [n1 * n_components]
    scl_real_t* projection2             // Output [n2 * n_components]
);

// =============================================================================
// kBET (k-nearest neighbor Batch Effect Test)
// =============================================================================

scl_error_t scl_alignment_kbet_score(
    scl_sparse_t neighbors,
    const scl_index_t* batch_labels,
    scl_size_t n_cells,
    scl_real_t* score                  // Output
);

#ifdef __cplusplus
}
#endif
