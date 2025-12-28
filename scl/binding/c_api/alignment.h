#pragma once

// =============================================================================
// FILE: scl/binding/c_api/alignment.h
// BRIEF: C API for multi-modal data alignment and batch integration
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "core.h"

// =============================================================================
// Mutual Nearest Neighbors (MNN) Pairs
// =============================================================================

scl_error_t scl_mnn_pairs_f32_csr(
    scl_sparse_matrix_t data1,
    scl_sparse_matrix_t data2,
    scl_index_t k,
    scl_index_t* mnn_cell1,       // Output [max_pairs]
    scl_index_t* mnn_cell2,       // Output [max_pairs]
    scl_size_t max_pairs,
    scl_size_t* n_pairs           // Output: actual number of pairs
);

scl_error_t scl_mnn_pairs_f64_csr(
    scl_sparse_matrix_t data1,
    scl_sparse_matrix_t data2,
    scl_index_t k,
    scl_index_t* mnn_cell1,
    scl_index_t* mnn_cell2,
    scl_size_t max_pairs,
    scl_size_t* n_pairs
);

// =============================================================================
// Anchor Finding (Seurat-style)
// =============================================================================

scl_error_t scl_find_anchors_f32_csr(
    scl_sparse_matrix_t data1,
    scl_sparse_matrix_t data2,
    scl_index_t k,
    scl_index_t* anchor_cell1,    // Output [max_anchors]
    scl_index_t* anchor_cell2,     // Output [max_anchors]
    scl_real_t* anchor_scores,    // Output [max_anchors]
    scl_size_t max_anchors,
    scl_size_t* n_anchors         // Output: actual number of anchors
);

scl_error_t scl_find_anchors_f64_csr(
    scl_sparse_matrix_t data1,
    scl_sparse_matrix_t data2,
    scl_index_t k,
    scl_index_t* anchor_cell1,
    scl_index_t* anchor_cell2,
    double* anchor_scores,
    scl_size_t max_anchors,
    scl_size_t* n_anchors
);

// =============================================================================
// Label Transfer via Anchors
// =============================================================================

scl_error_t scl_transfer_labels(
    const scl_index_t* anchor_cell1,
    const scl_index_t* anchor_cell2,
    const scl_real_t* anchor_weights,
    scl_size_t n_anchors,
    const scl_index_t* source_labels,
    scl_size_t n_source,
    scl_size_t n_target,
    scl_index_t* target_labels,   // Output [n_target]
    scl_real_t* transfer_confidence  // Output [n_target]
);

// =============================================================================
// Integration Quality Score
// =============================================================================

scl_error_t scl_integration_score_f32_csr(
    scl_sparse_matrix_t integrated_data,
    const scl_index_t* batch_labels,
    scl_size_t n_cells,
    scl_sparse_matrix_t neighbors,
    scl_real_t* score             // Output
);

scl_error_t scl_integration_score_f64_csr(
    scl_sparse_matrix_t integrated_data,
    const scl_index_t* batch_labels,
    scl_size_t n_cells,
    scl_sparse_matrix_t neighbors,
    double* score
);

// =============================================================================
// Batch Mixing Metric
// =============================================================================

scl_error_t scl_batch_mixing(
    const scl_index_t* batch_labels,
    scl_size_t n_cells,
    scl_sparse_matrix_t neighbors,
    scl_real_t* mixing_scores     // Output [n_cells]
);

// =============================================================================
// kBET Score
// =============================================================================

scl_error_t scl_kbet_score(
    scl_sparse_matrix_t neighbors,
    const scl_index_t* batch_labels,
    scl_size_t n_cells,
    scl_real_t* score             // Output
);

#ifdef __cplusplus
}
#endif
