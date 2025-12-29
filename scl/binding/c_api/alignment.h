#pragma once

// =============================================================================
// FILE: scl/binding/c_api/alignment.h
// BRIEF: C API for multi-modal data alignment and batch integration
// =============================================================================
//
// METHODS:
//   - MNN (Mutual Nearest Neighbors): fastMNN-style integration
//   - Anchors: Seurat-style anchor finding and label transfer
//   - CCA: Canonical Correlation Analysis for multimodal alignment
//   - kBET: Batch effect quantification
//
// APPLICATIONS:
//   - Batch effect correction
//   - Multi-sample integration
//   - Cross-dataset label transfer
//   - Integration quality assessment
//
// WORKFLOW:
//   1. Find MNN pairs or anchors between datasets
//   2. Compute correction vectors
//   3. Smooth corrections (optional)
//   4. Transfer labels via anchors
//   5. Assess integration quality (mixing, kBET)
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
// Mutual Nearest Neighbors (MNN)
// =============================================================================

/// @brief Find mutual nearest neighbor pairs between two datasets
/// @param[in] data1 First dataset (non-null)
/// @param[in] data2 Second dataset (non-null)
/// @param[in] k Number of nearest neighbors
/// @param[out] mnn_cell1 Cell indices from data1 (non-null, caller-allocated)
/// @param[out] mnn_cell2 Cell indices from data2 (non-null, caller-allocated)
/// @param[out] n_pairs Number of MNN pairs found (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note Caller must pre-allocate arrays with sufficient size
scl_error_t scl_alignment_mnn_pairs(
    scl_sparse_t data1,
    scl_sparse_t data2,
    scl_index_t k,
    scl_index_t* mnn_cell1,
    scl_index_t* mnn_cell2,
    scl_size_t* n_pairs
);

// =============================================================================
// Anchor Finding (Seurat-style)
// =============================================================================

/// @brief Find integration anchors between datasets
/// @param[in] data1 First dataset (non-null)
/// @param[in] data2 Second dataset (non-null)
/// @param[in] k Number of nearest neighbors for anchor scoring
/// @param[out] anchor_cell1 Anchor cell indices from data1 (non-null)
/// @param[out] anchor_cell2 Anchor cell indices from data2 (non-null)
/// @param[out] anchor_scores Anchor quality scores (non-null)
/// @param[out] n_anchors Number of anchors found (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_alignment_find_anchors(
    scl_sparse_t data1,
    scl_sparse_t data2,
    scl_index_t k,
    scl_index_t* anchor_cell1,
    scl_index_t* anchor_cell2,
    scl_real_t* anchor_scores,
    scl_size_t* n_anchors
);

// =============================================================================
// Label Transfer
// =============================================================================

/// @brief Transfer labels from source to target via anchors
/// @param[in] anchor_cell1 Source cell indices [n_anchors] (non-null)
/// @param[in] anchor_cell2 Target cell indices [n_anchors] (non-null)
/// @param[in] anchor_weights Anchor weights [n_anchors] (non-null)
/// @param[in] n_anchors Number of anchors
/// @param[in] source_labels Source labels [n_source] (non-null)
/// @param[in] n_source Number of source cells
/// @param[in] n_target Number of target cells
/// @param[out] target_labels Output labels [n_target] (non-null)
/// @param[out] transfer_confidence Confidence scores [n_target] (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_alignment_transfer_labels(
    const scl_index_t* anchor_cell1,
    const scl_index_t* anchor_cell2,
    const scl_real_t* anchor_weights,
    scl_size_t n_anchors,
    const scl_index_t* source_labels,
    scl_size_t n_source,
    scl_size_t n_target,
    scl_index_t* target_labels,
    scl_real_t* transfer_confidence
);

// =============================================================================
// Integration Quality
// =============================================================================

/// @brief Compute integration quality score (entropy-based)
/// @param[in] integrated_data Integrated expression data (non-null)
/// @param[in] batch_labels Batch assignment [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] neighbors KNN graph on integrated space (non-null)
/// @param[out] score Integration quality score [0, 1] (non-null)
/// @return SCL_OK on success, error code otherwise
/// @note Higher score = better mixing across batches
scl_error_t scl_alignment_integration_score(
    scl_sparse_t integrated_data,
    const scl_index_t* batch_labels,
    scl_size_t n_cells,
    scl_sparse_t neighbors,
    scl_real_t* score
);

/// @brief Compute batch mixing scores per cell
/// @param[in] batch_labels Batch assignment [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] neighbors KNN graph (non-null)
/// @param[out] mixing_scores Mixing scores [n_cells] (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_alignment_batch_mixing(
    const scl_index_t* batch_labels,
    scl_size_t n_cells,
    scl_sparse_t neighbors,
    scl_real_t* mixing_scores
);

/// @brief Compute kBET score for batch effect testing
/// @param[in] neighbors KNN graph (non-null)
/// @param[in] batch_labels Batch assignment [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[out] score kBET acceptance rate [0, 1] (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_alignment_kbet_score(
    scl_sparse_t neighbors,
    const scl_index_t* batch_labels,
    scl_size_t n_cells,
    scl_real_t* score
);

// =============================================================================
// Correction Vectors
// =============================================================================

/// @brief Compute MNN-based correction vectors
/// @param[in] data1 First dataset (non-null)
/// @param[in] data2 Second dataset (non-null)
/// @param[in] mnn_cell1 MNN indices from data1 [n_pairs] (non-null)
/// @param[in] mnn_cell2 MNN indices from data2 [n_pairs] (non-null)
/// @param[in] n_pairs Number of MNN pairs
/// @param[out] correction_vectors Corrections [n2 * n_features] (non-null)
/// @param[in] n_features Number of features
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_alignment_compute_correction_vectors(
    scl_sparse_t data1,
    scl_sparse_t data2,
    const scl_index_t* mnn_cell1,
    const scl_index_t* mnn_cell2,
    scl_size_t n_pairs,
    scl_real_t* correction_vectors,
    scl_size_t n_features
);

/// @brief Smooth correction vectors using Gaussian kernel
/// @param[in] data2 Second dataset (non-null)
/// @param[in,out] correction_vectors Corrections [n2 * n_features] (non-null)
/// @param[in] n_features Number of features
/// @param[in] sigma Gaussian kernel bandwidth
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_alignment_smooth_correction_vectors(
    scl_sparse_t data2,
    scl_real_t* correction_vectors,
    scl_size_t n_features,
    scl_real_t sigma
);

// =============================================================================
// CCA Projection
// =============================================================================

/// @brief Project data onto CCA space for alignment
/// @param[in] data1 First modality (non-null)
/// @param[in] data2 Second modality (non-null)
/// @param[in] n_components Number of CCA components
/// @param[out] projection1 Projections [n1 * n_components] (non-null)
/// @param[out] projection2 Projections [n2 * n_components] (non-null)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_alignment_cca_projection(
    scl_sparse_t data1,
    scl_sparse_t data2,
    scl_size_t n_components,
    scl_real_t* projection1,
    scl_real_t* projection2
);

#ifdef __cplusplus
}
#endif
