#pragma once

// =============================================================================
// FILE: scl/binding/c_api/bbknn.h
// BRIEF: C API for Batch Balanced K-Nearest Neighbors (BBKNN)
// =============================================================================
//
// PURPOSE:
//   - Batch-aware neighbor graph construction
//   - Mitigates batch effects by balancing neighbors across batches
//   - Essential for batch correction in single-cell analysis
//
// ALGORITHM:
//   - For each cell, find k nearest neighbors within EACH batch
//   - Total neighbors per cell: n_batches * k
//   - Uses cosine distance (1 - cosine similarity)
//
// OPTIMIZATIONS:
//   - SIMD-accelerated sparse dot products
//   - Cache-blocked distance computation
//   - Efficient k-heap with manual sift operations
//   - Batch-grouped processing (minimizes redundant distance calculations)
//   - Prefetching for indirect memory access
//
// APPLICATIONS:
//   - Single-cell RNA-seq batch correction
//   - Multi-dataset integration
//   - Removing technical variation while preserving biology
//
// THREAD SAFETY:
//   - All operations are thread-safe
//   - Automatic parallelization over cells
//
// MEMORY:
//   - Output arrays must be pre-allocated: [n_cells * n_batches * k]
//   - Temporary memory: O(n_cells + n_threads * n_batches * k)
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Compute Norms (Utility Function)
// =============================================================================

/// @brief Precompute squared L2 norms for each cell (optimization)
/// @param[in] matrix Expression matrix (non-null)
/// @param[out] norms_sq Squared norms [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @return SCL_OK on success, error code otherwise
/// @note Use this for repeated BBKNN calls on same matrix
scl_error_t scl_bbknn_compute_norms(
    scl_sparse_t matrix,
    scl_real_t* norms_sq,
    scl_size_t n_cells
);

// =============================================================================
// BBKNN: Batch Balanced K-Nearest Neighbors
// =============================================================================

/// @brief Compute batch-balanced KNN graph
/// @param[in] matrix Expression matrix (cells x genes) (non-null)
/// @param[in] batch_labels Batch assignment [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] n_batches Number of batches
/// @param[in] k Neighbors per batch
/// @param[out] out_indices Neighbor indices [n_cells * n_batches * k] (non-null)
/// @param[out] out_distances Neighbor distances [n_cells * n_batches * k] (non-null)
/// @return SCL_OK on success, error code otherwise
///
/// @details For each cell i:
///   - Finds k nearest neighbors in batch 0 -> out[i*n_batches*k + 0*k : 1*k]
///   - Finds k nearest neighbors in batch 1 -> out[i*n_batches*k + 1*k : 2*k]
///   - ...
///   - Total neighbors per cell: n_batches * k
///
/// @note Distance metric: Euclidean (L2)
/// @note Caller must pre-allocate output arrays
/// @note Automatically parallelized over cells
scl_error_t scl_bbknn(
    scl_sparse_t matrix,
    const int32_t* batch_labels,
    scl_size_t n_cells,
    scl_size_t n_batches,
    scl_size_t k,
    scl_index_t* out_indices,
    scl_real_t* out_distances
);

/// @brief BBKNN with precomputed norms (faster for repeated calls)
/// @param[in] matrix Expression matrix (non-null)
/// @param[in] batch_labels Batch assignment [n_cells] (non-null)
/// @param[in] n_cells Number of cells
/// @param[in] n_batches Number of batches
/// @param[in] k Neighbors per batch
/// @param[in] norms_sq Precomputed squared norms [n_cells] (non-null)
/// @param[out] out_indices Neighbor indices (non-null)
/// @param[out] out_distances Neighbor distances (non-null)
/// @return SCL_OK on success, error code otherwise
///
/// @note Use scl_bbknn_compute_norms() first to precompute norms
/// @note Skips norm computation (faster for multiple BBKNN calls)
scl_error_t scl_bbknn_with_norms(
    scl_sparse_t matrix,
    const int32_t* batch_labels,
    scl_size_t n_cells,
    scl_size_t n_batches,
    scl_size_t k,
    const scl_real_t* norms_sq,
    scl_index_t* out_indices,
    scl_real_t* out_distances
);

#ifdef __cplusplus
}
#endif
