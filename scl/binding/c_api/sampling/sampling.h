#pragma once

// =============================================================================
// FILE: scl/binding/c_api/sampling/sampling.h
// BRIEF: C API for advanced sampling strategies
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Geometric Sketching
// =============================================================================

// Geometric sketching for preserving rare populations
scl_error_t scl_sampling_geometric_sketching(
    scl_sparse_t data,                  // Expression matrix (cells x genes, CSR)
    scl_size_t target_size,             // Target number of cells to select
    scl_index_t* selected_indices,      // Output: selected cell indices [target_size]
    scl_size_t* n_selected,              // Output: actual number selected
    uint64_t seed
);

// =============================================================================
// Density-Preserving Sampling
// =============================================================================

// Density-preserving downsampling using neighbor graph
scl_error_t scl_sampling_density_preserving(
    scl_sparse_t data,                  // Expression matrix (cells x genes, CSR)
    scl_sparse_t neighbors,             // Neighbor graph (cells x cells, CSR)
    scl_size_t target_size,
    scl_index_t* selected_indices,
    scl_size_t* n_selected
);

// =============================================================================
// Landmark Selection
// =============================================================================

// Landmark selection using KMeans++ initialization
scl_error_t scl_sampling_landmark_selection(
    scl_sparse_t data,
    scl_size_t n_landmarks,
    scl_index_t* landmark_indices,
    scl_size_t* n_selected,
    uint64_t seed
);

// =============================================================================
// Representative Cells
// =============================================================================

// Select representative cells per cluster (closest to centroid)
scl_error_t scl_sampling_representative_cells(
    scl_sparse_t data,
    const scl_index_t* cluster_labels,  // Cluster assignments [n_cells]
    scl_size_t n_cells,
    scl_size_t per_cluster,             // Number of representatives per cluster
    scl_index_t* representatives,        // Output: representative indices [max_count]
    scl_size_t* n_selected,              // Output: total number selected
    scl_size_t max_count,
    uint64_t seed
);

// =============================================================================
// Balanced Sampling
// =============================================================================

// Balanced sampling across groups
scl_error_t scl_sampling_balanced(
    const scl_index_t* labels,           // Group labels [n]
    scl_size_t n,
    scl_size_t target_size,
    scl_index_t* selected_indices,
    scl_size_t* n_selected,
    uint64_t seed
);

// =============================================================================
// Stratified Sampling
// =============================================================================

// Stratified sampling based on continuous variable
scl_error_t scl_sampling_stratified(
    const scl_real_t* values,           // Continuous values [n]
    scl_size_t n,
    scl_size_t n_strata,                // Number of strata
    scl_size_t target_size,
    scl_index_t* selected_indices,
    scl_size_t* n_selected,
    uint64_t seed
);

// =============================================================================
// Uniform Sampling
// =============================================================================

// Uniform random sampling
scl_error_t scl_sampling_uniform(
    scl_size_t n,
    scl_size_t target_size,
    scl_index_t* selected_indices,
    scl_size_t* n_selected,
    uint64_t seed
);

// =============================================================================
// Importance Sampling
// =============================================================================

// Importance sampling based on weights
scl_error_t scl_sampling_importance(
    const scl_real_t* weights,          // Sampling weights [n]
    scl_size_t n,
    scl_size_t target_size,
    scl_index_t* selected_indices,
    scl_size_t* n_selected,
    uint64_t seed
);

// =============================================================================
// Reservoir Sampling
// =============================================================================

// Reservoir sampling for streaming data
scl_error_t scl_sampling_reservoir(
    scl_size_t stream_size,
    scl_size_t reservoir_size,
    scl_index_t* reservoir,              // Output: reservoir [reservoir_size]
    scl_size_t* n_selected,
    uint64_t seed
);

#ifdef __cplusplus
}
#endif
