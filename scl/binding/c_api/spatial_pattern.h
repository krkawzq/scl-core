#pragma once

// =============================================================================
// FILE: scl/binding/c_api/spatial_pattern/spatial_pattern.h
// BRIEF: C API for spatial pattern detection
// =============================================================================

#include "scl/binding/c_api/core.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Spatial Variability (SpatialDE-style)
// =============================================================================

scl_error_t scl_spatial_pattern_variability(
    scl_sparse_t expression,               // Expression matrix [n_cells, n_genes]
    const scl_real_t* coordinates,         // [n_cells * n_dims]
    scl_size_t n_dims,
    scl_real_t* variability_scores,         // Output [n_genes]
    scl_real_t* p_values,                 // Output [n_genes]
    scl_size_t n_permutations,
    uint64_t seed
);

// =============================================================================
// Spatial Gradient Detection
// =============================================================================

scl_error_t scl_spatial_pattern_gradient(
    const scl_real_t* expression,          // [n_cells]
    const scl_real_t* coordinates,         // [n_cells * n_dims]
    scl_size_t n_cells,
    scl_size_t n_dims,
    scl_real_t* gradient_direction,        // Output [n_dims]
    scl_real_t* gradient_strength          // Output: scalar
);

// =============================================================================
// Periodic Pattern Detection
// =============================================================================

scl_error_t scl_spatial_pattern_periodic(
    scl_sparse_t expression,               // Expression matrix [n_cells, n_genes]
    const scl_real_t* coordinates,         // [n_cells * n_dims]
    scl_size_t n_dims,
    scl_real_t* periodicity_scores,         // Output [n_genes]
    scl_real_t* dominant_wavelengths,       // Output [n_genes]
    scl_size_t n_wavelengths,
    const scl_real_t* test_wavelengths     // [n_wavelengths]
);

// =============================================================================
// Boundary Detection
// =============================================================================

scl_error_t scl_spatial_pattern_boundary(
    scl_sparse_t expression,               // Expression matrix [n_cells, n_genes]
    const scl_real_t* coordinates,         // [n_cells * n_dims]
    scl_size_t n_dims,
    scl_real_t* boundary_scores,           // Output [n_cells]
    scl_size_t n_neighbors
);

// =============================================================================
// Spatial Domain Identification
// =============================================================================

scl_error_t scl_spatial_pattern_domain(
    scl_sparse_t expression,               // Expression matrix [n_cells, n_genes]
    const scl_real_t* coordinates,         // [n_cells * n_dims]
    scl_size_t n_dims,
    scl_index_t n_domains,
    scl_index_t* domain_labels,            // Output [n_cells]
    uint64_t seed
);

// =============================================================================
// Hotspot Analysis (Getis-Ord Gi*)
// =============================================================================

scl_error_t scl_spatial_pattern_hotspot(
    const scl_real_t* values,              // [n_cells]
    const scl_real_t* coordinates,         // [n_cells * n_dims]
    scl_size_t n_cells,
    scl_size_t n_dims,
    scl_real_t bandwidth,
    scl_real_t* gi_scores,                 // Output [n_cells]
    scl_real_t* z_scores                   // Output [n_cells]
);

// =============================================================================
// Spatial Autocorrelation Per Gene
// =============================================================================

scl_error_t scl_spatial_pattern_autocorrelation(
    scl_sparse_t expression,                // Expression matrix [n_cells, n_genes]
    const scl_real_t* coordinates,         // [n_cells * n_dims]
    scl_size_t n_dims,
    scl_real_t* morans_i,                  // Output [n_genes]
    scl_real_t* gearys_c                   // Output [n_genes]
);

// =============================================================================
// Spatial Smoothing
// =============================================================================

scl_error_t scl_spatial_pattern_smoothing(
    scl_sparse_t expression,                // Expression matrix [n_cells, n_genes]
    const scl_real_t* coordinates,         // [n_cells * n_dims]
    scl_size_t n_dims,
    scl_real_t bandwidth,
    scl_real_t* smoothed                   // Output [n_cells * n_genes]
);

// =============================================================================
// Spatial Co-expression
// =============================================================================

scl_error_t scl_spatial_pattern_coexpression(
    scl_sparse_t expression,                // Expression matrix [n_cells, n_genes]
    const scl_real_t* coordinates,         // [n_cells * n_dims]
    scl_size_t n_dims,
    const scl_index_t* gene_pairs,         // [n_pairs * 2]
    scl_size_t n_pairs,
    scl_real_t* coexpression_scores        // Output [n_pairs]
);

// =============================================================================
// Ripley's K Function
// =============================================================================

scl_error_t scl_spatial_pattern_ripleys_k(
    const scl_real_t* coordinates,         // [n_cells * n_dims]
    scl_size_t n_cells,
    scl_size_t n_dims,
    const scl_real_t* radii,               // [n_radii]
    scl_size_t n_radii,
    scl_real_t* k_values,                 // Output [n_radii]
    scl_real_t study_area
);

// =============================================================================
// Spatial Entropy
// =============================================================================

scl_error_t scl_spatial_pattern_entropy(
    const scl_index_t* labels,             // [n_cells]
    const scl_real_t* coordinates,         // [n_cells * n_dims]
    scl_size_t n_cells,
    scl_size_t n_dims,
    scl_real_t bandwidth,
    scl_real_t* entropy_scores             // Output [n_cells]
);

#ifdef __cplusplus
}
#endif
