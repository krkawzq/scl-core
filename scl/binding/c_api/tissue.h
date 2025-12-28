#pragma once

// =============================================================================
// FILE: scl/binding/c_api/tissue.h
// BRIEF: C API for tissue architecture and organization analysis
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Tissue Architecture Quantification
// =============================================================================

scl_error_t scl_tissue_architecture(
    const scl_real_t* coordinates,     // [n_cells * n_dims]
    scl_size_t n_cells,
    scl_size_t n_dims,
    const scl_index_t* cell_types,
    scl_real_t* density,               // Output [n_cells]
    scl_real_t* heterogeneity,         // Output [n_cells]
    scl_real_t* clustering_coef,       // Output [n_cells]
    scl_size_t n_neighbors
);

// =============================================================================
// Layer Assignment
// =============================================================================

scl_error_t scl_tissue_layer_assignment(
    const scl_real_t* coordinates,     // [n_cells * n_dims]
    scl_size_t n_cells,
    scl_size_t n_dims,
    scl_index_t n_layers,
    scl_index_t* layer_labels,         // Output [n_cells]
    scl_index_t reference_dim          // Dimension along which to define layers
);

// =============================================================================
// Radial Layer Assignment
// =============================================================================

scl_error_t scl_tissue_radial_layer_assignment(
    const scl_real_t* coordinates,     // [n_cells * n_dims]
    scl_size_t n_cells,
    scl_size_t n_dims,
    const scl_real_t* center,          // [n_dims]
    scl_index_t n_layers,
    scl_index_t* layer_labels          // Output [n_cells]
);

// =============================================================================
// Zonation Score
// =============================================================================

scl_error_t scl_tissue_zonation_score(
    scl_sparse_t expression,
    const scl_real_t* coordinates,
    scl_size_t n_dims,
    const scl_real_t* reference_point,  // [n_dims]
    scl_real_t* zonation_scores        // Output [n_cells]
);

// =============================================================================
// Morphological Features
// =============================================================================

scl_error_t scl_tissue_morphological_features(
    const scl_real_t* coordinates,     // [n_cells * n_dims]
    scl_size_t n_cells,
    scl_size_t n_dims,
    const scl_index_t* labels,
    scl_size_t n_groups,
    scl_real_t* area,                  // Output [n_groups]
    scl_real_t* perimeter,            // Output [n_groups]
    scl_real_t* circularity,          // Output [n_groups]
    scl_real_t* eccentricity          // Output [n_groups]
);

#ifdef __cplusplus
}
#endif
