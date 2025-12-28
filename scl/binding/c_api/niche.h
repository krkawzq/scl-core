#pragma once

// =============================================================================
// FILE: scl/binding/c_api/niche/niche.h
// BRIEF: C API for cellular neighborhood and microenvironment analysis
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Neighborhood Composition
// =============================================================================

scl_error_t scl_niche_neighborhood_composition(
    scl_sparse_t spatial_neighbors,       // Spatial adjacency graph
    const scl_index_t* cell_type_labels,  // [n_cells]
    scl_index_t n_cell_types,
    scl_real_t* composition_output        // Output [n_cells * n_cell_types]
);

// =============================================================================
// Neighborhood Enrichment Analysis
// =============================================================================

scl_error_t scl_niche_neighborhood_enrichment(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cell_types,
    scl_real_t* enrichment_scores,        // Output [n_cell_types * n_cell_types]
    scl_real_t* p_values,                 // Output [n_cell_types * n_cell_types]
    scl_index_t n_permutations            // Number of permutations
);

// =============================================================================
// Cell-Cell Contact Matrix
// =============================================================================

scl_error_t scl_niche_cell_cell_contact(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cell_types,
    scl_real_t* contact_matrix            // Output [n_cell_types * n_cell_types]
);

// =============================================================================
// Co-localization Score
// =============================================================================

scl_error_t scl_niche_colocalization_score(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cell_types,
    scl_index_t type_a,
    scl_index_t type_b,
    scl_real_t* colocalization,           // Output: colocalization score
    scl_real_t* p_value,                  // Output: p-value
    scl_index_t n_permutations
);

// =============================================================================
// Co-localization Matrix
// =============================================================================

scl_error_t scl_niche_colocalization_matrix(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cell_types,
    scl_real_t* coloc_matrix              // Output [n_cell_types * n_cell_types]
);

// =============================================================================
// Niche Similarity
// =============================================================================

scl_error_t scl_niche_similarity(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cell_types,
    const scl_index_t* query_cells,      // [n_query]
    scl_size_t n_query,
    scl_real_t* similarity_output         // Output [n_query * n_query]
);

// =============================================================================
// Niche Diversity
// =============================================================================

scl_error_t scl_niche_diversity(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cell_types,
    scl_real_t* diversity_output          // Output [n_cells]
);

// =============================================================================
// Niche Boundary Detection
// =============================================================================

scl_error_t scl_niche_boundary_score(
    scl_sparse_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cell_types,
    scl_real_t* boundary_scores           // Output [n_cells]
);

#ifdef __cplusplus
}
#endif
