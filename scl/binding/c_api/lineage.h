#pragma once

// =============================================================================
// FILE: scl/binding/c_api/lineage.h
// BRIEF: C API for lineage tracing and fate mapping
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "scl/binding/c_api/core_types.h"

// Lineage coupling matrix: computes coupling between clones and cell types
scl_error_t scl_lineage_coupling(
    const scl_index_t* clone_ids,      // [n_cells]
    const scl_index_t* cell_types,     // [n_cells]
    scl_real_t* coupling_matrix,       // [n_clones * n_types] (output)
    scl_size_t n_cells,
    scl_size_t n_clones,
    scl_size_t n_types
);

// Fate bias score: computes bias score for each clone
scl_error_t scl_fate_bias(
    const scl_index_t* clone_ids,      // [n_cells]
    const scl_index_t* cell_types,     // [n_cells]
    scl_real_t* bias_scores,           // [n_clones] (output)
    scl_size_t n_cells,
    scl_size_t n_types
);

// Build lineage tree from clone IDs and pseudotime
scl_error_t scl_build_lineage_tree(
    const scl_index_t* clone_ids,      // [n_cells]
    const scl_real_t* pseudotime,      // [n_cells]
    scl_index_t* parent,                // [n_clones] (output), -1 for root
    scl_size_t n_cells,
    scl_size_t n_clones
);

// Compute lineage distance matrix
scl_error_t scl_lineage_distance(
    const scl_index_t* parent,          // [n_clones] parent array
    scl_real_t* distance_matrix,       // [n_clones * n_clones] (output)
    scl_size_t n_clones
);

// Barcode clone assignment: assign clone IDs based on barcode hashes
scl_error_t scl_barcode_clone_assignment(
    const uint64_t* barcode_hashes,     // [n_cells]
    scl_index_t* clone_ids,            // [n_cells] (output)
    scl_size_t n_cells,
    scl_size_t* n_unique_clones         // (output)
);

// Clonal fate probability: probability of each clone giving rise to each type
scl_error_t scl_clonal_fate_probability(
    const scl_index_t* clone_ids,       // [n_cells]
    const scl_index_t* cell_types,      // [n_cells]
    scl_real_t* fate_probs,            // [n_clones * n_types] (output)
    scl_size_t n_cells,
    scl_size_t n_clones,
    scl_size_t n_types
);

// Fate bias per cell type
scl_error_t scl_fate_bias_per_type(
    const scl_index_t* clone_ids,      // [n_cells]
    const scl_index_t* cell_types,      // [n_cells]
    scl_real_t* bias_per_type,         // [n_types] (output)
    scl_size_t n_cells,
    scl_size_t n_types
);

// Lineage sharing between types
scl_error_t scl_lineage_sharing(
    const scl_index_t* clone_ids,       // [n_cells]
    const scl_index_t* cell_types,      // [n_cells]
    scl_real_t* sharing_matrix,        // [n_types * n_types] (output)
    scl_size_t n_cells,
    scl_size_t n_types
);

// Lineage commitment score per cell
scl_error_t scl_lineage_commitment(
    const scl_index_t* clone_ids,       // [n_cells]
    const scl_index_t* cell_types,      // [n_cells]
    scl_real_t* commitment_scores,     // [n_cells] (output)
    scl_size_t n_cells,
    scl_size_t n_types
);

// Progenitor score per cell
scl_error_t scl_progenitor_score(
    const scl_index_t* clone_ids,       // [n_cells]
    const scl_index_t* cell_types,      // [n_cells]
    const scl_real_t* pseudotime,      // [n_cells]
    scl_real_t* progenitor_scores,    // [n_cells] (output)
    scl_size_t n_cells,
    scl_size_t n_types
);

// Lineage transition probability matrix
scl_error_t scl_lineage_transition_probability(
    const scl_index_t* clone_ids,       // [n_cells]
    const scl_index_t* cell_types,      // [n_cells]
    const scl_real_t* pseudotime,      // [n_cells]
    scl_real_t* transition_prob,       // [n_types * n_types] (output)
    scl_size_t n_cells,
    scl_size_t n_types
);

// Clone generation assignment
scl_error_t scl_clone_generation(
    const scl_index_t* parent,          // [n_clones] parent array
    scl_index_t* generation,            // [n_clones] (output)
    scl_size_t n_clones
);

#ifdef __cplusplus
}
#endif
