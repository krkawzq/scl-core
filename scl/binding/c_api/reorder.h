#pragma once

// =============================================================================
// FILE: scl/binding/c_api/reorder/reorder.h
// BRIEF: C API for sparse matrix reordering
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Matrix Reordering
// =============================================================================

// Align secondary dimension (columns for CSR, rows for CSC) using index map
// Modifies matrix in-place and returns new lengths for each row/column
scl_error_t scl_reorder_align_secondary(
    scl_sparse_t matrix,                // Matrix to reorder (modified in-place)
    const scl_index_t* index_map,      // Mapping: old_index -> new_index [old_dim]
    scl_size_t old_dim,
    scl_index_t new_secondary_dim,      // New secondary dimension size
    scl_index_t* out_lengths           // Output: new lengths [primary_dim]
);

// Compute filtered NNZ after applying index map (does not modify matrix)
scl_error_t scl_reorder_compute_filtered_nnz(
    scl_sparse_t matrix,
    const scl_index_t* index_map,
    scl_size_t old_dim,
    scl_index_t new_secondary_dim,
    scl_size_t* out_nnz                 // Output: filtered NNZ
);

// Build inverse permutation: inverse[permutation[i]] = i
scl_error_t scl_reorder_build_inverse_permutation(
    const scl_index_t* permutation,     // Permutation array [n]
    scl_size_t n,
    scl_index_t* inverse                // Output: inverse permutation [n]
);

#ifdef __cplusplus
}
#endif
