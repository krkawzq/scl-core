#pragma once

// =============================================================================
// FILE: scl/binding/c_api/transition.h
// BRIEF: C API for cell state transition analysis
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Build Transition Matrix from Velocity Graph
// =============================================================================

scl_error_t scl_transition_matrix_from_velocity(
    scl_sparse_t velocity_graph,
    scl_index_t n,
    scl_real_t* row_stochastic_out      // Output [n * n], row-major
);

// =============================================================================
// Row-Normalize Sparse Matrix to Stochastic
// =============================================================================

scl_error_t scl_transition_row_normalize_to_stochastic(
    scl_sparse_t input,
    scl_index_t n,
    scl_real_t* output_values,          // Output [nnz]
    scl_size_t output_size
);

// =============================================================================
// Symmetrize Transition Matrix
// =============================================================================

scl_error_t scl_transition_symmetrize(
    scl_sparse_t transition_mat,
    scl_index_t n,
    scl_real_t* symmetric_out           // Output [n * n], row-major
);

// =============================================================================
// Stationary Distribution
// =============================================================================

scl_error_t scl_transition_stationary_distribution(
    scl_sparse_t transition_mat,
    scl_index_t n,
    scl_real_t* stationary,             // Output [n]
    scl_index_t max_iter,
    scl_real_t tolerance
);

#ifdef __cplusplus
}
#endif
