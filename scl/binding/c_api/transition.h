#pragma once

// =============================================================================
// FILE: scl/binding/c_api/transition.h
// BRIEF: C API for cell state transition analysis
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "scl/binding/c_api/core_types.h"

// Transition type
typedef enum {
    SCL_TRANSITION_FORWARD = 0,
    SCL_TRANSITION_BACKWARD = 1,
    SCL_TRANSITION_SYMMETRIC = 2
} scl_transition_type_t;

// Normalize transition matrix
scl_error_t scl_transition_normalize(
    scl_sparse_matrix_t transition_mat,
    scl_index_t n,
    scl_real_t* output
);

// Compute stationary distribution
scl_error_t scl_transition_stationary_distribution(
    scl_sparse_matrix_t transition_mat,
    scl_index_t n,
    scl_real_t* stationary,
    scl_index_t max_iter,
    scl_real_t tol
);

// Compute absorption probability
scl_error_t scl_transition_absorption_probability(
    scl_sparse_matrix_t transition_mat,
    const scl_index_t* terminal_states,
    scl_index_t n,
    scl_index_t n_terminal,
    scl_real_t* absorption_probs,
    scl_index_t max_iter,
    scl_real_t tol
);

#ifdef __cplusplus
}
#endif
