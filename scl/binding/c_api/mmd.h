#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "core_types.h"

// =============================================================================
// FILE: scl/binding/c_api/mmd.h
// BRIEF: C API declarations for Maximum Mean Discrepancy
// =============================================================================

// Compute MMD with RBF kernel between two sparse matrices
scl_error_t scl_mmd_rbf(
    scl_sparse_matrix_t matrix_x,      // First sparse matrix (CSR)
    scl_sparse_matrix_t matrix_y,      // Second sparse matrix (CSR)
    scl_real_t* output,                // Output MMD values [n_rows]
    scl_real_t gamma                   // RBF kernel parameter (default: 1.0)
);

#ifdef __cplusplus
}
#endif
