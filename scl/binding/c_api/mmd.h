#pragma once

// =============================================================================
// FILE: scl/binding/c_api/mmd/mmd.h
// BRIEF: C API for Maximum Mean Discrepancy (MMD) with RBF kernel
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Maximum Mean Discrepancy
// =============================================================================

// Compute MMD between two sparse matrices using RBF kernel
// Output: MMD^2 values [n_samples]
scl_error_t scl_mmd_rbf(
    scl_sparse_t mat_x,                // First matrix (samples x features, CSR)
    scl_sparse_t mat_y,                // Second matrix (samples x features, CSR)
    scl_real_t* output,                // Output: MMD^2 [n_samples]
    scl_real_t gamma                    // RBF kernel parameter (default 1.0)
);

#ifdef __cplusplus
}
#endif
