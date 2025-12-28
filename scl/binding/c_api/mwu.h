#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "core_types.h"

// =============================================================================
// FILE: scl/binding/c_api/mwu.h
// BRIEF: C API declarations for Mann-Whitney U Test
// =============================================================================

// Perform Mann-Whitney U test on sparse matrix with group labels
scl_error_t scl_mwu_test(
    scl_sparse_matrix_t matrix,        // Sparse matrix (CSR), shape (n_features, n_samples)
    const int32_t* group_ids,          // Group labels [n_samples], must be 0 or 1
    scl_real_t* out_u_stats,           // Output U statistics [n_features]
    scl_real_t* out_p_values,          // Output p-values [n_features]
    scl_real_t* out_log2_fc,           // Output log2 fold change [n_features]
    scl_real_t* out_auroc              // Optional output AUROC [n_features] or NULL
);

#ifdef __cplusplus
}
#endif
