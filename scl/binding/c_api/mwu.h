#pragma once

// =============================================================================
// FILE: scl/binding/c_api/mwu/mwu.h
// BRIEF: C API for Mann-Whitney U Test
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Mann-Whitney U Test
// =============================================================================

// Compute MWU test for each row/column in sparse matrix
// Groups are identified by group_ids: 0 = group 1, 1 = group 2
scl_error_t scl_mwu_test(
    scl_sparse_t matrix,                // Expression matrix (features x samples, CSR)
    const int32_t* group_ids,           // Group labels [n_samples]: 0 or 1
    scl_size_t n_samples,
    scl_real_t* out_u_stats,            // Output: U statistics [n_features]
    scl_real_t* out_p_values,            // Output: p-values [n_features]
    scl_real_t* out_log2_fc,             // Output: log2 fold change [n_features]
    scl_real_t* out_auroc                // Output: AUROC [n_features] (optional, can be NULL)
);

#ifdef __cplusplus
}
#endif
