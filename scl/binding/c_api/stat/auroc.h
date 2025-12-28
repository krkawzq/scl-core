#pragma once

// =============================================================================
// FILE: scl/binding/c_api/stat/auroc.h
// BRIEF: C API for AUROC (Area Under ROC Curve) computation
// =============================================================================

#include "scl/binding/c_api/core/core.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// AUROC Functions
// =============================================================================

// Compute AUROC and p-values for two-group comparison
scl_error_t scl_stat_auroc(
    scl_sparse_t matrix,
    const int32_t* group_ids,          // [secondary_dim] 0 or 1
    scl_real_t* out_auroc,             // [primary_dim] output
    scl_real_t* out_p_values,          // [primary_dim] output
    scl_index_t primary_dim
);

// Compute AUROC with log2 fold change
scl_error_t scl_stat_auroc_with_fc(
    scl_sparse_t matrix,
    const int32_t* group_ids,          // [secondary_dim] 0 or 1
    scl_real_t* out_auroc,              // [primary_dim] output
    scl_real_t* out_p_values,           // [primary_dim] output
    scl_real_t* out_log2_fc,            // [primary_dim] output
    scl_index_t primary_dim
);

#ifdef __cplusplus
}
#endif
