#pragma once

// =============================================================================
// FILE: scl/binding/c_api/stat/effect_size.h
// BRIEF: C API for effect size computation
// =============================================================================

#include "scl/binding/c_api/core/core.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Effect Size Types
// =============================================================================

typedef enum {
    SCL_EFFECT_SIZE_COHENS_D = 0,
    SCL_EFFECT_SIZE_HEDGES_G = 1,
    SCL_EFFECT_SIZE_GLASS_DELTA = 2,
    SCL_EFFECT_SIZE_CLES = 3
} scl_effect_size_type_t;

// =============================================================================
// Effect Size Functions
// =============================================================================

// Compute effect size for two-group comparison
scl_error_t scl_stat_effect_size(
    scl_sparse_t matrix,
    const int32_t* group_ids,          // [secondary_dim] 0 or 1
    scl_real_t* out_effect_size,       // [primary_dim] output
    scl_index_t primary_dim,
    scl_effect_size_type_t type
);

// Compute T-test with effect size
scl_error_t scl_stat_ttest_with_effect_size(
    scl_sparse_t matrix,
    const int32_t* group_ids,          // [secondary_dim] 0 or 1
    scl_real_t* out_t_stats,           // [primary_dim] output
    scl_real_t* out_p_values,          // [primary_dim] output
    scl_real_t* out_log2_fc,           // [primary_dim] output
    scl_real_t* out_effect_size,       // [primary_dim] output
    scl_index_t primary_dim,
    scl_effect_size_type_t es_type,
    int32_t use_welch                  // 1 for Welch's t-test, 0 for pooled
);

#ifdef __cplusplus
}
#endif
