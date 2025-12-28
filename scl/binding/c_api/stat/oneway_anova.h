#pragma once

// =============================================================================
// FILE: scl/binding/c_api/stat/oneway_anova.h
// BRIEF: C API for one-way ANOVA F-test
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// One-way ANOVA F-test for k groups
scl_error_t scl_stat_oneway_anova(
    scl_sparse_t matrix,              // Feature x sample matrix
    const int32_t* group_ids,         // Group assignments [n_samples]
    scl_size_t n_samples,
    scl_size_t n_groups,              // Number of groups (k >= 2)
    scl_real_t* F_stats,              // Output F statistics [n_features]
    scl_real_t* p_values,             // Output p-values [n_features]
    scl_size_t n_features
);

#ifdef __cplusplus
}
#endif
