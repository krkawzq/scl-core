#pragma once

// =============================================================================
// FILE: scl/binding/c_api/stat/kruskal_wallis.h
// BRIEF: C API for Kruskal-Wallis H test (non-parametric one-way ANOVA)
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// Kruskal-Wallis test for k groups
scl_error_t scl_stat_kruskal_wallis(
    scl_sparse_t matrix,              // Feature x sample matrix
    const int32_t* group_ids,         // Group assignments [n_samples]
    scl_size_t n_samples,
    scl_size_t n_groups,              // Number of groups (k >= 2)
    scl_real_t* H_stats,              // Output H statistics [n_features]
    scl_real_t* p_values,             // Output p-values [n_features]
    scl_size_t n_features
);

#ifdef __cplusplus
}
#endif
