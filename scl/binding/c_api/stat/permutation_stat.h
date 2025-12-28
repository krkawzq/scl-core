#pragma once

// =============================================================================
// FILE: scl/binding/c_api/stat/permutation_stat.h
// BRIEF: C API for permutation testing
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// Permutation statistic types
typedef enum {
    SCL_PERM_STAT_MWU = 0,       // Mann-Whitney U statistic
    SCL_PERM_STAT_MEAN_DIFF = 1, // Mean difference
    SCL_PERM_STAT_KS = 2         // Kolmogorov-Smirnov D statistic
} scl_perm_stat_type_t;

// Batch permutation test (reuse sort optimization)
scl_error_t scl_stat_batch_permutation(
    scl_sparse_t matrix,              // Feature x sample matrix
    const int32_t* group_ids,         // Group assignments (0 or 1) [n_samples]
    scl_size_t n_samples,
    scl_real_t* p_values,             // Output p-values [n_features]
    scl_size_t n_features,
    scl_size_t n_permutations,        // 0 = use default 1000
    scl_perm_stat_type_t stat_type,
    uint64_t seed
);

// Single feature permutation test
scl_error_t scl_stat_permutation_single(
    const scl_real_t* values,         // Feature values [n_samples]
    scl_size_t n_samples,
    const int32_t* group_ids,         // Group assignments (0 or 1) [n_samples]
    scl_size_t n_permutations,        // 0 = use default 1000
    scl_perm_stat_type_t stat_type,
    uint64_t seed,
    scl_real_t* p_value_out
);

#ifdef __cplusplus
}
#endif
