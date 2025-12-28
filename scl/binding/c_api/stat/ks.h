#pragma once

// =============================================================================
// FILE: scl/binding/c_api/stat/ks.h
// BRIEF: C API for Kolmogorov-Smirnov two-sample test
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// Kolmogorov-Smirnov two-sample test (groups 0 and 1)
scl_error_t scl_stat_ks_test(
    scl_sparse_t matrix,              // Feature x sample matrix
    const int32_t* group_ids,         // Group assignments (0 or 1) [n_samples]
    scl_size_t n_samples,
    scl_real_t* D_stats,              // Output D statistics [n_features]
    scl_real_t* p_values,             // Output p-values [n_features]
    scl_size_t n_features
);

#ifdef __cplusplus
}
#endif
