#pragma once

// =============================================================================
// FILE: scl/binding/c_api/group/group.h
// BRIEF: C API for group aggregation statistics
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Group Statistics
// =============================================================================

// Compute group-wise statistics (mean and variance) for sparse matrix
scl_error_t scl_group_stats(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    scl_size_t n_groups,
    const scl_size_t* group_sizes,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    int ddof,
    int include_zeros
);

#ifdef __cplusplus
}
#endif
