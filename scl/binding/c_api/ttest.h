#pragma once

// =============================================================================
// FILE: scl/binding/c_api/ttest.h
// BRIEF: C API for T-test with mask-based group partitioning
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// T-Test Computation
// =============================================================================

scl_error_t scl_ttest(
    scl_sparse_t matrix,
    const int32_t* group_ids,           // 0 or 1 for each cell
    scl_size_t n_cells,
    scl_real_t* out_t_stats,            // Output [primary_dim]
    scl_size_t t_stats_size,
    scl_real_t* out_p_values,           // Output [primary_dim]
    scl_size_t p_values_size,
    scl_real_t* out_log2_fc,            // Output [primary_dim]
    scl_size_t log2_fc_size,
    int use_welch                        // 1 = Welch's t-test, 0 = Student's
);

// =============================================================================
// Compute Group Statistics
// =============================================================================

scl_error_t scl_ttest_compute_group_stats(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    scl_size_t n_cells,
    scl_size_t n_groups,
    scl_real_t* out_means,               // Output [primary_dim * n_groups]
    scl_size_t means_size,
    scl_real_t* out_vars,                // Output [primary_dim * n_groups]
    scl_size_t vars_size,
    scl_size_t* out_counts               // Output [primary_dim * n_groups]
);

#ifdef __cplusplus
}
#endif
