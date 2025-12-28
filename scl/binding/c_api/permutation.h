#pragma once

// =============================================================================
// FILE: scl/binding/c_api/permutation/permutation.h
// BRIEF: C API for permutation testing framework
// =============================================================================

#include "scl/binding/c_api/core.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Permutation Correlation Test
// =============================================================================

scl_error_t scl_perm_correlation_test(
    const scl_real_t* x,                  // [n]
    const scl_real_t* y,                  // [n]
    scl_size_t n,
    scl_real_t observed_correlation,
    scl_size_t n_permutations,
    scl_real_t* p_value,                 // Output: p-value
    uint64_t seed
);

// =============================================================================
// FDR Corrections
// =============================================================================

scl_error_t scl_perm_fdr_correction_bh(
    const scl_real_t* p_values,           // [n]
    scl_size_t n,
    scl_real_t* q_values                 // Output [n]
);

scl_error_t scl_perm_fdr_correction_by(
    const scl_real_t* p_values,           // [n]
    scl_size_t n,
    scl_real_t* q_values                 // Output [n]
);

// =============================================================================
// Multiple Testing Corrections
// =============================================================================

scl_error_t scl_perm_bonferroni_correction(
    const scl_real_t* p_values,           // [n]
    scl_size_t n,
    scl_real_t* adjusted                  // Output [n]
);

scl_error_t scl_perm_holm_correction(
    const scl_real_t* p_values,           // [n]
    scl_size_t n,
    scl_real_t* adjusted                  // Output [n]
);

// =============================================================================
// Utilities
// =============================================================================

scl_error_t scl_perm_count_significant(
    const scl_real_t* p_values,           // [n]
    scl_size_t n,
    scl_real_t alpha,
    scl_size_t* n_significant            // Output: count of significant tests
);

scl_error_t scl_perm_get_significant_indices(
    const scl_real_t* p_values,           // [n]
    scl_size_t n,
    scl_real_t alpha,
    scl_index_t* indices,                 // Output [max_results]
    scl_size_t max_results,
    scl_size_t* n_results                // Output: actual number of results
);

// =============================================================================
// Batch Permutation Test
// =============================================================================

scl_error_t scl_perm_batch_test(
    scl_sparse_t matrix,
    const scl_index_t* group_labels,      // [n_cols]
    scl_size_t n_permutations,
    scl_real_t* p_values,                 // Output [n_rows]
    uint64_t seed
);

#ifdef __cplusplus
}
#endif
