#pragma once

// =============================================================================
// FILE: scl/binding/c_api/multiple_testing/multiple_testing.h
// BRIEF: C API for multiple testing correction methods
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// FDR Correction Methods
// =============================================================================

// Benjamini-Hochberg FDR correction
scl_error_t scl_benjamini_hochberg(
    const scl_real_t* p_values,        // Input p-values [n_tests]
    scl_size_t n_tests,
    scl_real_t* adjusted_p_values,     // Output: adjusted p-values [n_tests]
    scl_real_t fdr_level                // FDR level (default 0.05)
);

// Bonferroni correction
scl_error_t scl_bonferroni(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t* adjusted_p_values
);

// Benjamini-Yekutieli FDR (for arbitrary dependency)
scl_error_t scl_benjamini_yekutieli(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t* adjusted_p_values
);

// Holm-Bonferroni step-down procedure
scl_error_t scl_holm_bonferroni(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t* adjusted_p_values
);

// Hochberg step-up procedure
scl_error_t scl_hochberg(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t* adjusted_p_values
);

// =============================================================================
// Q-Value Estimation
// =============================================================================

// Storey q-value estimation
scl_error_t scl_storey_qvalue(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t* q_values,               // Output: q-values [n_tests]
    scl_real_t lambda                    // Lambda parameter (default 0.5)
);

// =============================================================================
// Local FDR
// =============================================================================

// Local FDR estimation
scl_error_t scl_local_fdr(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t* lfdr                    // Output: local FDR [n_tests]
);

// =============================================================================
// Utility Functions
// =============================================================================

// Count significant tests at given threshold
scl_error_t scl_count_significant(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t threshold,
    scl_size_t* count                    // Output: number of significant tests
);

// Get indices of significant tests
scl_error_t scl_significant_indices(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t threshold,
    scl_index_t* out_indices,            // Output: significant indices [max_count]
    scl_size_t* out_count,               // Output: number of significant tests
    scl_size_t max_count
);

// Compute negative log10 p-values (for visualization)
scl_error_t scl_neglog10_pvalues(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t* neglog_p                 // Output: -log10(p) [n_tests]
);

// Fisher's method for combining p-values
scl_error_t scl_fisher_combine(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    scl_real_t* chi2_stat                // Output: chi-squared statistic
);

// Stouffer's method for combining z-scores
scl_error_t scl_stouffer_combine(
    const scl_real_t* p_values,
    scl_size_t n_tests,
    const scl_real_t* weights,           // Optional weights [n_tests] or NULL
    scl_real_t* z_score                  // Output: combined z-score
);

#ifdef __cplusplus
}
#endif
