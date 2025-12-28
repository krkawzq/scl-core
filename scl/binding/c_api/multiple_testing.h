#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "core_types.h"

// =============================================================================
// FILE: scl/binding/c_api/multiple_testing.h
// BRIEF: C API declarations for Multiple Testing Correction
// =============================================================================

// Benjamini-Hochberg FDR correction
scl_error_t scl_benjamini_hochberg(
    const scl_real_t* p_values,        // Input p-values [n]
    scl_real_t* adjusted_p_values,     // Output adjusted p-values [n]
    scl_size_t n,                      // Number of tests
    scl_real_t fdr_level               // FDR level (default: 0.05)
);

// Bonferroni correction
scl_error_t scl_bonferroni(
    const scl_real_t* p_values,        // Input p-values [n]
    scl_real_t* adjusted_p_values,     // Output adjusted p-values [n]
    scl_size_t n                       // Number of tests
);

// Storey q-value estimation
scl_error_t scl_storey_qvalue(
    const scl_real_t* p_values,        // Input p-values [n]
    scl_real_t* q_values,              // Output q-values [n]
    scl_size_t n,                      // Number of tests
    scl_real_t lambda                   // Lambda parameter (default: 0.5)
);

// Local FDR estimation
scl_error_t scl_local_fdr(
    const scl_real_t* p_values,        // Input p-values [n]
    scl_real_t* lfdr,                  // Output local FDR [n]
    scl_size_t n                       // Number of tests
);

// Benjamini-Yekutieli FDR (for arbitrary dependency)
scl_error_t scl_benjamini_yekutieli(
    const scl_real_t* p_values,        // Input p-values [n]
    scl_real_t* adjusted_p_values,     // Output adjusted p-values [n]
    scl_size_t n                       // Number of tests
);

// Holm-Bonferroni step-down procedure
scl_error_t scl_holm_bonferroni(
    const scl_real_t* p_values,        // Input p-values [n]
    scl_real_t* adjusted_p_values,     // Output adjusted p-values [n]
    scl_size_t n                       // Number of tests
);

// Hochberg step-up procedure
scl_error_t scl_hochberg(
    const scl_real_t* p_values,        // Input p-values [n]
    scl_real_t* adjusted_p_values,     // Output adjusted p-values [n]
    scl_size_t n                       // Number of tests
);

// Count significant tests at given threshold
scl_size_t scl_count_significant(
    const scl_real_t* p_values,        // Input p-values [n]
    scl_size_t n,                      // Number of tests
    scl_real_t threshold               // P-value threshold
);

// Get indices of significant tests
scl_error_t scl_significant_indices(
    const scl_real_t* p_values,        // Input p-values [n]
    scl_size_t n,                      // Number of tests
    scl_real_t threshold,               // P-value threshold
    scl_index_t* out_indices,          // Output indices [out_count]
    scl_size_t* out_count               // Output: number of significant tests
);

// Compute negative log10 p-values
scl_error_t scl_neglog10_pvalues(
    const scl_real_t* p_values,        // Input p-values [n]
    scl_real_t* neglog_p,              // Output -log10(p) [n]
    scl_size_t n                       // Number of tests
);

// Fisher's method for combining p-values
scl_real_t scl_fisher_combine(
    const scl_real_t* p_values,        // Input p-values [n]
    scl_size_t n                       // Number of tests
);

// Stouffer's method for combining z-scores
scl_real_t scl_stouffer_combine(
    const scl_real_t* p_values,        // Input p-values [n]
    const scl_real_t* weights,          // Optional weights [n] or NULL
    scl_size_t n                       // Number of tests
);

#ifdef __cplusplus
}
#endif
