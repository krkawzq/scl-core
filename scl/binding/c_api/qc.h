#pragma once

// =============================================================================
// FILE: scl/binding/c_api/qc/qc.h
// BRIEF: C API for quality control metrics
// =============================================================================

#include "scl/binding/c_api/core.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Basic QC Metrics
// =============================================================================

// Compute basic QC metrics (n_genes, total_counts per cell)
scl_error_t scl_qc_compute_basic(
    scl_sparse_t matrix,
    scl_index_t* out_n_genes,         // [primary_dim] output
    scl_real_t* out_total_counts,     // [primary_dim] output
    scl_index_t primary_dim
);

// Compute subset percentage (e.g., mitochondrial genes)
scl_error_t scl_qc_compute_subset_pct(
    scl_sparse_t matrix,
    const uint8_t* subset_mask,      // [secondary_dim] 1 if in subset
    scl_real_t* out_pcts,             // [primary_dim] output percentages
    scl_index_t primary_dim
);

// Compute fused QC metrics (all in one pass)
scl_error_t scl_qc_compute_fused(
    scl_sparse_t matrix,
    const uint8_t* subset_mask,      // [secondary_dim] 1 if in subset
    scl_index_t* out_n_genes,         // [primary_dim] output
    scl_real_t* out_total_counts,     // [primary_dim] output
    scl_real_t* out_pcts,             // [primary_dim] output percentages
    scl_index_t primary_dim
);

#ifdef __cplusplus
}
#endif
