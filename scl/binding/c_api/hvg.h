#pragma once

// =============================================================================
// FILE: scl/binding/c_api/hvg/hvg.h
// BRIEF: C API for highly variable gene selection
// =============================================================================

#include "scl/binding/c_api/core.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compute mean and variance for each row/column
scl_error_t scl_hvg_compute_moments(
    scl_sparse_t matrix,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    int ddof
);

// Compute clipped moments (values clipped to threshold)
scl_error_t scl_hvg_compute_clipped_moments(
    scl_sparse_t matrix,
    const scl_real_t* clip_vals,
    scl_real_t* out_means,
    scl_real_t* out_vars
);

// Select highly variable genes by dispersion
scl_error_t scl_hvg_select_by_dispersion(
    scl_sparse_t matrix,
    scl_size_t n_top,
    scl_index_t* out_indices,
    uint8_t* out_mask,
    scl_real_t* out_dispersions
);

// Select highly variable genes by VST
scl_error_t scl_hvg_select_by_vst(
    scl_sparse_t matrix,
    const scl_real_t* clip_vals,
    scl_size_t n_top,
    scl_index_t* out_indices,
    uint8_t* out_mask,
    scl_real_t* out_variances
);

#ifdef __cplusplus
}
#endif
