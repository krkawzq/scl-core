#pragma once

// =============================================================================
// FILE: scl/binding/c_api/hotspot/hotspot.h
// BRIEF: C API for spatial hotspot detection
// =============================================================================

#include "scl/binding/c_api/core.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Local Moran's I (LISA)
scl_error_t scl_hotspot_local_morans_i(
    scl_sparse_t spatial_weights,
    const scl_real_t* values,
    scl_index_t n,
    scl_real_t* local_i,
    scl_real_t* z_scores,
    scl_real_t* p_values,
    scl_index_t n_permutations,
    uint64_t seed
);

// Getis-Ord Gi*
scl_error_t scl_hotspot_getis_ord_g_star(
    scl_sparse_t spatial_weights,
    const scl_real_t* values,
    scl_index_t n,
    scl_real_t* g_star,
    scl_real_t* z_scores,
    scl_real_t* p_values,
    int include_self,
    scl_index_t n_permutations,
    uint64_t seed
);

// Local Geary's C
scl_error_t scl_hotspot_local_gearys_c(
    scl_sparse_t spatial_weights,
    const scl_real_t* values,
    scl_index_t n,
    scl_real_t* local_c,
    scl_real_t* z_scores,
    scl_real_t* p_values,
    scl_index_t n_permutations,
    uint64_t seed
);

// Global Moran's I
scl_error_t scl_hotspot_global_morans_i(
    scl_sparse_t spatial_weights,
    const scl_real_t* values,
    scl_index_t n,
    scl_real_t* moran_i,
    scl_real_t* z_score,
    scl_real_t* p_value
);

// Global Geary's C
scl_error_t scl_hotspot_global_gearys_c(
    scl_sparse_t spatial_weights,
    const scl_real_t* values,
    scl_index_t n,
    scl_real_t* geary_c,
    scl_real_t* z_score,
    scl_real_t* p_value
);

#ifdef __cplusplus
}
#endif
