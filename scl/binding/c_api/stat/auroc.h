#pragma once

// =============================================================================
// FILE: scl/binding/c_api/stat/auroc.h
// BRIEF: C API for AUROC statistics
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "scl/binding/c_api/core_types.h"

// Compute AUROC
scl_error_t scl_stat_auroc(
    const scl_real_t* scores,
    const uint8_t* labels,
    scl_size_t n,
    scl_real_t* auroc
);

#ifdef __cplusplus
}
#endif
