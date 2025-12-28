#pragma once

// =============================================================================
// FILE: scl/binding/c_api/log1p/log1p.h
// BRIEF: C API for logarithmic transforms
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Logarithmic Transforms (In-Place)
// =============================================================================

scl_error_t scl_log1p_inplace(scl_sparse_t* matrix);

scl_error_t scl_log2p1_inplace(scl_sparse_t* matrix);

scl_error_t scl_expm1_inplace(scl_sparse_t* matrix);

#ifdef __cplusplus
}
#endif
