#pragma once

// =============================================================================
// FILE: scl/binding/c_api/gram/gram.h
// BRIEF: C API for Gram matrix computation
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Gram Matrix Computation
// =============================================================================

scl_error_t scl_gram_compute(
    scl_sparse_t matrix,
    scl_real_t* output,
    scl_size_t n_rows
);

#ifdef __cplusplus
}
#endif
