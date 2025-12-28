#pragma once

// =============================================================================
// FILE: scl/binding/c_api/merge.h
// BRIEF: C API for matrix merging operations
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Vertical Stack (vstack): Stack matrices vertically
// =============================================================================

scl_error_t scl_merge_vstack(
    scl_sparse_t matrix1,
    scl_sparse_t matrix2,
    scl_sparse_t* result                // Output: merged matrix
);

// =============================================================================
// Horizontal Stack (hstack): Stack matrices horizontally
// =============================================================================

scl_error_t scl_merge_hstack(
    scl_sparse_t matrix1,
    scl_sparse_t matrix2,
    scl_sparse_t* result                // Output: merged matrix
);

#ifdef __cplusplus
}
#endif
