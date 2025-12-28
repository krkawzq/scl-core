#pragma once

// =============================================================================
// WARNING: UNSAFE API - ABI UNSTABLE
// =============================================================================
//
// This header exposes internal struct layouts for advanced use cases.
// Using these types directly:
//   1. Bypasses registry-based memory management
//   2. May cause memory leaks or double-frees
//   3. ABI may change between library versions
//
// Only use if you understand the memory model and need zero-overhead access.
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Raw Sparse Matrix Layout (mirrors scl::Sparse internal layout)
// =============================================================================

typedef struct scl_sparse_raw {
    void** data_ptrs;           // Array of pointers to row/col data
    void** indices_ptrs;        // Array of pointers to row/col indices
    scl_index_t* lengths;       // Length of each row/col
    scl_index_t rows;
    scl_index_t cols;
    scl_index_t nnz;
    int is_csr;                 // 1 = CSR, 0 = CSC
    // NOTE: owns_data and is_view removed - lifecycle managed by registry
} scl_sparse_raw_t;

// =============================================================================
// Unsafe Conversion Functions
// =============================================================================

// Get raw struct from safe handle (does not transfer ownership)
// WARNING: Modifying the raw struct may corrupt registry state
scl_error_t scl_sparse_unsafe_get_raw(
    scl_sparse_t matrix,
    scl_sparse_raw_t* out
);

// Create safe handle from raw struct (transfers ownership to registry)
// WARNING: Caller must ensure raw struct was properly allocated
scl_error_t scl_sparse_unsafe_from_raw(
    const scl_sparse_raw_t* raw,
    scl_sparse_t* out
);

// =============================================================================
// Raw Dense Matrix Layout
// =============================================================================

typedef struct scl_dense_raw {
    scl_real_t* data;           // Row-major data pointer
    scl_index_t rows;
    scl_index_t cols;
    scl_index_t stride;         // Row stride (usually == cols)
    int owns_data;
} scl_dense_raw_t;

// Get raw struct from safe handle (does not transfer ownership)
// WARNING: Modifying the raw struct may corrupt registry state
scl_error_t scl_dense_unsafe_get_raw(
    scl_dense_t matrix,
    scl_dense_raw_t* out
);

// Create safe handle from raw struct (transfers ownership to registry)
// WARNING: Caller must ensure raw struct was properly allocated
scl_error_t scl_dense_unsafe_from_raw(
    const scl_dense_raw_t* raw,
    scl_dense_t* out
);

#ifdef __cplusplus
}
#endif
