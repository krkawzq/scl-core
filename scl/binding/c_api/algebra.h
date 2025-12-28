#pragma once

// =============================================================================
// FILE: scl/binding/c_api/algebra.h
// BRIEF: C API for sparse linear algebra operations
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Sparse Matrix-Vector Multiplication (SpMV)
// =============================================================================

// y = alpha * A * x + beta * y
scl_error_t scl_algebra_spmv(
    scl_sparse_t A,                    // Sparse matrix
    const scl_real_t* x,               // Input vector [secondary_dim]
    scl_size_t x_size,                 // Size of x
    scl_real_t* y,                     // Output vector [primary_dim] (in/out)
    scl_size_t y_size,                 // Size of y
    scl_real_t alpha,                  // Scaling factor for A*x
    scl_real_t beta                    // Scaling factor for y
);

// y = A * x (convenience wrapper)
scl_error_t scl_algebra_spmv_simple(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size
);

// y = alpha * A * x (scaled)
scl_error_t scl_algebra_spmv_scaled(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size,
    scl_real_t alpha
);

// y += A * x (additive)
scl_error_t scl_algebra_spmv_add(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size
);

// =============================================================================
// Transposed SpMV: y = alpha * A^T * x + beta * y
// =============================================================================

scl_error_t scl_algebra_spmv_transpose(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size,
    scl_real_t alpha,
    scl_real_t beta
);

// y = A^T * x (convenience wrapper)
scl_error_t scl_algebra_spmv_transpose_simple(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size
);

// =============================================================================
// Sparse Matrix-Dense Matrix Multiplication (SpMM)
// =============================================================================

// Y = alpha * A * X + beta * Y
// X: [secondary_dim x n_cols], row-major
// Y: [primary_dim x n_cols], row-major
scl_error_t scl_algebra_spmm(
    scl_sparse_t A,
    const scl_real_t* X,                // Dense matrix X [secondary_dim * n_cols]
    scl_index_t n_cols,                // Number of columns in X/Y
    scl_real_t* Y,                     // Dense matrix Y [primary_dim * n_cols] (in/out)
    scl_real_t alpha,
    scl_real_t beta
);

// =============================================================================
// Fused SpMV: y = alpha * A * x + beta * A * z + gamma * y
// =============================================================================

scl_error_t scl_algebra_spmv_fused_linear(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    const scl_real_t* z,
    scl_size_t z_size,
    scl_real_t* y,
    scl_size_t y_size,
    scl_real_t alpha,
    scl_real_t beta,
    scl_real_t gamma
);

// =============================================================================
// Row Operations
// =============================================================================

// Compute L2 norm of each row
scl_error_t scl_algebra_row_norms(
    scl_sparse_t A,
    scl_real_t* norms,                 // Output [primary_dim]
    scl_size_t norms_size
);

// Compute sum of each row
scl_error_t scl_algebra_row_sums(
    scl_sparse_t A,
    scl_real_t* sums,                  // Output [primary_dim]
    scl_size_t sums_size
);

// Extract diagonal elements
scl_error_t scl_algebra_extract_diagonal(
    scl_sparse_t A,
    scl_real_t* diag,                  // Output [min(rows, cols)]
    scl_size_t diag_size
);

// Scale rows: A_i = scale_factors[i] * A_i
scl_error_t scl_algebra_scale_rows(
    scl_sparse_t A,                    // Modified in-place
    const scl_real_t* scale_factors,   // [primary_dim]
    scl_size_t scale_factors_size
);

#ifdef __cplusplus
}
#endif
