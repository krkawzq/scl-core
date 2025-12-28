#pragma once

// =============================================================================
// FILE: scl/binding/c_api/algebra.h
// BRIEF: C API for sparse linear algebra operations
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "core.h"

// =============================================================================
// SpMV: y = alpha * A * x + beta * y
// =============================================================================

scl_error_t scl_spmv_f32_csr(
    scl_sparse_matrix_t A,        // CSR sparse matrix
    const scl_real_t* x,          // Input vector [n_cols]
    scl_size_t x_size,            // Size of x
    scl_real_t* y,                // Output vector [n_rows] (in/out)
    scl_size_t y_size,            // Size of y
    scl_real_t alpha,             // Scaling factor for A*x
    scl_real_t beta               // Scaling factor for y
);

scl_error_t scl_spmv_f64_csr(
    scl_sparse_matrix_t A,
    const double* x,
    scl_size_t x_size,
    double* y,
    scl_size_t y_size,
    double alpha,
    double beta
);

// =============================================================================
// SpMV Transpose: y = alpha * A^T * x + beta * y
// =============================================================================

scl_error_t scl_spmv_transpose_f32_csr(
    scl_sparse_matrix_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size,
    scl_real_t alpha,
    scl_real_t beta
);

scl_error_t scl_spmv_transpose_f64_csr(
    scl_sparse_matrix_t A,
    const double* x,
    scl_size_t x_size,
    double* y,
    scl_size_t y_size,
    double alpha,
    double beta
);

// =============================================================================
// SpMM: Y = alpha * A * X + beta * Y
// =============================================================================

scl_error_t scl_spmm_f32_csr(
    scl_sparse_matrix_t A,
    const scl_real_t* X,          // Input matrix [n_cols * n_cols_X], row-major
    scl_size_t n_cols_X,          // Number of columns in X
    scl_real_t* Y,                // Output matrix [n_rows * n_cols_X], row-major (in/out)
    scl_real_t alpha,
    scl_real_t beta
);

scl_error_t scl_spmm_f64_csr(
    scl_sparse_matrix_t A,
    const double* X,
    scl_size_t n_cols_X,
    double* Y,
    double alpha,
    double beta
);

// =============================================================================
// Fused SpMV: y = alpha * A * x + beta * A * z + gamma * y
// =============================================================================

scl_error_t scl_spmv_fused_linear_f32_csr(
    scl_sparse_matrix_t A,
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

scl_error_t scl_spmv_fused_linear_f64_csr(
    scl_sparse_matrix_t A,
    const double* x,
    scl_size_t x_size,
    const double* z,
    scl_size_t z_size,
    double* y,
    scl_size_t y_size,
    double alpha,
    double beta,
    double gamma
);

// =============================================================================
// Row Norms
// =============================================================================

scl_error_t scl_row_norms_f32_csr(
    scl_sparse_matrix_t A,
    scl_real_t* norms,            // Output [n_rows]
    scl_size_t norms_size
);

scl_error_t scl_row_norms_f64_csr(
    scl_sparse_matrix_t A,
    double* norms,
    scl_size_t norms_size
);

// =============================================================================
// Row Sums
// =============================================================================

scl_error_t scl_row_sums_f32_csr(
    scl_sparse_matrix_t A,
    scl_real_t* sums,             // Output [n_rows]
    scl_size_t sums_size
);

scl_error_t scl_row_sums_f64_csr(
    scl_sparse_matrix_t A,
    double* sums,
    scl_size_t sums_size
);

// =============================================================================
// Extract Diagonal
// =============================================================================

scl_error_t scl_extract_diagonal_f32_csr(
    scl_sparse_matrix_t A,
    scl_real_t* diag,             // Output [min(n_rows, n_cols)]
    scl_size_t diag_size
);

scl_error_t scl_extract_diagonal_f64_csr(
    scl_sparse_matrix_t A,
    double* diag,
    scl_size_t diag_size
);

// =============================================================================
// Scale Rows: A_i = scale_factors[i] * A_i
// =============================================================================

scl_error_t scl_scale_rows_f32_csr(
    scl_sparse_matrix_t A,        // Modified in-place
    const scl_real_t* scale_factors,  // [n_rows]
    scl_size_t scale_factors_size
);

scl_error_t scl_scale_rows_f64_csr(
    scl_sparse_matrix_t A,
    const double* scale_factors,
    scl_size_t scale_factors_size
);

#ifdef __cplusplus
}
#endif
