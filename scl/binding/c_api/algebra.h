#pragma once

// =============================================================================
// FILE: scl/binding/c_api/algebra.h
// BRIEF: C API for high-performance sparse linear algebra kernels
// =============================================================================
//
// OPERATIONS:
//   - SpMV: Sparse matrix-vector multiplication
//   - SpMM: Sparse matrix-dense matrix multiplication
//   - Row/column operations (norms, sums, scaling)
//   - Diagonal extraction
//
// PERFORMANCE:
//   - SIMD-accelerated kernels (AVX2/AVX-512)
//   - Tiered strategies for different sparsity patterns
//   - Automatic parallelization for large matrices
//   - Cache-aware tiling for SpMM
//
// THREAD SAFETY:
//   - All read-only operations are thread-safe
//   - In-place modifications (scale_rows) are NOT thread-safe
//   - Caller must synchronize concurrent modifications
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Sparse Matrix-Vector Multiplication (SpMV)
// =============================================================================

/// @brief General SpMV: y = alpha * A * x + beta * y
/// @param[in] A Sparse matrix (non-null, CSR or CSC)
/// @param[in] x Input vector [secondary_dim] (non-null)
/// @param[in] x_size Size of x (must be >= A.secondary_dim)
/// @param[in,out] y Output vector [primary_dim] (non-null)
/// @param[in] y_size Size of y (must be >= A.primary_dim)
/// @param[in] alpha Scaling factor for A*x
/// @param[in] beta Scaling factor for y
/// @return SCL_OK on success, error code otherwise
/// @note y is modified in-place
/// @note For CSR: primary=rows, secondary=cols
/// @note For CSC: primary=cols, secondary=rows
scl_error_t scl_algebra_spmv(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size,
    scl_real_t alpha,
    scl_real_t beta
);

/// @brief Simple SpMV: y = A * x
scl_error_t scl_algebra_spmv_simple(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size
);

/// @brief Scaled SpMV: y = alpha * A * x
scl_error_t scl_algebra_spmv_scaled(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size,
    scl_real_t alpha
);

/// @brief Additive SpMV: y += A * x
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

/// @brief Transposed SpMV: y = alpha * A^T * x + beta * y
/// @param[in] A Sparse matrix (non-null)
/// @param[in] x Input vector [primary_dim] (non-null)
/// @param[in] x_size Size of x
/// @param[in,out] y Output vector [secondary_dim] (non-null)
/// @param[in] y_size Size of y
/// @param[in] alpha Scaling factor
/// @param[in] beta Scaling factor
/// @return SCL_OK on success, error code otherwise
/// @note Uses atomic accumulation for thread safety
scl_error_t scl_algebra_spmv_transpose(
    scl_sparse_t A,
    const scl_real_t* x,
    scl_size_t x_size,
    scl_real_t* y,
    scl_size_t y_size,
    scl_real_t alpha,
    scl_real_t beta
);

/// @brief Simple transposed SpMV: y = A^T * x
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

/// @brief SpMM: Y = alpha * A * X + beta * Y
/// @param[in] A Sparse matrix (non-null)
/// @param[in] X Dense matrix [secondary_dim x n_cols], row-major (non-null)
/// @param[in] n_cols Number of columns in X and Y
/// @param[in,out] Y Dense matrix [primary_dim x n_cols], row-major (non-null)
/// @param[in] alpha Scaling factor for A*X
/// @param[in] beta Scaling factor for Y
/// @return SCL_OK on success, error code otherwise
/// @note Y is modified in-place
/// @note Uses cache-aware tiling for performance
scl_error_t scl_algebra_spmm(
    scl_sparse_t A,
    const scl_real_t* X,
    scl_index_t n_cols,
    scl_real_t* Y,
    scl_real_t alpha,
    scl_real_t beta
);

// =============================================================================
// Fused Operations
// =============================================================================

/// @brief Fused SpMV: y = alpha * A * x + beta * A * z + gamma * y
/// @param[in] A Sparse matrix (non-null)
/// @param[in] x First input vector (non-null)
/// @param[in] x_size Size of x
/// @param[in] z Second input vector (non-null)
/// @param[in] z_size Size of z
/// @param[in,out] y Output vector (non-null)
/// @param[in] y_size Size of y
/// @param[in] alpha Scaling for A*x
/// @param[in] beta Scaling for A*z
/// @param[in] gamma Scaling for y
/// @return SCL_OK on success, error code otherwise
/// @note Fuses two SpMV operations for better cache utilization
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

/// @brief Compute L2 norm of each row/column
/// @param[in] A Sparse matrix (non-null)
/// @param[out] norms Output array [primary_dim] (non-null)
/// @param[in] norms_size Size of norms array
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_algebra_row_norms(
    scl_sparse_t A,
    scl_real_t* norms,
    scl_size_t norms_size
);

/// @brief Compute sum of each row/column
/// @param[in] A Sparse matrix (non-null)
/// @param[out] sums Output array [primary_dim] (non-null)
/// @param[in] sums_size Size of sums array
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_algebra_row_sums(
    scl_sparse_t A,
    scl_real_t* sums,
    scl_size_t sums_size
);

/// @brief Extract diagonal elements
/// @param[in] A Sparse matrix (non-null)
/// @param[out] diag Output array [min(rows, cols)] (non-null)
/// @param[in] diag_size Size of diag array
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_algebra_extract_diagonal(
    scl_sparse_t A,
    scl_real_t* diag,
    scl_size_t diag_size
);

/// @brief Scale rows/columns in-place: A_i = scale_factors[i] * A_i
/// @param[in,out] A Sparse matrix (non-null, modified in-place)
/// @param[in] scale_factors Scaling factors [primary_dim] (non-null)
/// @param[in] scale_factors_size Size of scale_factors array
/// @return SCL_OK on success, error code otherwise
/// @note Modifies matrix in-place - NOT thread-safe
scl_error_t scl_algebra_scale_rows(
    scl_sparse_t A,
    const scl_real_t* scale_factors,
    scl_size_t scale_factors_size
);

#ifdef __cplusplus
}
#endif
