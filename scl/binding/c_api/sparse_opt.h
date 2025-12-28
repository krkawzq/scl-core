#pragma once

// =============================================================================
// FILE: scl/binding/c_api/sparse_opt/sparse_opt.h
// BRIEF: C API for sparse optimization (Lasso, Elastic Net, etc.)
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Proximal Operators
// =============================================================================

scl_error_t scl_sparse_opt_prox_l1(
    scl_real_t* x,                        // [n] Input/output array
    scl_size_t n,
    scl_real_t lambda
);

scl_error_t scl_sparse_opt_prox_elastic_net(
    scl_real_t* x,                        // [n] Input/output array
    scl_size_t n,
    scl_real_t lambda,
    scl_real_t l1_ratio
);

// =============================================================================
// Lasso Regression
// =============================================================================

scl_error_t scl_sparse_opt_lasso_coordinate_descent(
    scl_sparse_t X,                        // Feature matrix [n_samples, n_features]
    const scl_real_t* y,                   // Target vector [n_samples]
    scl_real_t alpha,                      // Regularization strength
    scl_real_t* coefficients,              // Output [n_features]
    scl_index_t max_iter,
    scl_real_t tol
);

// =============================================================================
// Elastic Net Regression
// =============================================================================

scl_error_t scl_sparse_opt_elastic_net_coordinate_descent(
    scl_sparse_t X,                        // Feature matrix [n_samples, n_features]
    const scl_real_t* y,                   // Target vector [n_samples]
    scl_real_t alpha,                      // Regularization strength
    scl_real_t l1_ratio,                   // L1 ratio (0 = Ridge, 1 = Lasso)
    scl_real_t* coefficients,              // Output [n_features]
    scl_index_t max_iter,
    scl_real_t tol
);

// =============================================================================
// FISTA Algorithm for Lasso
// =============================================================================

scl_error_t scl_sparse_opt_fista_lasso(
    scl_sparse_t X,                        // Feature matrix [n_samples, n_features]
    const scl_real_t* y,                   // Target vector [n_samples]
    scl_real_t alpha,                      // Regularization strength
    scl_real_t* coefficients,              // Output [n_features]
    scl_index_t max_iter,
    scl_real_t tol
);

// =============================================================================
// Iterative Hard Thresholding
// =============================================================================

scl_error_t scl_sparse_opt_iht(
    scl_sparse_t X,                        // Feature matrix [n_samples, n_features]
    const scl_real_t* y,                   // Target vector [n_samples]
    scl_index_t sparsity_level,             // Desired sparsity level
    scl_real_t* coefficients,              // Output [n_features]
    scl_index_t max_iter
);

#ifdef __cplusplus
}
#endif
