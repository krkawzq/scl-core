#pragma once

// =============================================================================
// FILE: scl/binding/c_api/sparse_opt/sparse_opt.h
// BRIEF: C API for sparse optimization (Lasso, Elastic Net, etc.)
// =============================================================================

#include "scl/binding/c_api/core/core.h"

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

// =============================================================================
// Regularization Path
// =============================================================================

// Compute Lasso solution path for multiple alpha values
scl_error_t scl_sparse_opt_lasso_path(
    scl_sparse_t X,
    const scl_real_t* y,
    const scl_real_t* alphas,             // Array of alpha values [n_alphas]
    scl_size_t n_alphas,
    scl_real_t* coefficient_paths,        // Output [n_alphas x n_features]
    scl_index_t max_iter
);

// =============================================================================
// Regularization Types
// =============================================================================

typedef enum {
    SCL_REG_L1 = 0,
    SCL_REG_L2 = 1,
    SCL_REG_ELASTIC_NET = 2,
    SCL_REG_SCAD = 3,
    SCL_REG_MCP = 4
} scl_regularization_type_t;

// Proximal gradient with custom regularization type
scl_error_t scl_sparse_opt_proximal_gradient(
    scl_sparse_t X,
    const scl_real_t* y,
    scl_real_t alpha,
    scl_regularization_type_t reg_type,
    scl_real_t* coefficients,
    scl_index_t max_iter,
    scl_real_t tol
);

// =============================================================================
// Group Lasso
// =============================================================================

scl_error_t scl_sparse_opt_group_lasso(
    scl_sparse_t X,
    const scl_real_t* y,
    const scl_index_t* group_indices,     // Feature group assignments [n_features]
    const scl_index_t* group_offsets,     // Group boundaries [n_groups+1]
    scl_index_t n_groups,
    scl_real_t alpha,
    scl_real_t* coefficients,
    scl_index_t max_iter
);

// =============================================================================
// Sparse Logistic Regression
// =============================================================================

scl_error_t scl_sparse_opt_logistic_regression(
    scl_sparse_t X,
    const scl_index_t* y_binary,          // Binary labels {0, 1} [n_samples]
    scl_real_t alpha,
    scl_real_t* coefficients,
    scl_index_t max_iter
);

#ifdef __cplusplus
}
#endif
