#pragma once

// =============================================================================
// FILE: scl/binding/c_api/sparse_opt.h
// BRIEF: C API for sparse optimization methods
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "scl/binding/c_api/core_types.h"

// Lasso regression
scl_error_t scl_sparse_opt_lasso(
    scl_sparse_matrix_t X,
    const scl_real_t* y,
    scl_real_t* coefficients,
    scl_index_t n_samples,
    scl_index_t n_features,
    scl_real_t alpha,
    scl_index_t max_iter,
    scl_real_t tol
);

// Elastic net regression
scl_error_t scl_sparse_opt_elastic_net(
    scl_sparse_matrix_t X,
    const scl_real_t* y,
    scl_real_t* coefficients,
    scl_index_t n_samples,
    scl_index_t n_features,
    scl_real_t alpha,
    scl_real_t l1_ratio,
    scl_index_t max_iter,
    scl_real_t tol
);

// Logistic Lasso regression
scl_error_t scl_sparse_opt_logistic_lasso(
    scl_sparse_matrix_t X,
    const uint8_t* y_binary,
    scl_real_t* coefficients,
    scl_index_t n_samples,
    scl_index_t n_features,
    scl_real_t alpha,
    scl_index_t max_iter,
    scl_real_t tol
);

#ifdef __cplusplus
}
#endif
