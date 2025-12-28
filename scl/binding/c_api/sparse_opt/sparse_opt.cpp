// =============================================================================
// FILE: scl/binding/c_api/sparse_opt/sparse_opt.cpp
// BRIEF: C API implementation for sparse optimization
// =============================================================================

#include "scl/binding/c_api/sparse_opt/sparse_opt.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/sparse_opt.hpp"
#include "scl/core/type.hpp"

extern "C" {

scl_error_t scl_sparse_opt_prox_l1(
    scl_real_t* x,
    scl_size_t n,
    scl_real_t lambda
) {
    if (!x) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<scl::Real> arr(reinterpret_cast<scl::Real*>(x), n);
        scl::kernel::sparse_opt::prox_l1(arr, static_cast<scl::Real>(lambda));
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_sparse_opt_prox_elastic_net(
    scl_real_t* x,
    scl_size_t n,
    scl_real_t lambda,
    scl_real_t l1_ratio
) {
    if (!x) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<scl::Real> arr(reinterpret_cast<scl::Real*>(x), n);
        scl::kernel::sparse_opt::prox_elastic_net(
            arr, static_cast<scl::Real>(lambda), static_cast<scl::Real>(l1_ratio)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_sparse_opt_lasso_coordinate_descent(
    scl_sparse_t X,
    const scl_real_t* y,
    scl_real_t alpha,
    scl_real_t* coefficients,
    scl_index_t max_iter,
    scl_real_t tol
) {
    if (!X || !y || !coefficients) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(X);
        scl::Index n_samples = sparse->rows();
        scl::Index n_features = sparse->cols();

        scl::Array<const scl::Real> y_arr(reinterpret_cast<const scl::Real*>(y),
                                         static_cast<scl::Size>(n_samples));
        scl::Array<scl::Real> coef_arr(reinterpret_cast<scl::Real*>(coefficients),
                                       static_cast<scl::Size>(n_features));

        sparse->visit([&](auto& m) {
            scl::kernel::sparse_opt::lasso_coordinate_descent(
                m, y_arr, static_cast<scl::Real>(alpha), coef_arr,
                max_iter, static_cast<scl::Real>(tol)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_sparse_opt_elastic_net_coordinate_descent(
    scl_sparse_t X,
    const scl_real_t* y,
    scl_real_t alpha,
    scl_real_t l1_ratio,
    scl_real_t* coefficients,
    scl_index_t max_iter,
    scl_real_t tol
) {
    if (!X || !y || !coefficients) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(X);
        scl::Index n_samples = sparse->rows();
        scl::Index n_features = sparse->cols();

        scl::Array<const scl::Real> y_arr(reinterpret_cast<const scl::Real*>(y),
                                         static_cast<scl::Size>(n_samples));
        scl::Array<scl::Real> coef_arr(reinterpret_cast<scl::Real*>(coefficients),
                                      static_cast<scl::Size>(n_features));

        sparse->visit([&](auto& m) {
            scl::kernel::sparse_opt::elastic_net_coordinate_descent(
                m, y_arr, static_cast<scl::Real>(alpha), static_cast<scl::Real>(l1_ratio),
                coef_arr, max_iter, static_cast<scl::Real>(tol)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_sparse_opt_fista_lasso(
    scl_sparse_t X,
    const scl_real_t* y,
    scl_real_t alpha,
    scl_real_t* coefficients,
    scl_index_t max_iter,
    scl_real_t tol
) {
    if (!X || !y || !coefficients) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(X);
        scl::Index n_samples = sparse->rows();
        scl::Index n_features = sparse->cols();

        scl::Array<const scl::Real> y_arr(reinterpret_cast<const scl::Real*>(y),
                                         static_cast<scl::Size>(n_samples));
        scl::Array<scl::Real> coef_arr(reinterpret_cast<scl::Real*>(coefficients),
                                      static_cast<scl::Size>(n_features));

        sparse->visit([&](auto& m) {
            scl::kernel::sparse_opt::fista_lasso(
                m, y_arr, static_cast<scl::Real>(alpha), coef_arr,
                max_iter, static_cast<scl::Real>(tol)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_sparse_opt_iht(
    scl_sparse_t X,
    const scl_real_t* y,
    scl_index_t sparsity_level,
    scl_real_t* coefficients,
    scl_index_t max_iter
) {
    if (!X || !y || !coefficients) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(X);
        scl::Index n_samples = sparse->rows();
        scl::Index n_features = sparse->cols();

        scl::Array<const scl::Real> y_arr(reinterpret_cast<const scl::Real*>(y),
                                         static_cast<scl::Size>(n_samples));
        scl::Array<scl::Real> coef_arr(reinterpret_cast<scl::Real*>(coefficients),
                                      static_cast<scl::Size>(n_features));

        sparse->visit([&](auto& m) {
            scl::kernel::sparse_opt::iht(
                m, y_arr, sparsity_level, coef_arr, max_iter
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"

