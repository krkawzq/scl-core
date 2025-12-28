// =============================================================================
// FILE: scl/binding/c_api/sparse_opt/sparse_opt.cpp
// BRIEF: C API implementation for sparse optimization
// =============================================================================

#include "scl/binding/c_api/sparse_opt.h"
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

// =============================================================================
// Regularization Path
// =============================================================================

scl_error_t scl_sparse_opt_lasso_path(
    scl_sparse_t X,
    const scl_real_t* y,
    const scl_real_t* alphas,
    scl_size_t n_alphas,
    scl_real_t* coefficient_paths,
    scl_index_t max_iter)
{
    if (!X || !y || !alphas || !coefficient_paths) {
        scl::binding::set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<scl::binding::SparseWrapper*>(X);
        
        scl::Index n_samples = wrapper->rows();
        scl::Index n_features = wrapper->cols();
        
        scl::Array<const scl::Real> y_arr(
            reinterpret_cast<const scl::Real*>(y),
            static_cast<scl::Size>(n_samples)
        );
        scl::Array<const scl::Real> alphas_arr(
            reinterpret_cast<const scl::Real*>(alphas),
            n_alphas
        );

        wrapper->visit([&](const auto& m) {
            scl::kernel::sparse_opt::lasso_path(
                m, y_arr, alphas_arr, 
                reinterpret_cast<scl::Real*>(coefficient_paths),
                max_iter
            );
        });

        scl::binding::clear_last_error();
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// Regularization Types
// =============================================================================

namespace {

scl::kernel::sparse_opt::RegularizationType convert_regularization_type(
    scl_regularization_type_t type)
{
    switch (type) {
        case SCL_REG_L1:
            return scl::kernel::sparse_opt::RegularizationType::L1;
        case SCL_REG_L2:
            return scl::kernel::sparse_opt::RegularizationType::L2;
        case SCL_REG_ELASTIC_NET:
            return scl::kernel::sparse_opt::RegularizationType::ELASTIC_NET;
        case SCL_REG_SCAD:
            return scl::kernel::sparse_opt::RegularizationType::SCAD;
        case SCL_REG_MCP:
            return scl::kernel::sparse_opt::RegularizationType::MCP;
        default:
            return scl::kernel::sparse_opt::RegularizationType::L1;
    }
}

} // anonymous namespace

scl_error_t scl_sparse_opt_proximal_gradient(
    scl_sparse_t X,
    const scl_real_t* y,
    scl_real_t alpha,
    scl_regularization_type_t reg_type,
    scl_real_t* coefficients,
    scl_index_t max_iter,
    scl_real_t tol)
{
    if (!X || !y || !coefficients) {
        scl::binding::set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<scl::binding::SparseWrapper*>(X);
        
        scl::Index n_samples = wrapper->rows();
        scl::Index n_features = wrapper->cols();
        
        scl::Array<const scl::Real> y_arr(
            reinterpret_cast<const scl::Real*>(y),
            static_cast<scl::Size>(n_samples)
        );
        scl::Array<scl::Real> coef_arr(
            reinterpret_cast<scl::Real*>(coefficients),
            static_cast<scl::Size>(n_features)
        );

        auto reg_type_cpp = convert_regularization_type(reg_type);

        wrapper->visit([&](const auto& m) {
            scl::kernel::sparse_opt::proximal_gradient(
                m, y_arr, static_cast<scl::Real>(alpha),
                reg_type_cpp, coef_arr, max_iter,
                static_cast<scl::Real>(tol)
            );
        });

        scl::binding::clear_last_error();
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// Group Lasso
// =============================================================================

scl_error_t scl_sparse_opt_group_lasso(
    scl_sparse_t X,
    const scl_real_t* y,
    const scl_index_t* group_indices,
    const scl_index_t* group_offsets,
    scl_index_t n_groups,
    scl_real_t alpha,
    scl_real_t* coefficients,
    scl_index_t max_iter)
{
    if (!X || !y || !group_indices || !group_offsets || !coefficients) {
        scl::binding::set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<scl::binding::SparseWrapper*>(X);
        
        scl::Index n_samples = wrapper->rows();
        scl::Index n_features = wrapper->cols();
        
        scl::Array<const scl::Real> y_arr(
            reinterpret_cast<const scl::Real*>(y),
            static_cast<scl::Size>(n_samples)
        );
        scl::Array<scl::Real> coef_arr(
            reinterpret_cast<scl::Real*>(coefficients),
            static_cast<scl::Size>(n_features)
        );

        wrapper->visit([&](const auto& m) {
            scl::kernel::sparse_opt::group_lasso(
                m, y_arr,
                reinterpret_cast<const scl::Index*>(group_indices),
                reinterpret_cast<const scl::Index*>(group_offsets),
                n_groups,
                static_cast<scl::Real>(alpha),
                coef_arr,
                max_iter
            );
        });

        scl::binding::clear_last_error();
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// Sparse Logistic Regression
// =============================================================================

scl_error_t scl_sparse_opt_logistic_regression(
    scl_sparse_t X,
    const scl_index_t* y_binary,
    scl_real_t alpha,
    scl_real_t* coefficients,
    scl_index_t max_iter)
{
    if (!X || !y_binary || !coefficients) {
        scl::binding::set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<scl::binding::SparseWrapper*>(X);
        
        scl::Index n_samples = wrapper->rows();
        scl::Index n_features = wrapper->cols();
        
        scl::Array<const scl::Index> y_arr(
            reinterpret_cast<const scl::Index*>(y_binary),
            static_cast<scl::Size>(n_samples)
        );
        scl::Array<scl::Real> coef_arr(
            reinterpret_cast<scl::Real*>(coefficients),
            static_cast<scl::Size>(n_features)
        );

        wrapper->visit([&](const auto& m) {
            scl::kernel::sparse_opt::sparse_logistic_regression(
                m, y_arr, static_cast<scl::Real>(alpha),
                coef_arr, max_iter
            );
        });

        scl::binding::clear_last_error();
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"

