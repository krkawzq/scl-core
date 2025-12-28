// =============================================================================
// FILE: scl/binding/c_api/sparse_opt.cpp
// BRIEF: C API implementation for sparse optimization methods
// =============================================================================

#include "scl/binding/c_api/sparse_opt.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/sparse_opt.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

static scl_error_t handle_exception() {
    try {
        throw;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::TypeError&) {
        return SCL_ERROR_TYPE;
    } catch (const scl::IOError&) {
        return SCL_ERROR_IO;
    } catch (const scl::NotImplementedError&) {
        return SCL_ERROR_NOT_IMPLEMENTED;
    } catch (const scl::InternalError&) {
        return SCL_ERROR_INTERNAL;
    } catch (const scl::Exception& e) {
        return static_cast<scl_error_t>(static_cast<int>(e.code()));
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_sparse_opt_lasso(
    scl_sparse_matrix_t X,
    const scl_real_t* y,
    scl_real_t* coefficients,
    scl_index_t n_samples,
    scl_index_t n_features,
    scl_real_t alpha,
    scl_index_t max_iter,
    scl_real_t tol
) {
    if (!X || !y || !coefficients) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(X);
        scl::Array<const scl::Real> y_arr(
            reinterpret_cast<const scl::Real*>(y),
            static_cast<scl::Size>(n_samples)
        );
        scl::Array<scl::Real> coef_arr(
            reinterpret_cast<scl::Real*>(coefficients),
            static_cast<scl::Size>(n_features)
        );
        scl::kernel::sparse_opt::lasso(
            *sparse,
            y_arr,
            coef_arr,
            static_cast<scl::Real>(alpha),
            static_cast<scl::Index>(max_iter),
            static_cast<scl::Real>(tol)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

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
) {
    if (!X || !y || !coefficients) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(X);
        scl::Array<const scl::Real> y_arr(
            reinterpret_cast<const scl::Real*>(y),
            static_cast<scl::Size>(n_samples)
        );
        scl::Array<scl::Real> coef_arr(
            reinterpret_cast<scl::Real*>(coefficients),
            static_cast<scl::Size>(n_features)
        );
        scl::kernel::sparse_opt::elastic_net(
            *sparse,
            y_arr,
            coef_arr,
            static_cast<scl::Real>(alpha),
            static_cast<scl::Real>(l1_ratio),
            static_cast<scl::Index>(max_iter),
            static_cast<scl::Real>(tol)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_opt_logistic_lasso(
    scl_sparse_matrix_t X,
    const uint8_t* y_binary,
    scl_real_t* coefficients,
    scl_index_t n_samples,
    scl_index_t n_features,
    scl_real_t alpha,
    scl_index_t max_iter,
    scl_real_t tol
) {
    if (!X || !y_binary || !coefficients) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(X);
        scl::Array<const scl::Byte> y_arr(
            reinterpret_cast<const scl::Byte*>(y_binary),
            static_cast<scl::Size>(n_samples)
        );
        scl::Array<scl::Real> coef_arr(
            reinterpret_cast<scl::Real*>(coefficients),
            static_cast<scl::Size>(n_features)
        );
        scl::kernel::sparse_opt::logistic_lasso(
            *sparse,
            y_arr,
            coef_arr,
            static_cast<scl::Real>(alpha),
            static_cast<scl::Index>(max_iter),
            static_cast<scl::Real>(tol)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::sparse_opt::lasso<scl::Real, true>(
    const scl::CSR&, scl::Array<const scl::Real>, scl::Array<scl::Real>, scl::Real, scl::Index, scl::Real);
template void scl::kernel::sparse_opt::elastic_net<scl::Real, true>(
    const scl::CSR&, scl::Array<const scl::Real>, scl::Array<scl::Real>, scl::Real, scl::Real, scl::Index, scl::Real);
template void scl::kernel::sparse_opt::logistic_lasso<scl::Real, true>(
    const scl::CSR&, scl::Array<const scl::Byte>, scl::Array<scl::Real>, scl::Real, scl::Index, scl::Real);

} // extern "C"
