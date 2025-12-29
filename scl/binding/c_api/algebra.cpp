// =============================================================================
// FILE: scl/binding/c_api/algebra.cpp
// BRIEF: C API implementation for sparse linear algebra kernels
// =============================================================================

#include "scl/binding/c_api/algebra.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/algebra.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// SpMV: y = alpha * A * x + beta * y
// =============================================================================

SCL_EXPORT scl_error_t scl_algebra_spmv(
    scl_sparse_t A,
    const scl_real_t* x,
    const scl_size_t x_size,
    scl_real_t* y,
    const scl_size_t y_size,
    const scl_real_t alpha,
    const scl_real_t beta) {
    
    SCL_C_API_CHECK_NULL(A, "Matrix A is null");
    SCL_C_API_CHECK_NULL(x, "Input vector x is null");
    SCL_C_API_CHECK_NULL(y, "Output vector y is null");
    
    SCL_C_API_TRY
        A->visit([&](auto& matrix) {
            const Array<const Real> x_arr(
                reinterpret_cast<const Real*>(x),
                x_size
            );
            Array<Real> y_arr(
                reinterpret_cast<Real*>(y),
                y_size
            );
            scl::kernel::algebra::spmv(
                matrix, x_arr, y_arr,
                static_cast<Real>(alpha),
                static_cast<Real>(beta)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_algebra_spmv_simple(
    scl_sparse_t A,
    const scl_real_t* x,
    const scl_size_t x_size,
    scl_real_t* y,
    const scl_size_t y_size) {
    
    return scl_algebra_spmv(A, x, x_size, y, y_size, 
                           static_cast<scl_real_t>(1), 
                           static_cast<scl_real_t>(0));
}

SCL_EXPORT scl_error_t scl_algebra_spmv_scaled(
    scl_sparse_t A,
    const scl_real_t* x,
    const scl_size_t x_size,
    scl_real_t* y,
    const scl_size_t y_size,
    const scl_real_t alpha) {
    
    return scl_algebra_spmv(A, x, x_size, y, y_size, alpha, 
                           static_cast<scl_real_t>(0));
}

SCL_EXPORT scl_error_t scl_algebra_spmv_add(
    scl_sparse_t A,
    const scl_real_t* x,
    const scl_size_t x_size,
    scl_real_t* y,
    const scl_size_t y_size) {
    
    return scl_algebra_spmv(A, x, x_size, y, y_size, 
                           static_cast<scl_real_t>(1), 
                           static_cast<scl_real_t>(1));
}

// =============================================================================
// Transposed SpMV
// =============================================================================

SCL_EXPORT scl_error_t scl_algebra_spmv_transpose(
    scl_sparse_t A,
    const scl_real_t* x,
    const scl_size_t x_size,
    scl_real_t* y,
    const scl_size_t y_size,
    const scl_real_t alpha,
    const scl_real_t beta) {
    
    SCL_C_API_CHECK_NULL(A, "Matrix A is null");
    SCL_C_API_CHECK_NULL(x, "Input vector x is null");
    SCL_C_API_CHECK_NULL(y, "Output vector y is null");
    
    SCL_C_API_TRY
        A->visit([&](auto& matrix) {
            const Array<const Real> x_arr(
                reinterpret_cast<const Real*>(x),
                x_size
            );
            Array<Real> y_arr(
                reinterpret_cast<Real*>(y),
                y_size
            );
            scl::kernel::algebra::spmv_transpose(
                matrix, x_arr, y_arr,
                static_cast<Real>(alpha),
                static_cast<Real>(beta)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_algebra_spmv_transpose_simple(
    scl_sparse_t A,
    const scl_real_t* x,
    const scl_size_t x_size,
    scl_real_t* y,
    const scl_size_t y_size) {
    
    return scl_algebra_spmv_transpose(A, x, x_size, y, y_size, 
                                     static_cast<scl_real_t>(1), 
                                     static_cast<scl_real_t>(0));
}

// =============================================================================
// SpMM: Y = alpha * A * X + beta * Y
// =============================================================================

SCL_EXPORT scl_error_t scl_algebra_spmm(
    scl_sparse_t A,
    const scl_real_t* X,
    const scl_index_t n_cols,
    scl_real_t* Y,
    const scl_real_t alpha,
    const scl_real_t beta) {
    
    SCL_C_API_CHECK_NULL(A, "Matrix A is null");
    SCL_C_API_CHECK_NULL(X, "Input matrix X is null");
    SCL_C_API_CHECK_NULL(Y, "Output matrix Y is null");
    SCL_C_API_CHECK(n_cols > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of columns must be positive");
    
    SCL_C_API_TRY
        A->visit([&](auto& matrix) {
            scl::kernel::algebra::spmm(
                matrix,
                reinterpret_cast<const Real*>(X),
                static_cast<Index>(n_cols),
                reinterpret_cast<Real*>(Y),
                static_cast<Real>(alpha),
                static_cast<Real>(beta)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Fused Operations
// =============================================================================

SCL_EXPORT scl_error_t scl_algebra_spmv_fused_linear(
    scl_sparse_t A,
    const scl_real_t* x,
    const scl_size_t x_size,
    const scl_real_t* z,
    const scl_size_t z_size,
    scl_real_t* y,
    const scl_size_t y_size,
    const scl_real_t alpha,
    const scl_real_t beta,
    const scl_real_t gamma) {
    
    SCL_C_API_CHECK_NULL(A, "Matrix A is null");
    SCL_C_API_CHECK_NULL(x, "Input vector x is null");
    SCL_C_API_CHECK_NULL(z, "Input vector z is null");
    SCL_C_API_CHECK_NULL(y, "Output vector y is null");
    
    SCL_C_API_TRY
        A->visit([&](auto& matrix) {
            const Array<const Real> x_arr(
                reinterpret_cast<const Real*>(x),
                x_size
            );
            const Array<const Real> z_arr(
                reinterpret_cast<const Real*>(z),
                z_size
            );
            Array<Real> y_arr(
                reinterpret_cast<Real*>(y),
                y_size
            );
            scl::kernel::algebra::spmv_fused_linear(
                matrix, x_arr, z_arr, y_arr,
                static_cast<Real>(alpha),
                static_cast<Real>(beta),
                static_cast<Real>(gamma)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Row Operations
// =============================================================================

SCL_EXPORT scl_error_t scl_algebra_row_norms(
    scl_sparse_t A,
    scl_real_t* norms,
    const scl_size_t norms_size) {
    
    SCL_C_API_CHECK_NULL(A, "Matrix A is null");
    SCL_C_API_CHECK_NULL(norms, "Output norms array is null");
    
    SCL_C_API_TRY
        A->visit([&](auto& matrix) {
            Array<Real> norms_arr(
                reinterpret_cast<Real*>(norms),
                norms_size
            );
            scl::kernel::algebra::row_norms(matrix, norms_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_algebra_row_sums(
    scl_sparse_t A,
    scl_real_t* sums,
    const scl_size_t sums_size) {

    SCL_C_API_CHECK_NULL(A, "Matrix A is null");
    SCL_C_API_CHECK_NULL(sums, "Output sums array is null");

    SCL_C_API_TRY
        Array<Real> sums_arr(reinterpret_cast<Real*>(sums), sums_size);

        A->visit([&](auto& matrix) {
            scl::kernel::algebra::row_sums(matrix, sums_arr);
        });

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_algebra_extract_diagonal(
    scl_sparse_t A,
    scl_real_t* diag,
    const scl_size_t diag_size) {
    
    SCL_C_API_CHECK_NULL(A, "Matrix A is null");
    SCL_C_API_CHECK_NULL(diag, "Output diagonal array is null");
    
    SCL_C_API_TRY
        A->visit([&](auto& matrix) {
            Array<Real> diag_arr(
                reinterpret_cast<Real*>(diag),
                diag_size
            );
            scl::kernel::algebra::extract_diagonal(matrix, diag_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_algebra_scale_rows(
    scl_sparse_t A,
    const scl_real_t* scale_factors,
    const scl_size_t scale_factors_size) {
    
    SCL_C_API_CHECK_NULL(A, "Matrix A is null");
    SCL_C_API_CHECK_NULL(scale_factors, "Scale factors array is null");
    
    SCL_C_API_TRY
        A->visit([&](auto& matrix) {
            const Array<const Real> scale_arr(
                reinterpret_cast<const Real*>(scale_factors),
                scale_factors_size
            );
            scl::kernel::algebra::scale_rows(matrix, scale_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
