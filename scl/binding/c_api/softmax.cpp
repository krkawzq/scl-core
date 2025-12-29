// =============================================================================
// FILE: scl/binding/c_api/softmax/softmax.cpp
// BRIEF: C API implementation for softmax operations
// =============================================================================

#include "scl/binding/c_api/softmax.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/softmax.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Dense Array Softmax
// =============================================================================

SCL_EXPORT scl_error_t scl_softmax_array(
    scl_real_t* values,
    const scl_size_t n,
    const scl_real_t temperature) {
    
    SCL_C_API_CHECK_NULL(values, "Values array is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");

    SCL_C_API_TRY
        Real* vals = reinterpret_cast<Real*>(values);
        
        if (temperature == static_cast<scl_real_t>(1.0)) {
            scl::kernel::softmax::softmax_inplace(vals, n);
        } else {
            scl::kernel::softmax::softmax_inplace(vals, n, static_cast<Real>(temperature));
        }
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_log_softmax_array(
    scl_real_t* values,
    const scl_size_t n,
    const scl_real_t temperature) {
    
    SCL_C_API_CHECK_NULL(values, "Values array is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");

    SCL_C_API_TRY
        Real* vals = reinterpret_cast<Real*>(values);
        
        if (temperature == static_cast<scl_real_t>(1.0)) {
            scl::kernel::softmax::log_softmax_inplace(vals, n);
        } else {
            scl::kernel::softmax::log_softmax_inplace(vals, n, static_cast<Real>(temperature));
        }
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Sparse Matrix Softmax
// =============================================================================

SCL_EXPORT scl_error_t scl_softmax_matrix(
    scl_sparse_t* matrix,
    const scl_real_t temperature) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix pointer is null");
    SCL_C_API_CHECK_NULL(*matrix, "Matrix handle is null");

    SCL_C_API_TRY
        if (temperature == static_cast<scl_real_t>(1.0)) {
            (*matrix)->visit([&](auto& m) {
                scl::kernel::softmax::softmax_inplace(m);
            });
        } else {
            (*matrix)->visit([&](auto& m) {
                scl::kernel::softmax::softmax_inplace(m, static_cast<Real>(temperature));
            });
        }
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_log_softmax_matrix(
    scl_sparse_t* matrix,
    const scl_real_t temperature) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix pointer is null");
    SCL_C_API_CHECK_NULL(*matrix, "Matrix handle is null");

    SCL_C_API_TRY
        if (temperature == static_cast<scl_real_t>(1.0)) {
            (*matrix)->visit([&](auto& m) {
                scl::kernel::softmax::log_softmax_inplace(m);
            });
        } else {
            (*matrix)->visit([&](auto& m) {
                scl::kernel::softmax::log_softmax_inplace(m, static_cast<Real>(temperature));
            });
        }
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
