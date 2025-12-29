// =============================================================================
// FILE: scl/binding/c_api/log1p.cpp
// BRIEF: C API implementation for logarithmic transforms
// =============================================================================

#include "scl/binding/c_api/log1p.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/log1p.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Log1p In-Place
// =============================================================================

SCL_EXPORT scl_error_t scl_log1p_inplace(scl_sparse_t* matrix) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle pointer is null");
    SCL_C_API_CHECK_NULL(*matrix, "Matrix handle is null");

    SCL_C_API_TRY
        (*matrix)->visit([&](auto& m) {
            scl::kernel::log1p::log1p_inplace(m);
        });

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Log2p1 In-Place
// =============================================================================

SCL_EXPORT scl_error_t scl_log2p1_inplace(scl_sparse_t* matrix) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle pointer is null");
    SCL_C_API_CHECK_NULL(*matrix, "Matrix handle is null");

    SCL_C_API_TRY
        (*matrix)->visit([&](auto& m) {
            scl::kernel::log1p::log2p1_inplace(m);
        });

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Expm1 In-Place
// =============================================================================

SCL_EXPORT scl_error_t scl_expm1_inplace(scl_sparse_t* matrix) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle pointer is null");
    SCL_C_API_CHECK_NULL(*matrix, "Matrix handle is null");

    SCL_C_API_TRY
        (*matrix)->visit([&](auto& m) {
            scl::kernel::log1p::expm1_inplace(m);
        });

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
