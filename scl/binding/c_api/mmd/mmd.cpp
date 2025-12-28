// =============================================================================
// FILE: scl/binding/c_api/mmd/mmd.cpp
// BRIEF: C API implementation for MMD
// =============================================================================

#include "scl/binding/c_api/mmd/mmd.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/mmd.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::mmd;

extern "C" {

scl_error_t scl_mmd_rbf(
    scl_sparse_t mat_x,
    scl_sparse_t mat_y,
    scl_real_t* output,
    scl_real_t gamma)
{
    if (!mat_x || !mat_y || !output) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper_x = static_cast<scl_sparse_matrix*>(mat_x);
        auto* wrapper_y = static_cast<scl_sparse_matrix*>(mat_y);
        
        if (!wrapper_x->valid() || !wrapper_y->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        const Index n = wrapper_x->rows();
        Array<Real> output_arr(reinterpret_cast<Real*>(output), static_cast<Size>(n));
        
        wrapper_x->visit([&](auto& mx) {
            wrapper_y->visit([&](auto& my) {
                mmd_rbf(mx, my, output_arr, gamma);
            });
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

