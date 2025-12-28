// =============================================================================
// FILE: scl/binding/c_api/softmax/softmax.cpp
// BRIEF: C API implementation for softmax operations
// =============================================================================

#include "scl/binding/c_api/softmax/softmax.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/softmax.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::softmax;

extern "C" {

// =============================================================================
// Dense Array Softmax
// =============================================================================

scl_error_t scl_softmax_array(
    scl_real_t* values,
    scl_size_t n,
    scl_real_t temperature)
{
    if (!values) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        Real* vals = reinterpret_cast<Real*>(values);
        
        if (temperature == 1.0f || temperature == 1.0) {
            softmax_inplace(vals, n);
        } else {
            softmax_inplace(vals, n, static_cast<Real>(temperature));
        }

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_log_softmax_array(
    scl_real_t* values,
    scl_size_t n,
    scl_real_t temperature)
{
    if (!values) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        Real* vals = reinterpret_cast<Real*>(values);
        
        if (temperature == 1.0f || temperature == 1.0) {
            log_softmax_inplace(vals, n);
        } else {
            log_softmax_inplace(vals, n, static_cast<Real>(temperature));
        }

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Sparse Matrix Softmax
// =============================================================================

scl_error_t scl_softmax_matrix(
    scl_sparse_t* matrix,
    scl_real_t temperature)
{
    if (!matrix || !*matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(*matrix);
        
        if (temperature == 1.0f || temperature == 1.0) {
            wrapper->visit([&](auto& m) {
                softmax_inplace(m);
            });
        } else {
            wrapper->visit([&](auto& m) {
                softmax_inplace(m, static_cast<Real>(temperature));
            });
        }

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_log_softmax_matrix(
    scl_sparse_t* matrix,
    scl_real_t temperature)
{
    if (!matrix || !*matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(*matrix);
        
        if (temperature == 1.0f || temperature == 1.0) {
            wrapper->visit([&](auto& m) {
                log_softmax_inplace(m);
            });
        } else {
            wrapper->visit([&](auto& m) {
                log_softmax_inplace(m, static_cast<Real>(temperature));
            });
        }

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

