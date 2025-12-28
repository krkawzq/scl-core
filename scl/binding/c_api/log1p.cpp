// =============================================================================
// FILE: scl/binding/c_api/log1p/log1p.cpp
// BRIEF: C API implementation for logarithmic transforms
// =============================================================================

#include "scl/binding/c_api/log1p.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/log1p.hpp"
#include "scl/core/sparse.hpp"

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::log1p;

extern "C" {

scl_error_t scl_log1p_inplace(scl_sparse_t* matrix)
{
    if (!matrix || !*matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(*matrix);
        wrapper->visit([&](auto& m) {
            log1p_inplace(m);
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_log2p1_inplace(scl_sparse_t* matrix)
{
    if (!matrix || !*matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(*matrix);
        wrapper->visit([&](auto& m) {
            log2p1_inplace(m);
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_expm1_inplace(scl_sparse_t* matrix)
{
    if (!matrix || !*matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(*matrix);
        wrapper->visit([&](auto& m) {
            expm1_inplace(m);
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

