// =============================================================================
// FILE: scl/binding/c_api/scale/scale.cpp
// BRIEF: C API implementation for scaling operations
// =============================================================================

#include "scl/binding/c_api/scale.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/scale.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::scale;

extern "C" {

scl_error_t scl_scale_standardize(
    scl_sparse_t matrix,
    const scl_real_t* means,
    const scl_real_t* stds,
    scl_size_t primary_dim,
    scl_real_t max_value,
    int zero_center)
{
    if (!matrix || !means || !stds) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(matrix);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<const Real> means_arr(reinterpret_cast<const Real*>(means), primary_dim);
        Array<const Real> stds_arr(reinterpret_cast<const Real*>(stds), primary_dim);
        
        wrapper->visit([&](auto& m) {
            standardize(m, means_arr, stds_arr, max_value, zero_center != 0);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_scale_rows(
    scl_sparse_t matrix,
    const scl_real_t* scales,
    scl_size_t primary_dim)
{
    if (!matrix || !scales) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(matrix);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<const Real> scales_arr(reinterpret_cast<const Real*>(scales), primary_dim);
        
        wrapper->visit([&](auto& m) {
            scale_rows(m, scales_arr);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_scale_shift_rows(
    scl_sparse_t matrix,
    const scl_real_t* offsets,
    scl_size_t primary_dim)
{
    if (!matrix || !offsets) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(matrix);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<const Real> offsets_arr(reinterpret_cast<const Real*>(offsets), primary_dim);
        
        wrapper->visit([&](auto& m) {
            shift_rows(m, offsets_arr);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

