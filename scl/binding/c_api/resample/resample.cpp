// =============================================================================
// FILE: scl/binding/c_api/resample/resample.cpp
// BRIEF: C API implementation for resampling
// =============================================================================

#include "scl/binding/c_api/resample/resample.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/resample.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::resample;

extern "C" {

scl_error_t scl_resample_downsample(
    scl_sparse_t matrix,
    scl_real_t target_sum,
    uint64_t seed)
{
    if (!matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(matrix);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        wrapper->visit([&](auto& m) {
            downsample(m, target_sum, seed);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_resample_downsample_variable(
    scl_sparse_t matrix,
    const scl_real_t* target_counts,
    scl_size_t primary_dim,
    uint64_t seed)
{
    if (!matrix || !target_counts) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(matrix);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<const Real> targets_arr(reinterpret_cast<const Real*>(target_counts), primary_dim);
        
        wrapper->visit([&](auto& m) {
            downsample_variable(m, targets_arr, seed);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_resample_binomial(
    scl_sparse_t matrix,
    scl_real_t p,
    uint64_t seed)
{
    if (!matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(matrix);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        wrapper->visit([&](auto& m) {
            binomial_resample(m, p, seed);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_resample_poisson(
    scl_sparse_t matrix,
    scl_real_t lambda,
    uint64_t seed)
{
    if (!matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(matrix);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        wrapper->visit([&](auto& m) {
            poisson_resample(m, lambda, seed);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

