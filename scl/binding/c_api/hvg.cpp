// =============================================================================
// FILE: scl/binding/c_api/hvg/hvg.cpp
// BRIEF: C API implementation for highly variable gene selection
// =============================================================================

#include "scl/binding/c_api/hvg.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/hvg.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

scl_error_t scl_hvg_compute_moments(
    scl_sparse_t matrix,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    int ddof)
{
    if (!matrix || !out_means || !out_vars) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Index primary_dim = matrix->is_csr ? matrix->rows() : matrix->cols();
        
        Array<Real> out_means_arr(
            reinterpret_cast<Real*>(out_means),
            static_cast<Size>(primary_dim)
        );
        Array<Real> out_vars_arr(
            reinterpret_cast<Real*>(out_vars),
            static_cast<Size>(primary_dim)
        );
        
        matrix->visit([&](auto& m) {
            scl::kernel::hvg::detail::compute_moments(
                m,
                out_means_arr,
                out_vars_arr,
                ddof
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_hvg_compute_clipped_moments(
    scl_sparse_t matrix,
    const scl_real_t* clip_vals,
    scl_real_t* out_means,
    scl_real_t* out_vars)
{
    if (!matrix || !clip_vals || !out_means || !out_vars) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Index primary_dim = matrix->is_csr ? matrix->rows() : matrix->cols();
        
        Array<const Real> clip_vals_arr(
            reinterpret_cast<const Real*>(clip_vals),
            static_cast<Size>(primary_dim)
        );
        Array<Real> out_means_arr(
            reinterpret_cast<Real*>(out_means),
            static_cast<Size>(primary_dim)
        );
        Array<Real> out_vars_arr(
            reinterpret_cast<Real*>(out_vars),
            static_cast<Size>(primary_dim)
        );
        
        matrix->visit([&](auto& m) {
            scl::kernel::hvg::detail::compute_clipped_moments(
                m,
                clip_vals_arr,
                out_means_arr,
                out_vars_arr
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_hvg_select_by_dispersion(
    scl_sparse_t matrix,
    scl_size_t n_top,
    scl_index_t* out_indices,
    uint8_t* out_mask,
    scl_real_t* out_dispersions)
{
    if (!matrix || !out_indices || !out_mask || !out_dispersions) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    if (n_top == 0) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "n_top must be > 0");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Index primary_dim = matrix->is_csr ? matrix->rows() : matrix->cols();
        
        if (static_cast<scl_size_t>(primary_dim) < n_top) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "n_top exceeds primary dimension");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<Index> out_indices_arr(
            reinterpret_cast<Index*>(out_indices),
            static_cast<Size>(n_top)
        );
        Array<uint8_t> out_mask_arr(
            out_mask,
            static_cast<Size>(primary_dim)
        );
        Array<Real> out_dispersions_arr(
            reinterpret_cast<Real*>(out_dispersions),
            static_cast<Size>(primary_dim)
        );
        
        matrix->visit([&](auto& m) {
            scl::kernel::hvg::select_by_dispersion(
                m,
                static_cast<Size>(n_top),
                out_indices_arr,
                out_mask_arr,
                out_dispersions_arr
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_hvg_select_by_vst(
    scl_sparse_t matrix,
    const scl_real_t* clip_vals,
    scl_size_t n_top,
    scl_index_t* out_indices,
    uint8_t* out_mask,
    scl_real_t* out_variances)
{
    if (!matrix || !clip_vals || !out_indices || !out_mask || !out_variances) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    if (n_top == 0) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "n_top must be > 0");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Index primary_dim = matrix->is_csr ? matrix->rows() : matrix->cols();
        
        if (static_cast<scl_size_t>(primary_dim) < n_top) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "n_top exceeds primary dimension");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<const Real> clip_vals_arr(
            reinterpret_cast<const Real*>(clip_vals),
            static_cast<Size>(primary_dim)
        );
        Array<Index> out_indices_arr(
            reinterpret_cast<Index*>(out_indices),
            static_cast<Size>(n_top)
        );
        Array<uint8_t> out_mask_arr(
            out_mask,
            static_cast<Size>(primary_dim)
        );
        Array<Real> out_variances_arr(
            reinterpret_cast<Real*>(out_variances),
            static_cast<Size>(primary_dim)
        );
        
        matrix->visit([&](auto& m) {
            scl::kernel::hvg::select_by_vst(
                m,
                clip_vals_arr,
                static_cast<Size>(n_top),
                out_indices_arr,
                out_mask_arr,
                out_variances_arr
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

