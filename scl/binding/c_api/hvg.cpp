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

// =============================================================================
// Moments Computation
// =============================================================================

SCL_EXPORT scl_error_t scl_hvg_compute_moments(
    scl_sparse_t matrix,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    const int ddof) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out_means, "Output means array is null");
    SCL_C_API_CHECK_NULL(out_vars, "Output variances array is null");
    
    SCL_C_API_TRY
        const Index primary_dim = matrix->is_csr ? matrix->rows() : matrix->cols();
        
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
                m, out_means_arr, out_vars_arr, ddof
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_hvg_compute_clipped_moments(
    scl_sparse_t matrix,
    const scl_real_t* clip_vals,
    scl_real_t* out_means,
    scl_real_t* out_vars) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(clip_vals, "Clip values array is null");
    SCL_C_API_CHECK_NULL(out_means, "Output means array is null");
    SCL_C_API_CHECK_NULL(out_vars, "Output variances array is null");
    
    SCL_C_API_TRY
        const Index primary_dim = matrix->is_csr ? matrix->rows() : matrix->cols();
        
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
                m, clip_vals_arr, out_means_arr, out_vars_arr
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// HVG Selection by Dispersion
// =============================================================================

SCL_EXPORT scl_error_t scl_hvg_select_by_dispersion(
    scl_sparse_t matrix,
    const scl_size_t n_top,
    scl_index_t* out_indices,
    uint8_t* out_mask,
    scl_real_t* out_dispersions) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out_indices, "Output indices array is null");
    SCL_C_API_CHECK_NULL(out_mask, "Output mask array is null");
    SCL_C_API_CHECK_NULL(out_dispersions, "Output dispersions array is null");
    SCL_C_API_CHECK(n_top > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of top genes must be positive");
    
    SCL_C_API_TRY
        const Index primary_dim = matrix->is_csr ? matrix->rows() : matrix->cols();
        
        Array<Index> indices_arr(out_indices, n_top);
        Array<uint8_t> mask_arr(out_mask, static_cast<Size>(primary_dim));
        Array<Real> disps_arr(
            reinterpret_cast<Real*>(out_dispersions),
            static_cast<Size>(primary_dim)
        );
        
        matrix->visit([&](auto& m) {
            scl::kernel::hvg::select_by_dispersion(
                m, n_top, indices_arr, mask_arr, disps_arr
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// HVG Selection by VST
// =============================================================================

SCL_EXPORT scl_error_t scl_hvg_select_by_vst(
    scl_sparse_t matrix,
    const scl_real_t* clip_vals,
    const scl_size_t n_top,
    scl_index_t* out_indices,
    uint8_t* out_mask,
    scl_real_t* out_variances) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(clip_vals, "Clip values array is null");
    SCL_C_API_CHECK_NULL(out_indices, "Output indices array is null");
    SCL_C_API_CHECK_NULL(out_mask, "Output mask array is null");
    SCL_C_API_CHECK_NULL(out_variances, "Output variances array is null");
    SCL_C_API_CHECK(n_top > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of top genes must be positive");
    
    SCL_C_API_TRY
        const Index primary_dim = matrix->is_csr ? matrix->rows() : matrix->cols();
        
        Array<const Real> clip_vals_arr(
            reinterpret_cast<const Real*>(clip_vals),
            static_cast<Size>(primary_dim)
        );
        Array<Index> indices_arr(out_indices, n_top);
        Array<uint8_t> mask_arr(out_mask, static_cast<Size>(primary_dim));
        Array<Real> vars_arr(
            reinterpret_cast<Real*>(out_variances),
            static_cast<Size>(primary_dim)
        );
        
        matrix->visit([&](auto& m) {
            scl::kernel::hvg::select_by_vst(
                m, clip_vals_arr, n_top, indices_arr, mask_arr, vars_arr
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
