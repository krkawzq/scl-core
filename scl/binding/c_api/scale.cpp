// =============================================================================
// FILE: scl/binding/c_api/scale.cpp
// BRIEF: C API implementation for scaling operations
// =============================================================================

#include "scl/binding/c_api/scale.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/scale.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Standardize
// =============================================================================

SCL_EXPORT scl_error_t scl_scale_standardize(
    scl_sparse_t matrix,
    const scl_real_t* means,
    const scl_real_t* stds,
    const scl_size_t primary_dim,
    const scl_real_t max_value,
    const int zero_center) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(means, "Means array pointer is null");
    SCL_C_API_CHECK_NULL(stds, "Stds array pointer is null");
    SCL_C_API_CHECK(primary_dim > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Primary dimension must be positive");

    SCL_C_API_TRY
        const Size primary_dim_sz = static_cast<Size>(primary_dim);
        Array<const Real> means_arr(reinterpret_cast<const Real*>(means), primary_dim_sz);
        Array<const Real> stds_arr(reinterpret_cast<const Real*>(stds), primary_dim_sz);
        
        matrix->visit([&](auto& m) {
            scl::kernel::scale::standardize(
                m, means_arr, stds_arr, 
                static_cast<Real>(max_value), 
                zero_center != 0
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Scale Rows
// =============================================================================

SCL_EXPORT scl_error_t scl_scale_rows(
    scl_sparse_t matrix,
    const scl_real_t* scales,
    const scl_size_t primary_dim) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(scales, "Scales array pointer is null");
    SCL_C_API_CHECK(primary_dim > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Primary dimension must be positive");

    SCL_C_API_TRY
        const Size primary_dim_sz = static_cast<Size>(primary_dim);
        Array<const Real> scales_arr(reinterpret_cast<const Real*>(scales), primary_dim_sz);
        
        matrix->visit([&](auto& m) {
            scl::kernel::scale::scale_rows(m, scales_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Shift Rows
// =============================================================================

SCL_EXPORT scl_error_t scl_scale_shift_rows(
    scl_sparse_t matrix,
    const scl_real_t* offsets,
    const scl_size_t primary_dim) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(offsets, "Offsets array pointer is null");
    SCL_C_API_CHECK(primary_dim > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Primary dimension must be positive");

    SCL_C_API_TRY
        const Size primary_dim_sz = static_cast<Size>(primary_dim);
        Array<const Real> offsets_arr(reinterpret_cast<const Real*>(offsets), primary_dim_sz);
        
        matrix->visit([&](auto& m) {
            scl::kernel::scale::shift_rows(m, offsets_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
