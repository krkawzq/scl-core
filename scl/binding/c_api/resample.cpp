// =============================================================================
// FILE: scl/binding/c_api/resample.cpp
// BRIEF: C API implementation for resampling
// =============================================================================

#include "scl/binding/c_api/resample.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/resample.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Downsample
// =============================================================================

SCL_EXPORT scl_error_t scl_resample_downsample(
    scl_sparse_t matrix,
    scl_real_t target_sum,
    uint64_t seed)
{
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");

    SCL_C_API_TRY
        matrix->visit([&](auto& m) {
            scl::kernel::resample::downsample(m, static_cast<Real>(target_sum), seed);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Downsample Variable
// =============================================================================

SCL_EXPORT scl_error_t scl_resample_downsample_variable(
    scl_sparse_t matrix,
    const scl_real_t* target_counts,
    scl_size_t primary_dim,
    uint64_t seed)
{
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(target_counts, "Target counts pointer is null");

    SCL_C_API_TRY {
        const Size primary_dim_sz = static_cast<Size>(primary_dim);
        Array<const Real> targets_arr(reinterpret_cast<const Real*>(target_counts), primary_dim_sz);
        
        matrix->visit([&](auto& m) {
            scl::kernel::resample::downsample_variable(m, targets_arr, seed);
        });
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// Binomial Resample
// =============================================================================

SCL_EXPORT scl_error_t scl_resample_binomial(
    scl_sparse_t matrix,
    scl_real_t p,
    uint64_t seed)
{
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");

    SCL_C_API_TRY {
        matrix->visit([&](auto& m) {
            scl::kernel::resample::binomial_resample(m, static_cast<Real>(p), seed);
        });
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// Poisson Resample
// =============================================================================

SCL_EXPORT scl_error_t scl_resample_poisson(
    scl_sparse_t matrix,
    scl_real_t lambda,
    uint64_t seed)
{
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");

    SCL_C_API_TRY {
        matrix->visit([&](auto& m) {
            scl::kernel::resample::poisson_resample(m, static_cast<Real>(lambda), seed);
        });
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

} // extern "C"
