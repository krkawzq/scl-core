// =============================================================================
// FILE: scl/binding/c_api/feature/feature.cpp
// BRIEF: C API implementation for feature statistics
// =============================================================================

#include "scl/binding/c_api/feature.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/feature.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Standard Moments
// =============================================================================

SCL_EXPORT scl_error_t scl_feature_standard_moments(
    scl_sparse_t matrix,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    const scl_size_t n_features,
    const int ddof) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out_means, "Output means array is null");
    SCL_C_API_CHECK_NULL(out_vars, "Output variances array is null");
    SCL_C_API_CHECK(n_features > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of features must be positive");
    
    SCL_C_API_TRY
        const Index primary_dim = matrix->rows();
        SCL_C_API_CHECK(static_cast<scl_size_t>(primary_dim) == n_features,
                       SCL_ERROR_DIMENSION_MISMATCH, "Feature count mismatch");
        
        Array<Real> means_arr(
            reinterpret_cast<Real*>(out_means),
            n_features
        );
        Array<Real> vars_arr(
            reinterpret_cast<Real*>(out_vars),
            n_features
        );
        
        matrix->visit([&](auto& m) {
            scl::kernel::feature::standard_moments(m, means_arr, vars_arr, ddof);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Clipped Moments
// =============================================================================

SCL_EXPORT scl_error_t scl_feature_clipped_moments(
    scl_sparse_t matrix,
    const scl_real_t* clip_vals,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    const scl_size_t n_features) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(clip_vals, "Clip values array is null");
    SCL_C_API_CHECK_NULL(out_means, "Output means array is null");
    SCL_C_API_CHECK_NULL(out_vars, "Output variances array is null");
    SCL_C_API_CHECK(n_features > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of features must be positive");
    
    SCL_C_API_TRY
        const Index primary_dim = matrix->rows();
        SCL_C_API_CHECK(static_cast<scl_size_t>(primary_dim) == n_features,
                       SCL_ERROR_DIMENSION_MISMATCH, "Feature count mismatch");
        
        Array<const Real> clip_arr(
            reinterpret_cast<const Real*>(clip_vals),
            n_features
        );
        Array<Real> means_arr(
            reinterpret_cast<Real*>(out_means),
            n_features
        );
        Array<Real> vars_arr(
            reinterpret_cast<Real*>(out_vars),
            n_features
        );
        
        matrix->visit([&](auto& m) {
            scl::kernel::feature::clipped_moments(m, clip_arr, means_arr, vars_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Detection Rate
// =============================================================================

SCL_EXPORT scl_error_t scl_feature_detection_rate(
    scl_sparse_t matrix,
    scl_real_t* out_rates,
    const scl_size_t n_features) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(out_rates, "Output rates array is null");
    SCL_C_API_CHECK(n_features > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of features must be positive");
    
    SCL_C_API_TRY
        const Index primary_dim = matrix->rows();
        SCL_C_API_CHECK(static_cast<scl_size_t>(primary_dim) == n_features,
                       SCL_ERROR_DIMENSION_MISMATCH, "Feature count mismatch");
        
        Array<Real> rates_arr(
            reinterpret_cast<Real*>(out_rates),
            n_features
        );
        
        matrix->visit([&](auto& m) {
            scl::kernel::feature::detection_rate(m, rates_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Dispersion
// =============================================================================

SCL_EXPORT scl_error_t scl_feature_dispersion(
    const scl_real_t* means,
    const scl_real_t* vars,
    scl_real_t* out_dispersion,
    const scl_size_t n_features) {
    
    SCL_C_API_CHECK_NULL(means, "Means array is null");
    SCL_C_API_CHECK_NULL(vars, "Variances array is null");
    SCL_C_API_CHECK_NULL(out_dispersion, "Output dispersion array is null");
    SCL_C_API_CHECK(n_features > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of features must be positive");
    
    SCL_C_API_TRY
        Array<const Real> means_arr(
            reinterpret_cast<const Real*>(means),
            n_features
        );
        Array<const Real> vars_arr(
            reinterpret_cast<const Real*>(vars),
            n_features
        );
        Array<Real> dispersion_arr(
            reinterpret_cast<Real*>(out_dispersion),
            n_features
        );
        
        scl::kernel::feature::dispersion(means_arr, vars_arr, dispersion_arr);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
