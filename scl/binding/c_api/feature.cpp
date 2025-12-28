// =============================================================================
// FILE: scl/binding/c_api/feature/feature.cpp
// BRIEF: C API implementation for feature statistics
// =============================================================================

#include "scl/binding/c_api/feature.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/feature.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"

#include <span>

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::feature;

extern "C" {

// =============================================================================
// Standard Moments
// =============================================================================

scl_error_t scl_feature_standard_moments(
    scl_sparse_t matrix,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    scl_size_t n_features,
    int ddof)
{
    if (!matrix || !out_means || !out_vars) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        Index primary_dim = wrapper->rows();
        if (static_cast<scl_size_t>(primary_dim) != n_features) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Feature count mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        Array<Real> means_arr(
            reinterpret_cast<Real*>(out_means),
            n_features
        );
        Array<Real> vars_arr(
            reinterpret_cast<Real*>(out_vars),
            n_features
        );

        wrapper->visit([&](auto& m) {
            standard_moments(m, means_arr, vars_arr, ddof);
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Clipped Moments
// =============================================================================

scl_error_t scl_feature_clipped_moments(
    scl_sparse_t matrix,
    const scl_real_t* clip_vals,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    scl_size_t n_features)
{
    if (!matrix || !clip_vals || !out_means || !out_vars) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        Index primary_dim = wrapper->rows();
        if (static_cast<scl_size_t>(primary_dim) != n_features) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Feature count mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

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

        wrapper->visit([&](auto& m) {
            clipped_moments(m, clip_arr, means_arr, vars_arr);
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Detection Rate
// =============================================================================

scl_error_t scl_feature_detection_rate(
    scl_sparse_t matrix,
    scl_real_t* out_rates,
    scl_size_t n_features)
{
    if (!matrix || !out_rates) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        Index primary_dim = wrapper->rows();
        if (static_cast<scl_size_t>(primary_dim) != n_features) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Feature count mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        Array<Real> rates_arr(
            reinterpret_cast<Real*>(out_rates),
            n_features
        );

        wrapper->visit([&](auto& m) {
            detection_rate(m, rates_arr);
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Dispersion
// =============================================================================

scl_error_t scl_feature_dispersion(
    const scl_real_t* means,
    const scl_real_t* vars,
    scl_real_t* out_dispersion,
    scl_size_t n_features)
{
    if (!means || !vars || !out_dispersion) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
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

        dispersion(means_arr, vars_arr, dispersion_arr);

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

