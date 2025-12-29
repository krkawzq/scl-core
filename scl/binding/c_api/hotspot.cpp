// =============================================================================
// FILE: scl/binding/c_api/hotspot/hotspot.cpp
// BRIEF: C API implementation for spatial hotspot detection
// =============================================================================

#include "scl/binding/c_api/hotspot.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/hotspot.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Local Moran's I
// =============================================================================

SCL_EXPORT scl_error_t scl_hotspot_local_morans_i(
    scl_sparse_t spatial_weights,
    const scl_real_t* values,
    const scl_index_t n,
    scl_real_t* local_i,
    scl_real_t* z_scores,
    scl_real_t* p_values,
    const scl_index_t n_permutations,
    const uint64_t seed) {
    
    SCL_C_API_CHECK_NULL(spatial_weights, "Spatial weights matrix is null");
    SCL_C_API_CHECK_NULL(values, "Values array is null");
    SCL_C_API_CHECK_NULL(local_i, "Output local I array is null");
    SCL_C_API_CHECK_NULL(z_scores, "Output z-scores array is null");
    SCL_C_API_CHECK_NULL(p_values, "Output p-values array is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");
    
    SCL_C_API_TRY
        Array<const Real> values_arr(
            reinterpret_cast<const Real*>(values),
            static_cast<Size>(n)
        );
        Array<Real> local_i_arr(
            reinterpret_cast<Real*>(local_i),
            static_cast<Size>(n)
        );
        Array<Real> z_scores_arr(
            reinterpret_cast<Real*>(z_scores),
            static_cast<Size>(n)
        );
        Array<Real> p_values_arr(
            reinterpret_cast<Real*>(p_values),
            static_cast<Size>(n)
        );
        
        spatial_weights->visit([&](auto& w) {
            scl::kernel::hotspot::local_morans_i(
                w, values_arr, n,
                local_i_arr, z_scores_arr, p_values_arr,
                n_permutations, seed
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Getis-Ord G*
// =============================================================================

SCL_EXPORT scl_error_t scl_hotspot_getis_ord_g_star(
    scl_sparse_t spatial_weights,
    const scl_real_t* values,
    const scl_index_t n,
    scl_real_t* g_star,
    scl_real_t* z_scores,
    scl_real_t* p_values,
    const int include_self) {
    
    SCL_C_API_CHECK_NULL(spatial_weights, "Spatial weights matrix is null");
    SCL_C_API_CHECK_NULL(values, "Values array is null");
    SCL_C_API_CHECK_NULL(g_star, "Output G* array is null");
    SCL_C_API_CHECK_NULL(z_scores, "Output z-scores array is null");
    SCL_C_API_CHECK_NULL(p_values, "Output p-values array is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");
    
    SCL_C_API_TRY
        Array<const Real> values_arr(
            reinterpret_cast<const Real*>(values),
            static_cast<Size>(n)
        );
        Array<Real> g_star_arr(
            reinterpret_cast<Real*>(g_star),
            static_cast<Size>(n)
        );
        Array<Real> z_scores_arr(
            reinterpret_cast<Real*>(z_scores),
            static_cast<Size>(n)
        );
        Array<Real> p_values_arr(
            reinterpret_cast<Real*>(p_values),
            static_cast<Size>(n)
        );
        
        spatial_weights->visit([&](auto& w) {
            scl::kernel::hotspot::getis_ord_g_star(
                w, values_arr, n,
                g_star_arr, z_scores_arr, p_values_arr,
                include_self != 0
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Local Geary's C
// =============================================================================

SCL_EXPORT scl_error_t scl_hotspot_local_gearys_c(
    scl_sparse_t spatial_weights,
    const scl_real_t* values,
    const scl_index_t n,
    scl_real_t* local_c,
    scl_real_t* z_scores,
    scl_real_t* p_values,
    const scl_index_t n_permutations,
    const uint64_t seed) {
    
    SCL_C_API_CHECK_NULL(spatial_weights, "Spatial weights matrix is null");
    SCL_C_API_CHECK_NULL(values, "Values array is null");
    SCL_C_API_CHECK_NULL(local_c, "Output local C array is null");
    SCL_C_API_CHECK_NULL(z_scores, "Output z-scores array is null");
    SCL_C_API_CHECK_NULL(p_values, "Output p-values array is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");
    
    SCL_C_API_TRY
        Array<const Real> values_arr(
            reinterpret_cast<const Real*>(values),
            static_cast<Size>(n)
        );
        Array<Real> local_c_arr(
            reinterpret_cast<Real*>(local_c),
            static_cast<Size>(n)
        );
        Array<Real> z_scores_arr(
            reinterpret_cast<Real*>(z_scores),
            static_cast<Size>(n)
        );
        Array<Real> p_values_arr(
            reinterpret_cast<Real*>(p_values),
            static_cast<Size>(n)
        );
        
        spatial_weights->visit([&](auto& w) {
            scl::kernel::hotspot::local_gearys_c(
                w, values_arr, n,
                local_c_arr, z_scores_arr, p_values_arr,
                n_permutations, seed
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Global Moran's I
// =============================================================================

SCL_EXPORT scl_error_t scl_hotspot_global_morans_i(
    scl_sparse_t spatial_weights,
    const scl_real_t* values,
    const scl_index_t n,
    scl_real_t* moran_i,
    scl_real_t* z_score,
    scl_real_t* p_value) {
    
    SCL_C_API_CHECK_NULL(spatial_weights, "Spatial weights matrix is null");
    SCL_C_API_CHECK_NULL(values, "Values array is null");
    SCL_C_API_CHECK_NULL(moran_i, "Output Moran's I pointer is null");
    SCL_C_API_CHECK_NULL(z_score, "Output z-score pointer is null");
    SCL_C_API_CHECK_NULL(p_value, "Output p-value pointer is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");
    
    SCL_C_API_TRY
        Array<const Real> values_arr(
            reinterpret_cast<const Real*>(values),
            static_cast<Size>(n)
        );
        
        Real mi = Real(0);
        Real z = Real(0);
        Real p = Real(0);
        
        spatial_weights->visit([&](auto& w) {
            scl::kernel::hotspot::global_morans_i(w, values_arr, n, mi, z, p);
        });
        
        *moran_i = static_cast<scl_real_t>(mi);
        *z_score = static_cast<scl_real_t>(z);
        *p_value = static_cast<scl_real_t>(p);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Global Geary's C
// =============================================================================

SCL_EXPORT scl_error_t scl_hotspot_global_gearys_c(
    scl_sparse_t spatial_weights,
    const scl_real_t* values,
    const scl_index_t n,
    scl_real_t* geary_c,
    scl_real_t* z_score,
    scl_real_t* p_value) {
    
    SCL_C_API_CHECK_NULL(spatial_weights, "Spatial weights matrix is null");
    SCL_C_API_CHECK_NULL(values, "Values array is null");
    SCL_C_API_CHECK_NULL(geary_c, "Output Geary's C pointer is null");
    SCL_C_API_CHECK_NULL(z_score, "Output z-score pointer is null");
    SCL_C_API_CHECK_NULL(p_value, "Output p-value pointer is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");
    
    SCL_C_API_TRY
        Array<const Real> values_arr(
            reinterpret_cast<const Real*>(values),
            static_cast<Size>(n)
        );
        
        Real gc = Real(0);
        Real z = Real(0);
        Real p = Real(0);
        
        spatial_weights->visit([&](auto& w) {
            scl::kernel::hotspot::global_gearys_c(w, values_arr, n, gc, z, p);
        });
        
        *geary_c = static_cast<scl_real_t>(gc);
        *z_score = static_cast<scl_real_t>(z);
        *p_value = static_cast<scl_real_t>(p);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
