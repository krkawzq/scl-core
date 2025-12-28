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

scl_error_t scl_hotspot_local_morans_i(
    scl_sparse_t spatial_weights,
    const scl_real_t* values,
    scl_index_t n,
    scl_real_t* local_i,
    scl_real_t* z_scores,
    scl_real_t* p_values,
    scl_index_t n_permutations,
    uint64_t seed)
{
    if (!spatial_weights || !values || !local_i || !z_scores || !p_values) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!spatial_weights->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
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
                w,
                values_arr,
                n,
                local_i_arr,
                z_scores_arr,
                p_values_arr,
                n_permutations,
                seed
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_hotspot_getis_ord_g_star(
    scl_sparse_t spatial_weights,
    const scl_real_t* values,
    scl_index_t n,
    scl_real_t* g_star,
    scl_real_t* z_scores,
    scl_real_t* p_values,
    int include_self,
    scl_index_t n_permutations,
    uint64_t seed)
{
    if (!spatial_weights || !values || !g_star || !z_scores || !p_values) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!spatial_weights->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
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
                w,
                values_arr,
                n,
                g_star_arr,
                z_scores_arr,
                p_values_arr,
                include_self != 0
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_hotspot_local_gearys_c(
    scl_sparse_t spatial_weights,
    const scl_real_t* values,
    scl_index_t n,
    scl_real_t* local_c,
    scl_real_t* z_scores,
    scl_real_t* p_values,
    scl_index_t n_permutations,
    uint64_t seed)
{
    if (!spatial_weights || !values || !local_c || !z_scores || !p_values) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!spatial_weights->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
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
                w,
                values_arr,
                n,
                local_c_arr,
                z_scores_arr,
                p_values_arr,
                n_permutations,
                seed
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_hotspot_global_morans_i(
    scl_sparse_t spatial_weights,
    const scl_real_t* values,
    scl_index_t n,
    scl_real_t* moran_i,
    scl_real_t* z_score,
    scl_real_t* p_value)
{
    if (!spatial_weights || !values || !moran_i || !z_score || !p_value) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!spatial_weights->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<const Real> values_arr(
            reinterpret_cast<const Real*>(values),
            static_cast<Size>(n)
        );
        
        Real moran_i_val, z_score_val, p_value_val;
        
        spatial_weights->visit([&](auto& w) {
            scl::kernel::hotspot::global_morans_i(
                w,
                values_arr,
                n,
                moran_i_val,
                z_score_val,
                p_value_val
            );
        });
        
        *moran_i = static_cast<scl_real_t>(moran_i_val);
        *z_score = static_cast<scl_real_t>(z_score_val);
        *p_value = static_cast<scl_real_t>(p_value_val);
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_hotspot_global_gearys_c(
    scl_sparse_t spatial_weights,
    const scl_real_t* values,
    scl_index_t n,
    scl_real_t* geary_c,
    scl_real_t* z_score,
    scl_real_t* p_value)
{
    if (!spatial_weights || !values || !geary_c || !z_score || !p_value) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!spatial_weights->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<const Real> values_arr(
            reinterpret_cast<const Real*>(values),
            static_cast<Size>(n)
        );
        
        Real geary_c_val, z_score_val, p_value_val;
        
        spatial_weights->visit([&](auto& w) {
            scl::kernel::hotspot::global_gearys_c(
                w,
                values_arr,
                n,
                geary_c_val,
                z_score_val,
                p_value_val
            );
        });
        
        *geary_c = static_cast<scl_real_t>(geary_c_val);
        *z_score = static_cast<scl_real_t>(z_score_val);
        *p_value = static_cast<scl_real_t>(p_value_val);
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

