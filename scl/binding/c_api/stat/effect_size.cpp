// =============================================================================
// FILE: scl/binding/c_api/stat/effect_size.cpp
// BRIEF: C API implementation for effect size computation
// =============================================================================

#include "scl/binding/c_api/stat/effect_size.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/stat/effect_size.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

scl_error_t scl_stat_effect_size(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    scl_real_t* out_effect_size,
    scl_index_t primary_dim,
    scl_effect_size_type_t type)
{
    if (!matrix || !group_ids || !out_effect_size) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Index actual_primary_dim = matrix->is_csr ? matrix->rows() : matrix->cols();
        Index secondary_dim = matrix->is_csr ? matrix->cols() : matrix->rows();
        
        if (static_cast<scl_index_t>(actual_primary_dim) != primary_dim) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Primary dimension mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }
        
        Array<const int32_t> group_ids_arr(
            group_ids,
            static_cast<Size>(secondary_dim)
        );
        Array<Real> out_effect_size_arr(
            reinterpret_cast<Real*>(out_effect_size),
            static_cast<Size>(primary_dim)
        );
        
        scl::kernel::stat::effect_size::EffectSizeType type_enum;
        switch (type) {
            case SCL_EFFECT_SIZE_COHENS_D:
                type_enum = scl::kernel::stat::effect_size::EffectSizeType::CohensD;
                break;
            case SCL_EFFECT_SIZE_HEDGES_G:
                type_enum = scl::kernel::stat::effect_size::EffectSizeType::HedgesG;
                break;
            case SCL_EFFECT_SIZE_GLASS_DELTA:
                type_enum = scl::kernel::stat::effect_size::EffectSizeType::GlassDelta;
                break;
            case SCL_EFFECT_SIZE_CLES:
                type_enum = scl::kernel::stat::effect_size::EffectSizeType::CLES;
                break;
            default:
                set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid effect size type");
                return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        matrix->visit([&](auto& m) {
            scl::kernel::stat::effect_size::effect_size(
                m,
                group_ids_arr,
                out_effect_size_arr,
                type_enum
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_stat_ttest_with_effect_size(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    scl_real_t* out_t_stats,
    scl_real_t* out_p_values,
    scl_real_t* out_log2_fc,
    scl_real_t* out_effect_size,
    scl_index_t primary_dim,
    scl_effect_size_type_t es_type,
    int32_t use_welch)
{
    if (!matrix || !group_ids || !out_t_stats || !out_p_values || 
        !out_log2_fc || !out_effect_size) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Index actual_primary_dim = matrix->is_csr ? matrix->rows() : matrix->cols();
        Index secondary_dim = matrix->is_csr ? matrix->cols() : matrix->rows();
        
        if (static_cast<scl_index_t>(actual_primary_dim) != primary_dim) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Primary dimension mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }
        
        Array<const int32_t> group_ids_arr(
            group_ids,
            static_cast<Size>(secondary_dim)
        );
        Array<Real> out_t_stats_arr(
            reinterpret_cast<Real*>(out_t_stats),
            static_cast<Size>(primary_dim)
        );
        Array<Real> out_p_values_arr(
            reinterpret_cast<Real*>(out_p_values),
            static_cast<Size>(primary_dim)
        );
        Array<Real> out_log2_fc_arr(
            reinterpret_cast<Real*>(out_log2_fc),
            static_cast<Size>(primary_dim)
        );
        Array<Real> out_effect_size_arr(
            reinterpret_cast<Real*>(out_effect_size),
            static_cast<Size>(primary_dim)
        );
        
        scl::kernel::stat::effect_size::EffectSizeType es_type_enum;
        switch (es_type) {
            case SCL_EFFECT_SIZE_COHENS_D:
                es_type_enum = scl::kernel::stat::effect_size::EffectSizeType::CohensD;
                break;
            case SCL_EFFECT_SIZE_HEDGES_G:
                es_type_enum = scl::kernel::stat::effect_size::EffectSizeType::HedgesG;
                break;
            case SCL_EFFECT_SIZE_GLASS_DELTA:
                es_type_enum = scl::kernel::stat::effect_size::EffectSizeType::GlassDelta;
                break;
            case SCL_EFFECT_SIZE_CLES:
                es_type_enum = scl::kernel::stat::effect_size::EffectSizeType::CLES;
                break;
            default:
                set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid effect size type");
                return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        matrix->visit([&](auto& m) {
            scl::kernel::stat::effect_size::ttest_with_effect_size(
                m,
                group_ids_arr,
                out_t_stats_arr,
                out_p_values_arr,
                out_log2_fc_arr,
                out_effect_size_arr,
                es_type_enum,
                use_welch != 0
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

