// =============================================================================
// FILE: scl/binding/c_api/stat/effect_size.cpp
// BRIEF: C API implementation for effect size computation
// =============================================================================

#include "scl/binding/c_api/stat/effect_size.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/stat/effect_size.hpp"
#include "scl/core/type.hpp"

namespace scl::binding {
    using namespace scl::kernel::stat::effect_size;
}

extern "C" {

namespace {
    [[nodiscard]] constexpr auto convert_effect_size_type(scl_effect_size_type_t type) noexcept 
        -> EffectSizeType {
        switch (type) {
            case SCL_EFFECT_SIZE_COHENS_D: return EffectSizeType::CohensD;
            case SCL_EFFECT_SIZE_HEDGES_G: return EffectSizeType::HedgesG;
            case SCL_EFFECT_SIZE_GLASS_DELTA: return EffectSizeType::GlassDelta;
            case SCL_EFFECT_SIZE_CLES: return EffectSizeType::CLES;
            default: return EffectSizeType::CohensD;
        }
    }
} // anonymous namespace

// =============================================================================
// Effect Size
// =============================================================================

scl_error_t scl_stat_effect_size(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    scl_real_t* out_effect_size,
    scl_index_t primary_dim,
    scl_effect_size_type_t type)
{
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(group_ids, "Group IDs pointer is null");
    SCL_C_API_CHECK_NULL(out_effect_size, "Output effect size pointer is null");

    SCL_C_API_TRY {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        SCL_C_API_CHECK(wrapper->valid(), SCL_ERROR_INVALID_ARGUMENT,
                       "Invalid sparse matrix");

        const Index actual_primary_dim = wrapper->is_csr_format() 
                                       ? wrapper->rows() 
                                       : wrapper->cols();
        const Index secondary_dim = wrapper->is_csr_format() 
                                  ? wrapper->cols() 
                                  : wrapper->rows();
        
        SCL_C_API_CHECK(static_cast<scl_index_t>(actual_primary_dim) == primary_dim,
                       SCL_ERROR_DIMENSION_MISMATCH,
                       "Primary dimension mismatch");

        const Size primary_dim_sz = static_cast<Size>(primary_dim);
        const Size secondary_dim_sz = static_cast<Size>(secondary_dim);
        
        Array<const int32_t> group_ids_arr(group_ids, secondary_dim_sz);
        Array<Real> effect_size_arr(reinterpret_cast<Real*>(out_effect_size), primary_dim_sz);
        
        wrapper->visit([&](auto& m) {
            effect_size(m, group_ids_arr, effect_size_arr, convert_effect_size_type(type));
        });
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// T-Test with Effect Size
// =============================================================================

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
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(group_ids, "Group IDs pointer is null");
    SCL_C_API_CHECK_NULL(out_t_stats, "Output t-statistics pointer is null");
    SCL_C_API_CHECK_NULL(out_p_values, "Output p-values pointer is null");
    SCL_C_API_CHECK_NULL(out_log2_fc, "Output log2 fold change pointer is null");
    SCL_C_API_CHECK_NULL(out_effect_size, "Output effect size pointer is null");

    SCL_C_API_TRY {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        SCL_C_API_CHECK(wrapper->valid(), SCL_ERROR_INVALID_ARGUMENT,
                       "Invalid sparse matrix");

        const Index actual_primary_dim = wrapper->is_csr_format() 
                                       ? wrapper->rows() 
                                       : wrapper->cols();
        const Index secondary_dim = wrapper->is_csr_format() 
                                  ? wrapper->cols() 
                                  : wrapper->rows();
        
        SCL_C_API_CHECK(static_cast<scl_index_t>(actual_primary_dim) == primary_dim,
                       SCL_ERROR_DIMENSION_MISMATCH,
                       "Primary dimension mismatch");

        const Size primary_dim_sz = static_cast<Size>(primary_dim);
        const Size secondary_dim_sz = static_cast<Size>(secondary_dim);
        
        Array<const int32_t> group_ids_arr(group_ids, secondary_dim_sz);
        Array<Real> t_stats_arr(reinterpret_cast<Real*>(out_t_stats), primary_dim_sz);
        Array<Real> p_values_arr(reinterpret_cast<Real*>(out_p_values), primary_dim_sz);
        Array<Real> log2_fc_arr(reinterpret_cast<Real*>(out_log2_fc), primary_dim_sz);
        Array<Real> effect_size_arr(reinterpret_cast<Real*>(out_effect_size), primary_dim_sz);
        
        wrapper->visit([&](auto& m) {
            ttest_with_effect_size(m, group_ids_arr, t_stats_arr, p_values_arr,
                                  log2_fc_arr, effect_size_arr,
                                  convert_effect_size_type(es_type),
                                  use_welch != 0);
        });
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

} // extern "C"
