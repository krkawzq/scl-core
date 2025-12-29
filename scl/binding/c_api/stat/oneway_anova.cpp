// =============================================================================
// FILE: scl/binding/c_api/stat/oneway_anova.cpp
// BRIEF: C API implementation for one-way ANOVA
// =============================================================================

#include "scl/binding/c_api/stat/oneway_anova.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/stat/oneway_anova.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// One-Way ANOVA
// =============================================================================

SCL_EXPORT scl_error_t scl_stat_oneway_anova(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    const scl_size_t n_samples,
    const scl_size_t n_groups,
    scl_real_t* F_stats,
    scl_real_t* p_values,
    const scl_size_t n_features) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(group_ids, "Group IDs pointer is null");
    SCL_C_API_CHECK_NULL(F_stats, "F statistics pointer is null");
    SCL_C_API_CHECK_NULL(p_values, "P-values pointer is null");
    SCL_C_API_CHECK(n_samples > 0 && n_groups > 0 && n_features > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");

    SCL_C_API_TRY
        const Size n_samples_sz = static_cast<Size>(n_samples);
        const Size n_groups_sz = static_cast<Size>(n_groups);
        const Size n_features_sz = static_cast<Size>(n_features);
        
        Array<const int32_t> groups_arr(group_ids, n_samples_sz);
        Array<Real> F_arr(reinterpret_cast<Real*>(F_stats), n_features_sz);
        Array<Real> pval_arr(reinterpret_cast<Real*>(p_values), n_features_sz);

        matrix->visit([&](auto& mat) {
            scl::kernel::stat::oneway_anova::oneway_anova(mat, groups_arr, n_groups_sz, F_arr, pval_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
