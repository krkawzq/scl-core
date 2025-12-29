// =============================================================================
// FILE: scl/binding/c_api/stat/kruskal_wallis.cpp
// BRIEF: C API implementation for Kruskal-Wallis test
// =============================================================================

#include "scl/binding/c_api/stat/kruskal_wallis.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/stat/kruskal_wallis.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Kruskal-Wallis Test
// =============================================================================

SCL_EXPORT scl_error_t scl_stat_kruskal_wallis(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    const scl_size_t n_samples,
    const scl_size_t n_groups,
    scl_real_t* H_stats,
    scl_real_t* p_values,
    const scl_size_t n_features) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(group_ids, "Group IDs pointer is null");
    SCL_C_API_CHECK_NULL(H_stats, "H statistics pointer is null");
    SCL_C_API_CHECK_NULL(p_values, "P-values pointer is null");
    SCL_C_API_CHECK(n_samples > 0 && n_groups > 0 && n_features > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");

    SCL_C_API_TRY
        const Size n_samples_sz = static_cast<Size>(n_samples);
        const Size n_groups_sz = static_cast<Size>(n_groups);
        const Size n_features_sz = static_cast<Size>(n_features);
        
        Array<const int32_t> groups_arr(group_ids, n_samples_sz);
        Array<Real> H_arr(reinterpret_cast<Real*>(H_stats), n_features_sz);
        Array<Real> pval_arr(reinterpret_cast<Real*>(p_values), n_features_sz);

        matrix->visit([&](auto& mat) {
            scl::kernel::stat::kruskal_wallis::kruskal_wallis(mat, groups_arr, n_groups_sz, H_arr, pval_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
