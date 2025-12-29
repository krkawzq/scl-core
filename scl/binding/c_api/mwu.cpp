// =============================================================================
// FILE: scl/binding/c_api/mwu.cpp
// BRIEF: C API implementation for Mann-Whitney U Test
// =============================================================================

#include "scl/binding/c_api/mwu.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/mwu.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Mann-Whitney U Test
// =============================================================================

SCL_EXPORT scl_error_t scl_mwu_test(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    const scl_size_t n_samples,
    scl_real_t* out_u_stats,
    scl_real_t* out_p_values,
    scl_real_t* out_log2_fc,
    scl_real_t* out_auroc) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(group_ids, "Group IDs pointer is null");
    SCL_C_API_CHECK_NULL(out_u_stats, "Output U statistics pointer is null");
    SCL_C_API_CHECK_NULL(out_p_values, "Output p-values pointer is null");
    SCL_C_API_CHECK_NULL(out_log2_fc, "Output log2 fold change pointer is null");
    SCL_C_API_CHECK(n_samples > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of samples must be positive");

    SCL_C_API_TRY
        const Index n_features = matrix->rows();
        const Size n_samples_sz = static_cast<Size>(n_samples);
        const Size n_features_sz = static_cast<Size>(n_features);

        // Wrap C arrays with Array views
        Array<const int32_t> groups_arr(group_ids, n_samples_sz);
        Array<Real> u_arr(reinterpret_cast<Real*>(out_u_stats), n_features_sz);
        Array<Real> p_arr(reinterpret_cast<Real*>(out_p_values), n_features_sz);
        Array<Real> fc_arr(reinterpret_cast<Real*>(out_log2_fc), n_features_sz);
        
        // Optional AUROC output
        Array<Real> auroc_arr;
        if (out_auroc != nullptr) {
            auroc_arr = Array<Real>(reinterpret_cast<Real*>(out_auroc), n_features_sz);
        }
        
        // Dispatch to kernel implementation
        matrix->visit([&](auto& m) {
            scl::kernel::mwu::mwu_test(m, groups_arr, u_arr, p_arr, fc_arr, auroc_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
