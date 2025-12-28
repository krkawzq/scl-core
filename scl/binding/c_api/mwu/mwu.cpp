// =============================================================================
// FILE: scl/binding/c_api/mwu/mwu.cpp
// BRIEF: C API implementation for Mann-Whitney U Test
// =============================================================================

#include "scl/binding/c_api/mwu/mwu.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/mwu.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::mwu;

extern "C" {

scl_error_t scl_mwu_test(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    scl_size_t n_samples,
    scl_real_t* out_u_stats,
    scl_real_t* out_p_values,
    scl_real_t* out_log2_fc,
    scl_real_t* out_auroc)
{
    if (!matrix || !group_ids || !out_u_stats || !out_p_values || !out_log2_fc) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(matrix);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        const Index n_features = wrapper->rows();
        Array<const int32_t> groups_arr(group_ids, n_samples);
        Array<Real> u_arr(reinterpret_cast<Real*>(out_u_stats), static_cast<Size>(n_features));
        Array<Real> p_arr(reinterpret_cast<Real*>(out_p_values), static_cast<Size>(n_features));
        Array<Real> fc_arr(reinterpret_cast<Real*>(out_log2_fc), static_cast<Size>(n_features));
        
        Array<Real> auroc_arr;
        if (out_auroc) {
            auroc_arr = Array<Real>(reinterpret_cast<Real*>(out_auroc), static_cast<Size>(n_features));
        }
        
        wrapper->visit([&](auto& m) {
            mwu_test(m, groups_arr, u_arr, p_arr, fc_arr, auroc_arr);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

