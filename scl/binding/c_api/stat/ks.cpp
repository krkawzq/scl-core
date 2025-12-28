// =============================================================================
// FILE: scl/binding/c_api/stat/ks.cpp
// BRIEF: C API implementation for Kolmogorov-Smirnov test
// =============================================================================

#include "scl/binding/c_api/stat/ks.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/stat/ks.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

scl_error_t scl_stat_ks_test(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    scl_size_t n_samples,
    scl_real_t* D_stats,
    scl_real_t* p_values,
    scl_size_t n_features
) {
    if (!matrix || !group_ids || !D_stats || !p_values) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const int32_t> groups(
            reinterpret_cast<const int32_t*>(group_ids),
            n_samples
        );
        scl::Array<scl::Real> D_arr(
            reinterpret_cast<scl::Real*>(D_stats),
            n_features
        );
        scl::Array<scl::Real> pval_arr(
            reinterpret_cast<scl::Real*>(p_values),
            n_features
        );

        matrix->visit([&](auto& mat) {
            scl::kernel::stat::ks::ks_test(mat, groups, D_arr, pval_arr);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
