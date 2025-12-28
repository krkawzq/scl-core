// =============================================================================
// FILE: scl/binding/c_api/stat/oneway_anova.cpp
// BRIEF: C API implementation for one-way ANOVA
// =============================================================================

#include "scl/binding/c_api/stat/oneway_anova.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/stat/oneway_anova.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

scl_error_t scl_stat_oneway_anova(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    scl_size_t n_samples,
    scl_size_t n_groups,
    scl_real_t* F_stats,
    scl_real_t* p_values,
    scl_size_t n_features
) {
    if (!matrix || !group_ids || !F_stats || !p_values) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const int32_t> groups(
            reinterpret_cast<const int32_t*>(group_ids),
            n_samples
        );
        scl::Array<scl::Real> F_arr(
            reinterpret_cast<scl::Real*>(F_stats),
            n_features
        );
        scl::Array<scl::Real> pval_arr(
            reinterpret_cast<scl::Real*>(p_values),
            n_features
        );

        matrix->visit([&](auto& mat) {
            scl::kernel::stat::oneway_anova::oneway_anova(
                mat, groups, static_cast<scl::Size>(n_groups),
                F_arr, pval_arr
            );
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
