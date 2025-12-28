// =============================================================================
// FILE: scl/binding/c_api/stat/permutation_stat.cpp
// BRIEF: C API implementation for permutation testing
// =============================================================================

#include "scl/binding/c_api/stat/permutation_stat.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/stat/permutation_stat.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

namespace {
    using namespace scl::kernel::stat::permutation_stat;
    constexpr scl::Size DEFAULT_N_PERMUTATIONS = 1000;
}

scl_error_t scl_stat_batch_permutation(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    scl_size_t n_samples,
    scl_real_t* p_values,
    scl_size_t n_features,
    scl_size_t n_permutations,
    scl_perm_stat_type_t stat_type,
    uint64_t seed
) {
    if (!matrix || !group_ids || !p_values) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const int32_t> groups(
            reinterpret_cast<const int32_t*>(group_ids),
            n_samples
        );
        scl::Array<scl::Real> pval_arr(
            reinterpret_cast<scl::Real*>(p_values),
            n_features
        );
        scl::Size n_perm = (n_permutations == 0) ?
            DEFAULT_N_PERMUTATIONS : n_permutations;

        PermStatType stat = (stat_type == SCL_PERM_STAT_MEAN_DIFF) ?
            PermStatType::MeanDiff :
            (stat_type == SCL_PERM_STAT_KS) ?
            PermStatType::KS : PermStatType::MWU;

        matrix->visit([&](auto& mat) {
            batch_permutation_reuse_sort(
                mat, groups, n_perm, pval_arr, stat, seed
            );
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_stat_permutation_single(
    const scl_real_t* values,
    scl_size_t n_samples,
    const int32_t* group_ids,
    scl_size_t n_permutations,
    scl_perm_stat_type_t stat_type,
    uint64_t seed,
    scl_real_t* p_value_out
) {
    if (!values || !group_ids || !p_value_out) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Real> vals_arr(
            reinterpret_cast<const scl::Real*>(values),
            n_samples
        );
        scl::Array<const int32_t> groups(
            reinterpret_cast<const int32_t*>(group_ids),
            n_samples
        );
        scl::Size n_perm = (n_permutations == 0) ?
            DEFAULT_N_PERMUTATIONS : n_permutations;

        PermStatType stat = (stat_type == SCL_PERM_STAT_MEAN_DIFF) ?
            PermStatType::MeanDiff :
            (stat_type == SCL_PERM_STAT_KS) ?
            PermStatType::KS : PermStatType::MWU;

        *p_value_out = static_cast<scl_real_t>(
            permutation_test_single(vals_arr, groups, n_perm, stat, seed)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
