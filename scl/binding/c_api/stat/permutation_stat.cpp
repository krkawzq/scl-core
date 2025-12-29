// =============================================================================
// FILE: scl/binding/c_api/stat/permutation_stat.cpp
// BRIEF: C API implementation for permutation testing
// =============================================================================

#include "scl/binding/c_api/stat/permutation_stat.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/stat/permutation_stat.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

// =============================================================================
// Helper Functions
// =============================================================================

namespace {
    constexpr Size DEFAULT_N_PERMUTATIONS = 1000;

    [[nodiscard]] constexpr auto convert_perm_stat_type(
        scl_perm_stat_type_t type) noexcept -> scl::kernel::stat::permutation_stat::PermStatType {
        using PST = scl::kernel::stat::permutation_stat::PermStatType;
        switch (type) {
            case SCL_PERM_STAT_MEAN_DIFF: return PST::MeanDiff;
            case SCL_PERM_STAT_KS: return PST::KS;
            case SCL_PERM_STAT_MWU: [[fallthrough]];
            default: return PST::MWU;
        }
    }
} // anonymous namespace

extern "C" {

// =============================================================================
// Batch Permutation Test
// =============================================================================

SCL_EXPORT scl_error_t scl_stat_batch_permutation(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    const scl_size_t n_samples,
    scl_real_t* p_values,
    const scl_size_t n_features,
    const scl_size_t n_permutations,
    const scl_perm_stat_type_t stat_type,
    const uint64_t seed) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(group_ids, "Group IDs pointer is null");
    SCL_C_API_CHECK_NULL(p_values, "P-values pointer is null");
    SCL_C_API_CHECK(n_samples > 0 && n_features > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");

    SCL_C_API_TRY
        const Size n_samples_sz = static_cast<Size>(n_samples);
        const Size n_features_sz = static_cast<Size>(n_features);
        const Size n_perm = (n_permutations == 0) 
                          ? DEFAULT_N_PERMUTATIONS 
                          : static_cast<Size>(n_permutations);
        
        Array<const int32_t> groups_arr(group_ids, n_samples_sz);
        Array<Real> pval_arr(reinterpret_cast<Real*>(p_values), n_features_sz);

        matrix->visit([&](auto& mat) {
            scl::kernel::stat::permutation_stat::batch_permutation_reuse_sort(
                mat, groups_arr, n_perm, pval_arr, convert_perm_stat_type(stat_type), seed
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Single Permutation Test
// =============================================================================

SCL_EXPORT scl_error_t scl_stat_permutation_single(
    const scl_real_t* values,
    const scl_size_t n_samples,
    const int32_t* group_ids,
    const scl_size_t n_permutations,
    const scl_perm_stat_type_t stat_type,
    const uint64_t seed,
    scl_real_t* p_value_out) {
    
    SCL_C_API_CHECK_NULL(values, "Values pointer is null");
    SCL_C_API_CHECK_NULL(group_ids, "Group IDs pointer is null");
    SCL_C_API_CHECK_NULL(p_value_out, "P-value output pointer is null");
    SCL_C_API_CHECK(n_samples > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of samples must be positive");

    SCL_C_API_TRY
        const Size n_samples_sz = static_cast<Size>(n_samples);
        const Size n_perm = (n_permutations == 0) 
                          ? DEFAULT_N_PERMUTATIONS 
                          : static_cast<Size>(n_permutations);
        
        Array<const Real> vals_arr(reinterpret_cast<const Real*>(values), n_samples_sz);
        Array<const int32_t> groups_arr(group_ids, n_samples_sz);

        const Real p_value = scl::kernel::stat::permutation_stat::permutation_test_single(
            vals_arr, groups_arr, n_perm, convert_perm_stat_type(stat_type), seed
        );
        
        *p_value_out = static_cast<scl_real_t>(p_value);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
