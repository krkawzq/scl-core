// =============================================================================
// FILE: scl/binding/c_api/permutation/permutation.cpp
// BRIEF: C API implementation for permutation testing
// =============================================================================

#include "scl/binding/c_api/permutation.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/permutation.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Correlation Test
// =============================================================================

SCL_EXPORT scl_error_t scl_perm_correlation_test(
    const scl_real_t* x,
    const scl_real_t* y,
    const scl_size_t n,
    const scl_real_t observed_correlation,
    const scl_size_t n_permutations,
    scl_real_t* p_value,
    const uint64_t seed) {
    
    SCL_C_API_CHECK_NULL(x, "X array is null");
    SCL_C_API_CHECK_NULL(y, "Y array is null");
    SCL_C_API_CHECK_NULL(p_value, "P-value pointer is null");
    SCL_C_API_CHECK(n > 0 && n_permutations > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");

    SCL_C_API_TRY
        Array<const Real> x_arr(reinterpret_cast<const Real*>(x), n);
        Array<const Real> y_arr(reinterpret_cast<const Real*>(y), n);

        const Real pval = scl::kernel::permutation::permutation_correlation_test(
            x_arr, y_arr, static_cast<Real>(observed_correlation),
            n_permutations, seed
        );

        *p_value = static_cast<scl_real_t>(pval);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// FDR Correction (BH)
// =============================================================================

SCL_EXPORT scl_error_t scl_perm_fdr_correction_bh(
    const scl_real_t* p_values,
    const scl_size_t n,
    scl_real_t* q_values) {
    
    SCL_C_API_CHECK_NULL(p_values, "P-values array is null");
    SCL_C_API_CHECK_NULL(q_values, "Q-values array is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");

    SCL_C_API_TRY
        Array<const Real> pvals(reinterpret_cast<const Real*>(p_values), n);
        Array<Real> qvals(reinterpret_cast<Real*>(q_values), n);

        scl::kernel::permutation::fdr_correction_bh(pvals, qvals);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// FDR Correction (BY)
// =============================================================================

SCL_EXPORT scl_error_t scl_perm_fdr_correction_by(
    const scl_real_t* p_values,
    const scl_size_t n,
    scl_real_t* q_values) {
    
    SCL_C_API_CHECK_NULL(p_values, "P-values array is null");
    SCL_C_API_CHECK_NULL(q_values, "Q-values array is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");

    SCL_C_API_TRY
        Array<const Real> pvals(reinterpret_cast<const Real*>(p_values), n);
        Array<Real> qvals(reinterpret_cast<Real*>(q_values), n);

        scl::kernel::permutation::fdr_correction_by(pvals, qvals);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
