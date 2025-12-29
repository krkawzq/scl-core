// =============================================================================
// FILE: scl/binding/c_api/multiple_testing/multiple_testing.cpp
// BRIEF: C API implementation for multiple testing correction
// =============================================================================

#include "scl/binding/c_api/multiple_testing.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/multiple_testing.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Benjamini-Hochberg (FDR)
// =============================================================================

SCL_EXPORT scl_error_t scl_benjamini_hochberg(
    const scl_real_t* p_values,
    const scl_size_t n_tests,
    scl_real_t* adjusted_p_values,
    const scl_real_t fdr_level) {
    
    SCL_C_API_CHECK_NULL(p_values, "P-values array is null");
    SCL_C_API_CHECK_NULL(adjusted_p_values, "Adjusted p-values array is null");
    SCL_C_API_CHECK(n_tests > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of tests must be positive");
    SCL_C_API_CHECK(fdr_level > 0 && fdr_level < 1, SCL_ERROR_INVALID_ARGUMENT,
                   "FDR level must be in (0, 1)");
    
    SCL_C_API_TRY
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p_values), n_tests);
        Array<Real> adj_arr(reinterpret_cast<Real*>(adjusted_p_values), n_tests);
        
        scl::kernel::multiple_testing::benjamini_hochberg(p_arr, adj_arr, static_cast<Real>(fdr_level));
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Bonferroni
// =============================================================================

SCL_EXPORT scl_error_t scl_bonferroni(
    const scl_real_t* p_values,
    const scl_size_t n_tests,
    scl_real_t* adjusted_p_values) {
    
    SCL_C_API_CHECK_NULL(p_values, "P-values array is null");
    SCL_C_API_CHECK_NULL(adjusted_p_values, "Adjusted p-values array is null");
    SCL_C_API_CHECK(n_tests > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of tests must be positive");
    
    SCL_C_API_TRY
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p_values), n_tests);
        Array<Real> adj_arr(reinterpret_cast<Real*>(adjusted_p_values), n_tests);
        
        scl::kernel::multiple_testing::bonferroni(p_arr, adj_arr);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Benjamini-Yekutieli
// =============================================================================

SCL_EXPORT scl_error_t scl_benjamini_yekutieli(
    const scl_real_t* p_values,
    const scl_size_t n_tests,
    scl_real_t* adjusted_p_values) {
    
    SCL_C_API_CHECK_NULL(p_values, "P-values array is null");
    SCL_C_API_CHECK_NULL(adjusted_p_values, "Adjusted p-values array is null");
    SCL_C_API_CHECK(n_tests > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of tests must be positive");
    
    SCL_C_API_TRY
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p_values), n_tests);
        Array<Real> adj_arr(reinterpret_cast<Real*>(adjusted_p_values), n_tests);
        
        scl::kernel::multiple_testing::benjamini_yekutieli(p_arr, adj_arr);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Holm-Bonferroni
// =============================================================================

SCL_EXPORT scl_error_t scl_holm_bonferroni(
    const scl_real_t* p_values,
    const scl_size_t n_tests,
    scl_real_t* adjusted_p_values) {
    
    SCL_C_API_CHECK_NULL(p_values, "P-values array is null");
    SCL_C_API_CHECK_NULL(adjusted_p_values, "Adjusted p-values array is null");
    SCL_C_API_CHECK(n_tests > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of tests must be positive");
    
    SCL_C_API_TRY
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p_values), n_tests);
        Array<Real> adj_arr(reinterpret_cast<Real*>(adjusted_p_values), n_tests);
        
        scl::kernel::multiple_testing::holm_bonferroni(p_arr, adj_arr);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
