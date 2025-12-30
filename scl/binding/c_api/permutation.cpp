// =============================================================================
// FILE: scl/binding/c_api/permutation/permutation.cpp
// BRIEF: C API implementation for permutation testing
// =============================================================================

#include "scl/binding/c_api/permutation.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/permutation.hpp"
#include "scl/core/type.hpp"

extern "C" {

using scl::Index;
using scl::Size;
using scl::Real;
using scl::Array;
using scl::binding::SparseWrapper;

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
// Multiple Testing Corrections
// =============================================================================

SCL_EXPORT scl_error_t scl_perm_bonferroni_correction(
    const scl_real_t* p_values,
    const scl_size_t n,
    scl_real_t* adjusted) {
    
    SCL_C_API_CHECK_NULL(p_values, "P-values array is null");
    SCL_C_API_CHECK_NULL(adjusted, "Adjusted p-values array is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");
    
    SCL_C_API_TRY
        Array<const Real> pvals(reinterpret_cast<const Real*>(p_values), n);
        Array<Real> adj(reinterpret_cast<Real*>(adjusted), n);
        
        scl::kernel::permutation::bonferroni_correction(pvals, adj);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_perm_holm_correction(
    const scl_real_t* p_values,
    const scl_size_t n,
    scl_real_t* adjusted) {
    
    SCL_C_API_CHECK_NULL(p_values, "P-values array is null");
    SCL_C_API_CHECK_NULL(adjusted, "Adjusted p-values array is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");
    
    SCL_C_API_TRY
        Array<const Real> pvals(reinterpret_cast<const Real*>(p_values), n);
        Array<Real> adj(reinterpret_cast<Real*>(adjusted), n);
        
        scl::kernel::permutation::holm_correction(pvals, adj);
        
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

// =============================================================================
// Utilities
// =============================================================================

SCL_EXPORT scl_error_t scl_perm_count_significant(
    const scl_real_t* p_values,
    scl_size_t n,
    scl_real_t alpha,
    scl_size_t* n_significant) {
    
    SCL_C_API_CHECK_NULL(p_values, "P-values array is null");
    SCL_C_API_CHECK_NULL(n_significant, "Output count pointer is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");
    SCL_C_API_CHECK(alpha > 0.0 && alpha <= 1.0, SCL_ERROR_INVALID_ARGUMENT,
                   "Alpha must be in (0, 1]");
    
    SCL_C_API_TRY
        Array<const Real> pvals(reinterpret_cast<const Real*>(p_values), n);
        const Size count = scl::kernel::permutation::count_significant(
            pvals, static_cast<Real>(alpha)
        );
        *n_significant = count;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_perm_get_significant_indices(
    const scl_real_t* p_values,
    scl_size_t n,
    scl_real_t alpha,
    scl_index_t* indices,
    scl_size_t max_results,
    scl_size_t* n_results) {
    
    SCL_C_API_CHECK_NULL(p_values, "P-values array is null");
    SCL_C_API_CHECK_NULL(indices, "Indices array is null");
    SCL_C_API_CHECK_NULL(n_results, "Output count pointer is null");
    SCL_C_API_CHECK(n > 0 && max_results > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array sizes must be positive");
    SCL_C_API_CHECK(alpha > 0.0 && alpha <= 1.0, SCL_ERROR_INVALID_ARGUMENT,
                   "Alpha must be in (0, 1]");
    
    SCL_C_API_TRY
        Array<const Real> pvals(reinterpret_cast<const Real*>(p_values), n);
        Array<Index> idx_arr(reinterpret_cast<Index*>(indices), max_results);
        Size count = 0;
        
        scl::kernel::permutation::get_significant_indices(
            pvals, static_cast<Real>(alpha), idx_arr, count
        );
        
        *n_results = count;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Batch Permutation Test
// =============================================================================

SCL_EXPORT scl_error_t scl_perm_batch_test(
    scl_sparse_t matrix,
    const scl_index_t* group_labels,
    scl_size_t n_permutations,
    scl_real_t* p_values,
    uint64_t seed) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix is null");
    SCL_C_API_CHECK_NULL(group_labels, "Group labels array is null");
    SCL_C_API_CHECK_NULL(p_values, "P-values array is null");
    SCL_C_API_CHECK(n_permutations > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of permutations must be positive");
    
    SCL_C_API_TRY
        // batch_permutation_test requires CSR format
        if (!matrix->is_csr_format()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        const auto& mat = matrix->as_csr();
        const Index n_rows = mat.rows();
        const Index n_cols = mat.cols();
        
        Array<const Index> labels_arr(
            reinterpret_cast<const Index*>(group_labels),
            static_cast<Size>(n_cols)
        );
        Array<Real> pval_arr(
            reinterpret_cast<Real*>(p_values),
            static_cast<Size>(n_rows)
        );
        
        scl::kernel::permutation::batch_permutation_test(
            mat, labels_arr, n_permutations, pval_arr, seed
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
