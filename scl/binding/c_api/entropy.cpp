// =============================================================================
// FILE: scl/binding/c_api/entropy/entropy.cpp
// BRIEF: C API implementation for information theory measures
// =============================================================================

#include "scl/binding/c_api/entropy.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/entropy.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Entropy Measures
// =============================================================================

SCL_EXPORT scl_error_t scl_entropy_discrete_entropy(
    const scl_real_t* probabilities,
    const scl_size_t n,
    const int use_log2,
    scl_real_t* entropy_out) {
    
    SCL_C_API_CHECK_NULL(probabilities, "Probabilities array is null");
    SCL_C_API_CHECK_NULL(entropy_out, "Output entropy pointer is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");
    
    SCL_C_API_TRY
        Array<const Real> probs_arr(
            reinterpret_cast<const Real*>(probabilities), n
        );
        
        *entropy_out = static_cast<scl_real_t>(
            scl::kernel::entropy::discrete_entropy(probs_arr, use_log2 != 0)
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_entropy_count_entropy(
    const scl_real_t* counts,
    const scl_size_t n,
    const int use_log2,
    scl_real_t* entropy_out) {
    
    SCL_C_API_CHECK_NULL(counts, "Counts array is null");
    SCL_C_API_CHECK_NULL(entropy_out, "Output entropy pointer is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");
    
    SCL_C_API_TRY
        *entropy_out = static_cast<scl_real_t>(
            scl::kernel::entropy::count_entropy(
                reinterpret_cast<const Real*>(counts),
                n, use_log2 != 0
            )
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_entropy_row_entropy(
    scl_sparse_t X,
    scl_real_t* entropies,
    const scl_size_t n_rows,
    const int normalize,
    const int use_log2) {
    
    SCL_C_API_CHECK_NULL(X, "Matrix is null");
    SCL_C_API_CHECK_NULL(entropies, "Output entropies array is null");
    SCL_C_API_CHECK(n_rows > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of rows must be positive");
    
    SCL_C_API_TRY
        Array<Real> ent_arr(
            reinterpret_cast<Real*>(entropies), n_rows
        );
        
        X->visit([&](auto& mat) {
            scl::kernel::entropy::row_entropy(mat, ent_arr, normalize != 0, use_log2 != 0);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Mutual Information
// =============================================================================

SCL_EXPORT scl_error_t scl_entropy_mutual_information(
    const scl_index_t* x_binned,
    const scl_index_t* y_binned,
    scl_size_t n,
    scl_index_t n_bins_x,
    scl_index_t n_bins_y,
    int use_log2,
    scl_real_t* mi_out) {
    
    SCL_C_API_CHECK_NULL(x_binned, "X binned array is null");
    SCL_C_API_CHECK_NULL(y_binned, "Y binned array is null");
    SCL_C_API_CHECK_NULL(mi_out, "Output MI pointer is null");
    SCL_C_API_CHECK(n > 0 && n_bins_x > 0 && n_bins_y > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        *mi_out = static_cast<scl_real_t>(
            scl::kernel::entropy::mutual_information(
                x_binned, y_binned, n, n_bins_x, n_bins_y, use_log2 != 0
            )
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_entropy_normalized_mi(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_size_t n,
    scl_index_t n_clusters1,
    scl_index_t n_clusters2,
    scl_real_t* nmi_out) {
    
    SCL_C_API_CHECK_NULL(labels1, "Labels1 array is null");
    SCL_C_API_CHECK_NULL(labels2, "Labels2 array is null");
    SCL_C_API_CHECK_NULL(nmi_out, "Output NMI pointer is null");
    SCL_C_API_CHECK(n > 0 && n_clusters1 > 0 && n_clusters2 > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        Array<const Index> l1_arr(labels1, n);
        Array<const Index> l2_arr(labels2, n);
        
        *nmi_out = static_cast<scl_real_t>(
            scl::kernel::entropy::normalized_mi(l1_arr, l2_arr, n_clusters1, n_clusters2)
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// KL Divergence
// =============================================================================

SCL_EXPORT scl_error_t scl_entropy_kl_divergence(
    const scl_real_t* p,
    const scl_real_t* q,
    scl_size_t n,
    int use_log2,
    scl_real_t* kl_out) {
    
    SCL_C_API_CHECK_NULL(p, "Distribution p array is null");
    SCL_C_API_CHECK_NULL(q, "Distribution q array is null");
    SCL_C_API_CHECK_NULL(kl_out, "Output KL divergence pointer is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");
    
    SCL_C_API_TRY
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p), n);
        Array<const Real> q_arr(reinterpret_cast<const Real*>(q), n);
        
        *kl_out = static_cast<scl_real_t>(
            scl::kernel::entropy::kl_divergence(p_arr, q_arr, use_log2 != 0)
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_entropy_js_divergence(
    const scl_real_t* p,
    const scl_real_t* q,
    scl_size_t n,
    int use_log2,
    scl_real_t* js_out) {
    
    SCL_C_API_CHECK_NULL(p, "Distribution p array is null");
    SCL_C_API_CHECK_NULL(q, "Distribution q array is null");
    SCL_C_API_CHECK_NULL(js_out, "Output JS divergence pointer is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");
    
    SCL_C_API_TRY
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p), n);
        Array<const Real> q_arr(reinterpret_cast<const Real*>(q), n);
        
        *js_out = static_cast<scl_real_t>(
            scl::kernel::entropy::js_divergence(p_arr, q_arr, use_log2 != 0)
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Cross Entropy
// =============================================================================

SCL_EXPORT scl_error_t scl_entropy_cross_entropy(
    const scl_real_t* p,
    const scl_real_t* q,
    const scl_size_t n,
    scl_real_t* ce_out) {
    
    SCL_C_API_CHECK_NULL(p, "Distribution p array is null");
    SCL_C_API_CHECK_NULL(q, "Distribution q array is null");
    SCL_C_API_CHECK_NULL(ce_out, "Output cross entropy pointer is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");
    
    SCL_C_API_TRY
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p), n);
        Array<const Real> q_arr(reinterpret_cast<const Real*>(q), n);
        
        *ce_out = static_cast<scl_real_t>(
            scl::kernel::entropy::cross_entropy(p_arr, q_arr)
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Conditional Entropy
// =============================================================================

SCL_EXPORT scl_error_t scl_entropy_conditional_entropy(
    const scl_index_t* x_binned,
    const scl_index_t* y_binned,
    scl_size_t n,
    scl_index_t n_bins_x,
    scl_index_t n_bins_y,
    int use_log2,
    scl_real_t* cond_entropy_out) {
    
    SCL_C_API_CHECK_NULL(x_binned, "X binned array is null");
    SCL_C_API_CHECK_NULL(y_binned, "Y binned array is null");
    SCL_C_API_CHECK_NULL(cond_entropy_out, "Output conditional entropy pointer is null");
    SCL_C_API_CHECK(n > 0 && n_bins_x > 0 && n_bins_y > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        *cond_entropy_out = static_cast<scl_real_t>(
            scl::kernel::entropy::conditional_entropy(
                x_binned, y_binned, n, n_bins_x, n_bins_y, use_log2 != 0
            )
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Additional Functions
// =============================================================================

SCL_EXPORT scl_error_t scl_entropy_symmetric_kl(
    const scl_real_t* p,
    const scl_real_t* q,
    scl_size_t n,
    int use_log2,
    scl_real_t* sym_kl_out) {
    
    SCL_C_API_CHECK_NULL(p, "Distribution p array is null");
    SCL_C_API_CHECK_NULL(q, "Distribution q array is null");
    SCL_C_API_CHECK_NULL(sym_kl_out, "Output symmetric KL pointer is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");
    
    SCL_C_API_TRY
        Array<const Real> p_arr(reinterpret_cast<const Real*>(p), n);
        Array<const Real> q_arr(reinterpret_cast<const Real*>(q), n);
        
        Real kl_pq = scl::kernel::entropy::kl_divergence(p_arr, q_arr, use_log2 != 0);
        Real kl_qp = scl::kernel::entropy::kl_divergence(q_arr, p_arr, use_log2 != 0);
        
        *sym_kl_out = static_cast<scl_real_t>((kl_pq + kl_qp) * Real(0.5));
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_entropy_joint_entropy(
    const scl_index_t* x_binned,
    const scl_index_t* y_binned,
    scl_size_t n,
    scl_index_t n_bins_x,
    scl_index_t n_bins_y,
    int use_log2,
    scl_real_t* joint_entropy_out) {
    
    SCL_C_API_CHECK_NULL(x_binned, "X binned array is null");
    SCL_C_API_CHECK_NULL(y_binned, "Y binned array is null");
    SCL_C_API_CHECK_NULL(joint_entropy_out, "Output joint entropy pointer is null");
    SCL_C_API_CHECK(n > 0 && n_bins_x > 0 && n_bins_y > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        *joint_entropy_out = static_cast<scl_real_t>(
            scl::kernel::entropy::joint_entropy(
                x_binned, y_binned, n, n_bins_x, n_bins_y, use_log2 != 0
            )
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_entropy_marginal_entropy(
    const scl_index_t* binned,
    scl_size_t n,
    scl_index_t n_bins,
    int use_log2,
    scl_real_t* marginal_entropy_out) {
    
    SCL_C_API_CHECK_NULL(binned, "Binned array is null");
    SCL_C_API_CHECK_NULL(marginal_entropy_out, "Output marginal entropy pointer is null");
    SCL_C_API_CHECK(n > 0 && n_bins > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        *marginal_entropy_out = static_cast<scl_real_t>(
            scl::kernel::entropy::marginal_entropy(binned, n, n_bins, use_log2 != 0)
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_entropy_adjusted_mi(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_size_t n,
    scl_index_t n_clusters1,
    scl_index_t n_clusters2,
    scl_real_t* ami_out) {
    
    SCL_C_API_CHECK_NULL(labels1, "Labels1 array is null");
    SCL_C_API_CHECK_NULL(labels2, "Labels2 array is null");
    SCL_C_API_CHECK_NULL(ami_out, "Output AMI pointer is null");
    SCL_C_API_CHECK(n > 0 && n_clusters1 > 0 && n_clusters2 > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        Array<const Index> l1_arr(labels1, n);
        Array<const Index> l2_arr(labels2, n);
        
        *ami_out = static_cast<scl_real_t>(
            scl::kernel::entropy::adjusted_mi(l1_arr, l2_arr, n_clusters1, n_clusters2)
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_entropy_discretize_equal_width(
    const scl_real_t* values,
    scl_size_t n,
    scl_index_t n_bins,
    scl_index_t* binned) {
    
    SCL_C_API_CHECK_NULL(values, "Values array is null");
    SCL_C_API_CHECK_NULL(binned, "Output binned array is null");
    SCL_C_API_CHECK(n > 0 && n_bins > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        scl::kernel::entropy::discretize_equal_width(
            reinterpret_cast<const Real*>(values), n, n_bins, binned
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_entropy_discretize_equal_frequency(
    const scl_real_t* values,
    scl_size_t n,
    scl_index_t n_bins,
    scl_index_t* binned) {
    
    SCL_C_API_CHECK_NULL(values, "Values array is null");
    SCL_C_API_CHECK_NULL(binned, "Output binned array is null");
    SCL_C_API_CHECK(n > 0 && n_bins > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        scl::kernel::entropy::discretize_equal_frequency(
            reinterpret_cast<const Real*>(values), n, n_bins, binned
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_entropy_gini_impurity(
    const scl_real_t* probabilities,
    scl_size_t n,
    scl_real_t* gini_out) {
    
    SCL_C_API_CHECK_NULL(probabilities, "Probabilities array is null");
    SCL_C_API_CHECK_NULL(gini_out, "Output Gini impurity pointer is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");
    
    SCL_C_API_TRY
        Array<const Real> probs_arr(reinterpret_cast<const Real*>(probabilities), n);
        
        *gini_out = static_cast<scl_real_t>(
            scl::kernel::entropy::gini_impurity(probs_arr)
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
