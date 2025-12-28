#pragma once

// =============================================================================
// FILE: scl/binding/c_api/entropy/entropy.h
// BRIEF: C API for information theory measures
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Entropy Measures
// =============================================================================

// Discrete entropy from probabilities
scl_error_t scl_entropy_discrete_entropy(
    const scl_real_t* probabilities,
    scl_size_t n,
    int use_log2,                     // 1 = log base 2, 0 = natural log
    scl_real_t* entropy_out
);

// Entropy from count data
scl_error_t scl_entropy_count_entropy(
    const scl_real_t* counts,
    scl_size_t n,
    int use_log2,
    scl_real_t* entropy_out
);

// Row entropy for sparse matrix
scl_error_t scl_entropy_row_entropy(
    scl_sparse_t X,
    scl_real_t* entropies,            // Output [n_rows]
    scl_size_t n_rows,
    int normalize,                    // 1 = normalize by max entropy
    int use_log2
);

// =============================================================================
// Divergence Measures
// =============================================================================

// KL divergence
scl_error_t scl_entropy_kl_divergence(
    const scl_real_t* p,
    const scl_real_t* q,
    scl_size_t n,
    int use_log2,
    scl_real_t* kl_out
);

// Jensen-Shannon divergence
scl_error_t scl_entropy_js_divergence(
    const scl_real_t* p,
    const scl_real_t* q,
    scl_size_t n,
    int use_log2,
    scl_real_t* js_out
);

// Symmetric KL divergence
scl_error_t scl_entropy_symmetric_kl(
    const scl_real_t* p,
    const scl_real_t* q,
    scl_size_t n,
    int use_log2,
    scl_real_t* sym_kl_out
);

// =============================================================================
// Mutual Information
// =============================================================================

// Mutual information I(X; Y)
scl_error_t scl_entropy_mutual_information(
    const scl_index_t* x_binned,      // Binned values for X
    const scl_index_t* y_binned,      // Binned values for Y
    scl_size_t n,
    scl_index_t n_bins_x,
    scl_index_t n_bins_y,
    int use_log2,
    scl_real_t* mi_out
);

// Joint entropy H(X, Y)
scl_error_t scl_entropy_joint_entropy(
    const scl_index_t* x_binned,
    const scl_index_t* y_binned,
    scl_size_t n,
    scl_index_t n_bins_x,
    scl_index_t n_bins_y,
    int use_log2,
    scl_real_t* joint_entropy_out
);

// Conditional entropy H(Y | X)
scl_error_t scl_entropy_conditional_entropy(
    const scl_index_t* x_binned,
    const scl_index_t* y_binned,
    scl_size_t n,
    scl_index_t n_bins_x,
    scl_index_t n_bins_y,
    int use_log2,
    scl_real_t* cond_entropy_out
);

// Marginal entropy from binned data
scl_error_t scl_entropy_marginal_entropy(
    const scl_index_t* binned,
    scl_size_t n,
    scl_index_t n_bins,
    int use_log2,
    scl_real_t* marginal_entropy_out
);

// =============================================================================
// Normalized Mutual Information
// =============================================================================

// Normalized mutual information (NMI)
scl_error_t scl_entropy_normalized_mi(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_size_t n,
    scl_index_t n_clusters1,
    scl_index_t n_clusters2,
    scl_real_t* nmi_out
);

// Adjusted mutual information (AMI)
scl_error_t scl_entropy_adjusted_mi(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_size_t n,
    scl_index_t n_clusters1,
    scl_index_t n_clusters2,
    scl_real_t* ami_out
);

// =============================================================================
// Discretization
// =============================================================================

// Equal-width discretization
scl_error_t scl_entropy_discretize_equal_width(
    const scl_real_t* values,
    scl_size_t n,
    scl_index_t n_bins,
    scl_index_t* binned              // Output bin indices
);

// Equal-frequency discretization
scl_error_t scl_entropy_discretize_equal_frequency(
    const scl_real_t* values,
    scl_size_t n,
    scl_index_t n_bins,
    scl_index_t* binned
);

// =============================================================================
// Other Measures
// =============================================================================

// Cross-entropy loss
scl_error_t scl_entropy_cross_entropy(
    const scl_real_t* true_probs,
    const scl_real_t* pred_probs,
    scl_size_t n,
    scl_real_t* cross_entropy_out
);

// Gini impurity
scl_error_t scl_entropy_gini_impurity(
    const scl_real_t* probabilities,
    scl_size_t n,
    scl_real_t* gini_out
);

#ifdef __cplusplus
}
#endif
