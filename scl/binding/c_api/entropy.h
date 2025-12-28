#pragma once

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// FILE: scl/binding/c_api/entropy.h
// BRIEF: C API for information theory measures
// =============================================================================

// Discrete entropy
scl_real_t scl_entropy_discrete_entropy(
    const scl_real_t* probabilities,
    scl_size_t n_probs,
    int use_log2
);

// Entropy from count data
scl_real_t scl_entropy_count_entropy(
    const scl_index_t* counts,
    scl_size_t n_counts,
    int use_log2
);

// Row entropy
scl_error_t scl_entropy_row_entropy(
    scl_sparse_matrix_t X,
    scl_real_t* entropies,
    scl_size_t n_rows,
    int normalize,
    int use_log2
);

// KL divergence
scl_real_t scl_entropy_kl_divergence(
    const scl_real_t* p,
    const scl_real_t* q,
    scl_size_t n,
    int use_log2
);

// JS divergence
scl_real_t scl_entropy_js_divergence(
    const scl_real_t* p,
    const scl_real_t* q,
    scl_size_t n,
    int use_log2
);

// Symmetric KL divergence
scl_real_t scl_entropy_symmetric_kl(
    const scl_real_t* p,
    const scl_real_t* q,
    scl_size_t n,
    int use_log2
);

// Equal-width discretization
scl_error_t scl_entropy_discretize_equal_width(
    const scl_real_t* values,
    scl_size_t n,
    scl_index_t n_bins,
    scl_index_t* binned
);

// Equal-frequency discretization
scl_error_t scl_entropy_discretize_equal_frequency(
    const scl_real_t* values,
    scl_size_t n,
    scl_index_t n_bins,
    scl_index_t* binned
);

// 2D histogram
scl_error_t scl_entropy_histogram_2d(
    const scl_index_t* x_binned,
    const scl_index_t* y_binned,
    scl_size_t n,
    scl_index_t n_bins_x,
    scl_index_t n_bins_y,
    scl_size_t* counts
);

// Joint entropy
scl_real_t scl_entropy_joint_entropy(
    const scl_index_t* x_binned,
    const scl_index_t* y_binned,
    scl_size_t n,
    scl_index_t n_bins_x,
    scl_index_t n_bins_y,
    int use_log2
);

// Marginal entropy
scl_real_t scl_entropy_marginal_entropy(
    const scl_index_t* binned,
    scl_size_t n,
    scl_index_t n_bins,
    int use_log2
);

// Conditional entropy
scl_real_t scl_entropy_conditional_entropy(
    const scl_index_t* x_binned,
    const scl_index_t* y_binned,
    scl_size_t n,
    scl_index_t n_bins_x,
    scl_index_t n_bins_y,
    int use_log2
);

// Mutual information
scl_real_t scl_entropy_mutual_information(
    const scl_index_t* x_binned,
    const scl_index_t* y_binned,
    scl_size_t n,
    scl_index_t n_bins_x,
    scl_index_t n_bins_y,
    int use_log2
);

// Normalized mutual information
scl_real_t scl_entropy_normalized_mi(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_size_t n,
    scl_index_t n_clusters1,
    scl_index_t n_clusters2
);

// Adjusted mutual information
scl_real_t scl_entropy_adjusted_mi(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_size_t n,
    scl_index_t n_clusters1,
    scl_index_t n_clusters2
);

// Feature selection by mutual information
scl_error_t scl_entropy_select_features_mi(
    scl_sparse_matrix_t X,
    const scl_index_t* target,
    scl_size_t n_samples,
    scl_index_t n_features,
    scl_index_t n_to_select,
    scl_index_t* selected_features,
    scl_real_t* mi_scores,
    scl_index_t n_bins
);

// mRMR feature selection
scl_error_t scl_entropy_mrmr_selection(
    scl_sparse_matrix_t X,
    const scl_index_t* target,
    scl_size_t n_samples,
    scl_index_t n_features,
    scl_index_t n_to_select,
    scl_index_t* selected_features,
    scl_index_t n_bins
);

// Cross-entropy loss
scl_real_t scl_entropy_cross_entropy(
    const scl_real_t* true_probs,
    const scl_real_t* pred_probs,
    scl_size_t n
);

// Perplexity
scl_real_t scl_entropy_perplexity(
    const scl_real_t* true_probs,
    const scl_real_t* pred_probs,
    scl_size_t n
);

// Information gain
scl_real_t scl_entropy_information_gain(
    const scl_index_t* feature_binned,
    const scl_index_t* target,
    scl_size_t n,
    scl_index_t n_feature_bins,
    scl_index_t n_target_classes
);

// Gini impurity
scl_real_t scl_entropy_gini_impurity(
    const scl_real_t* probabilities,
    scl_size_t n
);

#ifdef __cplusplus
}
#endif
