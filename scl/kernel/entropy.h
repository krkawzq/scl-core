// =============================================================================
// FILE: scl/kernel/entropy.h
// BRIEF: API reference for information theory measures for sparse data analysis
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::entropy {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real LOG_BASE_E = Real(2.718281828459045);
    constexpr Real LOG_2 = Real(0.693147180559945);
    constexpr Real INV_LOG_2 = Real(1.4426950408889634);
    constexpr Real EPSILON = Real(1e-15);
    constexpr Index DEFAULT_N_BINS = 10;
    constexpr Size PARALLEL_THRESHOLD = 128;
    constexpr Size SIMD_THRESHOLD = 16;
    constexpr size_t PREFETCH_DISTANCE = 64;
}

// =============================================================================
// Entropy Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: count_entropy
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Shannon entropy from count array.
 *
 * PARAMETERS:
 *     counts    [in]  Count values [n]
 *     n         [in]  Number of elements
 *     use_log2  [in]  If true, use log base 2
 *
 * PRECONDITIONS:
 *     - All counts >= 0
 *
 * POSTCONDITIONS:
 *     - Returns entropy H = -sum(p_i * log(p_i))
 *     - Returns 0 if total count is zero
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
template <typename T>
Real count_entropy(
    const T* counts,                        // Count values [n]
    Size n,                                  // Number of elements
    bool use_log2 = false                    // Use log base 2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: row_entropy
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Shannon entropy for each row of a sparse matrix.
 *
 * PARAMETERS:
 *     X         [in]  Sparse matrix (CSR or CSC)
 *     entropies [out] Entropy values [n_rows]
 *     normalize [in]  If true, normalize by maximum entropy
 *     use_log2  [in]  If true, use log base 2
 *
 * PRECONDITIONS:
 *     - entropies.len >= X.rows()
 *
 * POSTCONDITIONS:
 *     - entropies[i] contains entropy of row i
 *     - If normalize=true, values are in [0, 1]
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per row
 *
 * THREAD SAFETY:
 *     Safe - parallelized over rows
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void row_entropy(
    const Sparse<T, IsCSR>& X,              // Sparse matrix input
    Array<Real> entropies,                   // Output entropies [n_rows]
    bool normalize = false,                   // Normalize by max entropy
    bool use_log2 = false                    // Use log base 2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: kl_divergence
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Kullback-Leibler divergence between two probability distributions.
 *
 * PARAMETERS:
 *     p        [in]  First distribution [n]
 *     q        [in]  Second distribution [n]
 *     use_log2 [in]  If true, use log base 2
 *
 * PRECONDITIONS:
 *     - p.len == q.len
 *     - Both arrays represent probability distributions
 *
 * POSTCONDITIONS:
 *     - Returns KL(p || q) = sum(p_i * log(p_i / q_i))
 *     - Returns large value if q_i = 0 and p_i > 0
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
Real kl_divergence(
    Array<const Real> p,                     // First distribution [n]
    Array<const Real> q,                      // Second distribution [n]
    bool use_log2 = false                    // Use log base 2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: js_divergence
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Jensen-Shannon divergence between two probability distributions.
 *
 * PARAMETERS:
 *     p        [in]  First distribution [n]
 *     q        [in]  Second distribution [n]
 *     use_log2 [in]  If true, use log base 2
 *
 * PRECONDITIONS:
 *     - p.len == q.len
 *     - Both arrays represent probability distributions
 *
 * POSTCONDITIONS:
 *     - Returns JS(p || q) = 0.5 * KL(p || m) + 0.5 * KL(q || m) where m = (p+q)/2
 *     - Always finite and symmetric
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
Real js_divergence(
    Array<const Real> p,                     // First distribution [n]
    Array<const Real> q,                      // Second distribution [n]
    bool use_log2 = false                    // Use log base 2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: discretize_equal_width
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Discretize continuous values into equal-width bins.
 *
 * PARAMETERS:
 *     values  [in]  Continuous values [n]
 *     n       [in]  Number of values
 *     n_bins  [in]  Number of bins
 *     binned  [out] Binned indices [n]
 *
 * PRECONDITIONS:
 *     - binned has capacity >= n
 *     - n_bins > 0
 *
 * POSTCONDITIONS:
 *     - binned[i] contains bin index in [0, n_bins-1]
 *     - All values in same bin have same range
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over values
 * -------------------------------------------------------------------------- */
template <typename T>
void discretize_equal_width(
    const T* values,                         // Continuous values [n]
    Size n,                                   // Number of values
    Index n_bins,                            // Number of bins
    Index* binned                             // Output bin indices [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: discretize_equal_frequency
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Discretize continuous values into equal-frequency bins.
 *
 * PARAMETERS:
 *     values  [in]  Continuous values [n]
 *     n       [in]  Number of values
 *     n_bins  [in]  Number of bins
 *     binned  [out] Binned indices [n]
 *
 * PRECONDITIONS:
 *     - binned has capacity >= n
 *     - n_bins > 0
 *
 * POSTCONDITIONS:
 *     - binned[i] contains bin index in [0, n_bins-1]
 *     - Each bin contains approximately n/n_bins values
 *
 * COMPLEXITY:
 *     Time:  O(n log n) for sorting
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - requires sorting
 * -------------------------------------------------------------------------- */
template <typename T>
void discretize_equal_frequency(
    const T* values,                         // Continuous values [n]
    Size n,                                   // Number of values
    Index n_bins,                            // Number of bins
    Index* binned                             // Output bin indices [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: histogram_2d
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute 2D histogram from binned data.
 *
 * PARAMETERS:
 *     x_binned  [in]  Binned x values [n]
 *     y_binned  [in]  Binned y values [n]
 *     n         [in]  Number of samples
 *     n_bins_x  [in]  Number of x bins
 *     n_bins_y  [in]  Number of y bins
 *     counts    [out] Histogram counts [n_bins_x * n_bins_y]
 *
 * PRECONDITIONS:
 *     - counts has capacity >= n_bins_x * n_bins_y
 *
 * POSTCONDITIONS:
 *     - counts[i * n_bins_y + j] contains count for bin (i, j)
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(n_bins_x * n_bins_y) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized with atomic accumulation
 * -------------------------------------------------------------------------- */
void histogram_2d(
    const Index* x_binned,                   // Binned x values [n]
    const Index* y_binned,                   // Binned y values [n]
    Size n,                                   // Number of samples
    Index n_bins_x,                          // Number of x bins
    Index n_bins_y,                          // Number of y bins
    Size* counts                              // Output histogram [n_bins_x * n_bins_y]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: joint_entropy
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute joint entropy H(X, Y) from binned data.
 *
 * PARAMETERS:
 *     x_binned  [in]  Binned x values [n]
 *     y_binned  [in]  Binned y values [n]
 *     n         [in]  Number of samples
 *     n_bins_x  [in]  Number of x bins
 *     n_bins_y  [in]  Number of y bins
 *     use_log2  [in]  If true, use log base 2
 *
 * PRECONDITIONS:
 *     - All bin indices are valid
 *
 * POSTCONDITIONS:
 *     - Returns H(X, Y) = -sum(p_ij * log(p_ij))
 *
 * COMPLEXITY:
 *     Time:  O(n + n_bins_x * n_bins_y)
 *     Space: O(n_bins_x * n_bins_y) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - uses parallel histogram
 * -------------------------------------------------------------------------- */
Real joint_entropy(
    const Index* x_binned,                   // Binned x values [n]
    const Index* y_binned,                   // Binned y values [n]
    Size n,                                   // Number of samples
    Index n_bins_x,                          // Number of x bins
    Index n_bins_y,                          // Number of y bins
    bool use_log2 = false                    // Use log base 2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: marginal_entropy
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute marginal entropy H(X) from binned data.
 *
 * PARAMETERS:
 *     binned   [in]  Binned values [n]
 *     n        [in]  Number of samples
 *     n_bins   [in]  Number of bins
 *     use_log2 [in]  If true, use log base 2
 *
 * PRECONDITIONS:
 *     - All bin indices are valid
 *
 * POSTCONDITIONS:
 *     - Returns H(X) = -sum(p_i * log(p_i))
 *
 * COMPLEXITY:
 *     Time:  O(n + n_bins)
 *     Space: O(n_bins) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - uses parallel histogram
 * -------------------------------------------------------------------------- */
Real marginal_entropy(
    const Index* binned,                     // Binned values [n]
    Size n,                                   // Number of samples
    Index n_bins,                            // Number of bins
    bool use_log2 = false                    // Use log base 2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: conditional_entropy
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute conditional entropy H(Y | X) from binned data.
 *
 * PARAMETERS:
 *     x_binned  [in]  Binned x values [n]
 *     y_binned  [in]  Binned y values [n]
 *     n         [in]  Number of samples
 *     n_bins_x  [in]  Number of x bins
 *     n_bins_y  [in]  Number of y bins
 *     use_log2  [in]  If true, use log base 2
 *
 * PRECONDITIONS:
 *     - All bin indices are valid
 *
 * POSTCONDITIONS:
 *     - Returns H(Y | X) = H(X, Y) - H(X)
 *
 * COMPLEXITY:
 *     Time:  O(n + n_bins_x * n_bins_y)
 *     Space: O(n_bins_x * n_bins_y) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - uses parallel operations
 * -------------------------------------------------------------------------- */
Real conditional_entropy(
    const Index* x_binned,                   // Binned x values [n]
    const Index* y_binned,                   // Binned y values [n]
    Size n,                                   // Number of samples
    Index n_bins_x,                          // Number of x bins
    Index n_bins_y,                          // Number of y bins
    bool use_log2 = false                    // Use log base 2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: mutual_information
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute mutual information I(X; Y) from binned data.
 *
 * PARAMETERS:
 *     x_binned  [in]  Binned x values [n]
 *     y_binned  [in]  Binned y values [n]
 *     n         [in]  Number of samples
 *     n_bins_x  [in]  Number of x bins
 *     n_bins_y  [in]  Number of y bins
 *     use_log2  [in]  If true, use log base 2
 *
 * PRECONDITIONS:
 *     - All bin indices are valid
 *
 * POSTCONDITIONS:
 *     - Returns I(X; Y) = H(X) + H(Y) - H(X, Y)
 *     - Always >= 0
 *
 * COMPLEXITY:
 *     Time:  O(n + n_bins_x * n_bins_y)
 *     Space: O(n_bins_x * n_bins_y) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - uses parallel operations
 * -------------------------------------------------------------------------- */
Real mutual_information(
    const Index* x_binned,                   // Binned x values [n]
    const Index* y_binned,                   // Binned y values [n]
    Size n,                                   // Number of samples
    Index n_bins_x,                          // Number of x bins
    Index n_bins_y,                          // Number of y bins
    bool use_log2 = false                    // Use log base 2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: normalized_mi
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute normalized mutual information between two labelings.
 *
 * PARAMETERS:
 *     labels1      [in]  First labeling [n]
 *     labels2      [in]  Second labeling [n]
 *     n_clusters1  [in]  Number of clusters in first labeling
 *     n_clusters2  [in]  Number of clusters in second labeling
 *
 * PRECONDITIONS:
 *     - labels1.len == labels2.len
 *     - All label indices are valid
 *
 * POSTCONDITIONS:
 *     - Returns NMI = 2 * I(X;Y) / (H(X) + H(Y))
 *     - Values in [0, 1], where 1 indicates perfect agreement
 *
 * COMPLEXITY:
 *     Time:  O(n + n_clusters1 * n_clusters2)
 *     Space: O(n_clusters1 * n_clusters2) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - uses parallel operations
 * -------------------------------------------------------------------------- */
Real normalized_mi(
    Array<const Index> labels1,             // First labeling [n]
    Array<const Index> labels2,              // Second labeling [n]
    Index n_clusters1,                       // Number of clusters in labels1
    Index n_clusters2                        // Number of clusters in labels2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: adjusted_mi
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute adjusted mutual information (corrected for chance).
 *
 * PARAMETERS:
 *     labels1      [in]  First labeling [n]
 *     labels2      [in]  Second labeling [n]
 *     n_clusters1  [in]  Number of clusters in first labeling
 *     n_clusters2  [in]  Number of clusters in second labeling
 *
 * PRECONDITIONS:
 *     - labels1.len == labels2.len
 *     - All label indices are valid
 *
 * POSTCONDITIONS:
 *     - Returns AMI = (MI - E[MI]) / (max(H1, H2) - E[MI])
 *     - Values in [-1, 1], where 1 indicates perfect agreement
 *
 * COMPLEXITY:
 *     Time:  O(n + n_clusters1 * n_clusters2)
 *     Space: O(n_clusters1 * n_clusters2) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - uses parallel operations
 * -------------------------------------------------------------------------- */
Real adjusted_mi(
    Array<const Index> labels1,             // First labeling [n]
    Array<const Index> labels2,              // Second labeling [n]
    Index n_clusters1,                       // Number of clusters in labels1
    Index n_clusters2                        // Number of clusters in labels2
);

/* -----------------------------------------------------------------------------
 * FUNCTION: select_features_mi
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Select top features using mutual information with target.
 *
 * PARAMETERS:
 *     X                [in]  Feature matrix (CSR or CSC)
 *     target           [in]  Target labels [n_samples]
 *     n_features       [in]  Total number of features
 *     n_to_select      [in]  Number of features to select
 *     selected_features [out] Selected feature indices [n_to_select]
 *     mi_scores        [out] MI scores for all features [n_features]
 *     n_bins           [in]  Number of bins for discretization
 *
 * PRECONDITIONS:
 *     - selected_features has capacity >= n_to_select
 *     - mi_scores has capacity >= n_features
 *
 * POSTCONDITIONS:
 *     - selected_features contains top n_to_select features by MI
 *     - mi_scores contains MI score for each feature
 *
 * COMPLEXITY:
 *     Time:  O(n_features * n_samples * log(nnz_per_sample))
 *     Space: O(n_samples) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential feature processing
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void select_features_mi(
    const Sparse<T, IsCSR>& X,              // Feature matrix
    Array<const Index> target,               // Target labels [n_samples]
    Index n_features,                        // Total number of features
    Index n_to_select,                       // Number to select
    Array<Index> selected_features,          // Output selected features [n_to_select]
    Array<Real> mi_scores,                   // Output MI scores [n_features]
    Index n_bins = config::DEFAULT_N_BINS    // Discretization bins
);

/* -----------------------------------------------------------------------------
 * FUNCTION: mrmr_selection
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Select features using minimum Redundancy Maximum Relevance (mRMR).
 *
 * PARAMETERS:
 *     X                [in]  Feature matrix (CSR or CSC)
 *     target           [in]  Target labels [n_samples]
 *     n_features       [in]  Total number of features
 *     n_to_select      [in]  Number of features to select
 *     selected_features [out] Selected feature indices [n_to_select]
 *     n_bins           [in]  Number of bins for discretization
 *
 * PRECONDITIONS:
 *     - selected_features has capacity >= n_to_select
 *
 * POSTCONDITIONS:
 *     - selected_features contains mRMR-selected features
 *     - Features maximize relevance and minimize redundancy
 *
 * COMPLEXITY:
 *     Time:  O(n_to_select * n_features * n_samples)
 *     Space: O(n_features * n_samples) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential greedy selection
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void mrmr_selection(
    const Sparse<T, IsCSR>& X,              // Feature matrix
    Array<const Index> target,               // Target labels [n_samples]
    Index n_features,                         // Total number of features
    Index n_to_select,                       // Number to select
    Array<Index> selected_features,          // Output selected features [n_to_select]
    Index n_bins = config::DEFAULT_N_BINS    // Discretization bins
);

} // namespace scl::kernel::entropy

