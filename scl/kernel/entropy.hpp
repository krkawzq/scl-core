#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/simd.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <vector>

// =============================================================================
// FILE: scl/kernel/entropy.hpp
// BRIEF: Information theory measures for sparse data
//
// STRATEGIC POSITION: Sparse + Nonlinear (Tier 4 - Advanced)
// - Nonlinear information measures
// - Sparse-aware computation
// - Feature selection and clustering evaluation
//
// APPLICATIONS:
// - Feature selection (mutual information)
// - Clustering quality (entropy)
// - Distribution comparison (KL/JS divergence)
// - Gene-gene correlation (MI)
//
// KEY CHALLENGE:
// Entropy estimation from sparse count data
// Handle zero counts appropriately
// =============================================================================

namespace scl::kernel::entropy {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real LOG_BASE = M_E;          // Natural log (can be 2 for bits)
    constexpr Real EPSILON = Real(1e-12);
    constexpr Index DEFAULT_N_BINS = 10;    // For discretization
}

// =============================================================================
// Shannon Entropy (TODO: Implementation)
// =============================================================================

// TODO: Compute Shannon entropy from sparse count data
template <typename T, bool IsCSR>
void shannon_entropy(
    const Sparse<T, IsCSR>& X,
    Array<Real> entropies,                  // Per row/column
    bool normalize = false                   // Divide by log(n) for [0,1]
);

// TODO: Entropy of discrete distribution
Real discrete_entropy(
    Array<const Real> probabilities
);

// TODO: Entropy from count vector (sparse)
template <typename T>
Real sparse_entropy(
    const T* values,
    Size nnz,
    Size total_count
);

// =============================================================================
// Mutual Information (TODO: Implementation)
// =============================================================================

// TODO: Mutual information between two variables
// Requires discretization for continuous data
template <typename T, bool IsCSR>
void mutual_information(
    const Sparse<T, IsCSR>& X,
    const Sparse<T, IsCSR>& Y,
    Sparse<Real, IsCSR>& mi_matrix          // Pairwise MI
);

// TODO: Conditional mutual information I(X;Y|Z)
template <typename T>
Real conditional_mi(
    Array<const T> x,
    Array<const T> y,
    Array<const T> z
);

// TODO: Normalized mutual information (for clustering evaluation)
Real normalized_mi(
    Array<const Index> labels1,
    Array<const Index> labels2,
    Index n_clusters1,
    Index n_clusters2
);

// TODO: Adjusted mutual information (chance-corrected)
Real adjusted_mi(
    Array<const Index> labels1,
    Array<const Index> labels2,
    Index n_clusters1,
    Index n_clusters2
);

// =============================================================================
// Divergence Measures (TODO: Implementation)
// =============================================================================

// TODO: Kullback-Leibler divergence D(P||Q)
template <typename T>
Real kl_divergence(
    const T* p_values,
    const T* q_values,
    Size n
);

// TODO: Sparse KL divergence (for count data)
template <typename T>
Real sparse_kl_divergence(
    const T* p_values,
    const Index* p_indices,
    Size p_nnz,
    const T* q_values,
    const Index* q_indices,
    Size q_nnz,
    Size dimension
);

// TODO: Jensen-Shannon divergence (symmetric)
template <typename T>
Real js_divergence(
    const T* p_values,
    const T* q_values,
    Size n
);

// TODO: Symmetric KL divergence (D(P||Q) + D(Q||P)) / 2
template <typename T>
Real symmetric_kl(
    const T* p_values,
    const T* q_values,
    Size n
);

// =============================================================================
// Joint and Conditional Entropy (TODO: Implementation)
// =============================================================================

// TODO: Joint entropy H(X,Y)
template <typename T>
Real joint_entropy(
    Array<const T> x,
    Array<const T> y,
    Index n_bins_x,
    Index n_bins_y
);

// TODO: Conditional entropy H(Y|X)
template <typename T>
Real conditional_entropy(
    Array<const T> y,
    Array<const T> x,
    Index n_bins_y,
    Index n_bins_x
);

// =============================================================================
// Feature Selection (TODO: Implementation)
// =============================================================================

// TODO: Select features by mutual information with target
template <typename T, bool IsCSR>
void select_features_mi(
    const Sparse<T, IsCSR>& X,
    Array<const Index> target,
    Index n_features_to_select,
    std::vector<Index>& selected_features,
    std::vector<Real>& mi_scores
);

// TODO: mRMR feature selection (minimum Redundancy Maximum Relevance)
template <typename T, bool IsCSR>
void mrmr_selection(
    const Sparse<T, IsCSR>& X,
    Array<const Index> target,
    Index n_features,
    std::vector<Index>& selected_features
);

// =============================================================================
// Discretization (TODO: Implementation)
// =============================================================================

// TODO: Equal-width binning
template <typename T>
void discretize_equal_width(
    const T* values,
    Size n,
    Index n_bins,
    Index* binned
);

// TODO: Equal-frequency binning (quantile-based)
template <typename T>
void discretize_equal_frequency(
    const T* values,
    Size n,
    Index n_bins,
    Index* binned
);

// TODO: Discretize sparse matrix
template <typename T, bool IsCSR>
void discretize_sparse(
    const Sparse<T, IsCSR>& X,
    Index n_bins,
    Sparse<Index, IsCSR>& X_binned
);

// =============================================================================
// Helper Functions (TODO: Implementation)
// =============================================================================

namespace detail {

// TODO: Compute 2D histogram (for joint distributions)
template <typename T>
void histogram_2d(
    const T* x,
    const T* y,
    Size n,
    Index n_bins_x,
    Index n_bins_y,
    std::vector<Size>& counts          // Flattened 2D histogram
);

// TODO: Estimate probability from counts
void counts_to_probabilities(
    const Size* counts,
    Size n_bins,
    Real* probabilities
);

// TODO: Compute log with zero handling
SCL_FORCE_INLINE Real safe_log(Real x);

// TODO: SIMD entropy computation
void entropy_simd(
    const Real* probabilities,
    Size n,
    Real& entropy
);

} // namespace detail

} // namespace scl::kernel::entropy

