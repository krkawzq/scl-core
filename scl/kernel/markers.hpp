#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <vector>
#include <algorithm>

// =============================================================================
// FILE: scl/kernel/markers.hpp
// BRIEF: Marker gene selection and specificity scoring
//
// STRATEGIC POSITION: Sparse + Nonlinear (Tier 3 - Biology-Specific)
// - Combines existing statistical tests
// - Sparse-aware ranking and scoring
// - Core single-cell analysis task
//
// APPLICATIONS:
// - Identify marker genes for cell types
// - Compute gene specificity scores
// - One-vs-rest differential expression
// - Marker overlap analysis
//
// METHODS:
// - Rank-based selection (fold-change, effect size)
// - Specificity scores (Tau, Gini coefficient)
// - Statistical significance (p-values from ttest/mwu)
// =============================================================================

namespace scl::kernel::markers {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_MIN_FC = Real(1.5);      // Minimum fold change
    constexpr Real DEFAULT_MIN_PCT = Real(0.1);     // Minimum expression %
    constexpr Real DEFAULT_MAX_PVAL = Real(0.05);   // Maximum p-value
}

// =============================================================================
// Ranking Methods
// =============================================================================

enum class RankingMethod {
    FOLD_CHANGE,      // Log fold change
    EFFECT_SIZE,      // Cohen's d or similar
    P_VALUE,          // Statistical significance
    COMBINED          // Combined score (FC * -log10(p))
};

// =============================================================================
// Core Functions (TODO: Implementation)
// =============================================================================

// TODO: Rank genes for each cluster (one-vs-rest)
template <typename T, bool IsCSR>
void rank_genes_groups(
    const Sparse<T, IsCSR>& X,
    Array<const Index> group_labels,
    Index n_groups,
    RankingMethod method,
    std::vector<std::vector<Index>>& ranked_genes,  // Per group
    std::vector<std::vector<Real>>& scores           // Per group
);

// TODO: Compute gene specificity (Tau score)
// Tau = sum((1 - x_i/x_max)) / (N - 1)
// where x_i is expression in tissue i, x_max is max expression
template <typename T, bool IsCSR>
void tau_specificity(
    const Sparse<T, IsCSR>& X,              // Genes × Groups
    Array<Real> tau_scores                   // Output per gene
);

// TODO: Compute Gini coefficient for gene specificity
template <typename T, bool IsCSR>
void gini_specificity(
    const Sparse<T, IsCSR>& X,
    Array<Real> gini_scores
);

// TODO: One-vs-rest comparison
template <typename T, bool IsCSR>
void one_vs_rest(
    const Sparse<T, IsCSR>& X,
    Array<const Index> group_labels,
    Index target_group,
    Array<Real> fold_changes,
    Array<Real> p_values
);

// =============================================================================
// Advanced Functions (TODO: Implementation)
// =============================================================================

// TODO: Marker overlap analysis
// Compute Jaccard similarity of marker sets between groups
void marker_overlap(
    const std::vector<std::vector<Index>>& marker_sets,
    Sparse<Real, true>& overlap_matrix    // Symmetric matrix
);

// TODO: Filter markers by criteria
void filter_markers(
    const std::vector<Index>& candidate_genes,
    Array<const Real> fold_changes,
    Array<const Real> p_values,
    Array<const Real> pct_expressed,
    Real min_fc,
    Real max_pval,
    Real min_pct,
    std::vector<Index>& filtered_genes
);

// TODO: Top N markers per group
void top_n_markers_per_group(
    const std::vector<std::vector<Index>>& ranked_genes,
    const std::vector<std::vector<Real>>& scores,
    Index n_top,
    std::vector<std::vector<Index>>& top_markers
);

// TODO: Unique markers (non-overlapping across groups)
void unique_markers(
    const std::vector<std::vector<Index>>& marker_sets,
    std::vector<std::vector<Index>>& unique_sets
);

// =============================================================================
// Expression Statistics (TODO: Implementation)
// =============================================================================

// TODO: Compute percentage of cells expressing each gene
template <typename T, bool IsCSR>
void percent_expressed(
    const Sparse<T, IsCSR>& X,
    Array<const Index> group_labels,
    Index n_groups,
    Sparse<Real, !IsCSR>& pct_matrix     // Genes × Groups
);

// TODO: Compute fold change (log)
template <typename T, bool IsCSR>
void log_fold_change(
    const Sparse<T, IsCSR>& X,
    Array<const Index> group_labels,
    Index target_group,
    Index reference_group,
    Array<Real> log_fc
);

// TODO: Compute mean expression per group
template <typename T, bool IsCSR>
void group_mean_expression(
    const Sparse<T, IsCSR>& X,
    Array<const Index> group_labels,
    Index n_groups,
    Sparse<Real, !IsCSR>& mean_expression  // Genes × Groups
);

// =============================================================================
// Helper Functions (TODO: Implementation)
// =============================================================================

namespace detail {

// TODO: Compute Cohen's d effect size
Real cohens_d(
    Real mean1,
    Real mean2,
    Real var1,
    Real var2,
    Size n1,
    Size n2
);

// TODO: Compute combined score (FC * -log10(p))
Real combined_score(
    Real fold_change,
    Real p_value
);

// TODO: Sort genes by score (descending)
void sort_by_score(
    const std::vector<Index>& genes,
    const std::vector<Real>& scores,
    std::vector<Index>& sorted_genes,
    std::vector<Real>& sorted_scores
);

// TODO: Compute Gini coefficient from sorted values
Real gini_coefficient(
    const Real* values,
    Size n
);

} // namespace detail

} // namespace scl::kernel::markers

