#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/threading/parallel_for.hpp"

#include <vector>
#include <algorithm>

// =============================================================================
// FILE: scl/kernel/scoring.hpp
// BRIEF: Gene set scoring and cell signature analysis
//
// STRATEGIC POSITION: Sparse + Nonlinear (Tier 3 - Biology-Specific)
// - Efficient scoring on sparse matrices
// - Nonlinear aggregation (ranks, quantiles)
// - Core functional annotation task
//
// APPLICATIONS:
// - Gene set enrichment scoring (AUCell-like)
// - Module scoring (Seurat-style)
// - Cell cycle phase scoring
// - Pathway activity scoring
// - Custom signature scoring
//
// METHODS:
// - Rank-based scoring (AUCell approach)
// - Mean expression scoring
// - Weighted scoring
// - Control gene background correction
// =============================================================================

namespace scl::kernel::scoring {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size DEFAULT_N_CONTROL = 100;         // Control genes
    constexpr Size DEFAULT_N_BINS = 25;             // Expression bins
    constexpr Real DEFAULT_QUANTILE = Real(0.05);   // AUC quantile
}

// =============================================================================
// Scoring Methods
// =============================================================================

enum class ScoringMethod {
    MEAN,             // Simple mean of gene set
    RANK_BASED,       // AUCell-like rank-based
    WEIGHTED,         // Weighted by gene importance
    SEURAT_MODULE,    // Seurat module score with control
    ZSCORE            // Z-score normalized
};

// =============================================================================
// Core Functions (TODO: Implementation)
// =============================================================================

// TODO: Gene set scoring (generic interface)
template <typename T, bool IsCSR>
void gene_set_score(
    const Sparse<T, IsCSR>& X,
    Array<const Index> gene_set,
    ScoringMethod method,
    Array<Real> scores                      // Output: score per cell
);

// TODO: AUCell-like rank-based scoring
template <typename T, bool IsCSR>
void auc_score(
    const Sparse<T, IsCSR>& X,
    Array<const Index> gene_set,
    Array<Real> scores,
    Real quantile = config::DEFAULT_QUANTILE
);

// TODO: Seurat module score with control genes
template <typename T, bool IsCSR>
void module_score(
    const Sparse<T, IsCSR>& X,
    Array<const Index> gene_set,
    Array<Real> scores,
    Size n_control = config::DEFAULT_N_CONTROL,
    Size n_bins = config::DEFAULT_N_BINS
);

// TODO: Cell cycle scoring (S and G2M scores)
template <typename T, bool IsCSR>
void cell_cycle_score(
    const Sparse<T, IsCSR>& X,
    Array<const Index> s_genes,
    Array<const Index> g2m_genes,
    Array<Real> s_scores,
    Array<Real> g2m_scores,
    Array<Index> phase_labels              // Output: G1, S, G2M
);

// TODO: Multi-signature scoring (multiple gene sets at once)
template <typename T, bool IsCSR>
void multi_signature_score(
    const Sparse<T, IsCSR>& X,
    const std::vector<std::vector<Index>>& gene_sets,
    ScoringMethod method,
    std::vector<Array<Real>>& scores       // One per signature
);

// =============================================================================
// Advanced Functions (TODO: Implementation)
// =============================================================================

// TODO: Weighted gene set scoring
template <typename T, bool IsCSR>
void weighted_score(
    const Sparse<T, IsCSR>& X,
    Array<const Index> gene_set,
    Array<const Real> gene_weights,
    Array<Real> scores
);

// TODO: Differential scoring (signature A - signature B)
template <typename T, bool IsCSR>
void differential_score(
    const Sparse<T, IsCSR>& X,
    Array<const Index> positive_genes,
    Array<const Index> negative_genes,
    Array<Real> scores
);

// TODO: Quantile-based scoring
template <typename T, bool IsCSR>
void quantile_score(
    const Sparse<T, IsCSR>& X,
    Array<const Index> gene_set,
    Real quantile,
    Array<Real> scores
);

// TODO: Competitive gene set scoring (gene set vs background)
template <typename T, bool IsCSR>
void competitive_score(
    const Sparse<T, IsCSR>& X,
    Array<const Index> gene_set,
    Array<Real> scores,
    Real& enrichment_pvalue
);

// =============================================================================
// Control Gene Selection (TODO: Implementation)
// =============================================================================

// TODO: Select control genes matched by expression level
template <typename T, bool IsCSR>
void select_control_genes(
    const Sparse<T, IsCSR>& X,
    Array<const Index> gene_set,
    Size n_control,
    Size n_bins,
    std::vector<Index>& control_genes
);

// TODO: Bin genes by mean expression
template <typename T, bool IsCSR>
void bin_genes_by_expression(
    const Sparse<T, IsCSR>& X,
    Size n_bins,
    std::vector<std::vector<Index>>& bins
);

// =============================================================================
// Helper Functions (TODO: Implementation)
// =============================================================================

namespace detail {

// TODO: Rank expression values within each cell
template <typename T>
void rank_expression(
    const T* values,
    const Index* indices,
    Size nnz,
    Size n_features,
    Real* ranks                            // Output: dense ranks
);

// TODO: Compute AUC from ranks
Real compute_auc(
    const Real* ranks,
    const Index* gene_set,
    Size n_genes_in_set,
    Size n_total_genes,
    Real quantile
);

// TODO: Z-score normalization per cell
template <typename T>
void zscore_normalize(
    const T* values,
    Size n,
    Real* normalized
);

// TODO: Sample control genes from bin
void sample_from_bin(
    const std::vector<Index>& bin,
    const std::vector<Index>& exclude,
    Size n_samples,
    std::vector<Index>& sampled,
    uint64_t seed
);

} // namespace detail

} // namespace scl::kernel::scoring

