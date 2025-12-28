// =============================================================================
// FILE: scl/kernel/grn.h
// BRIEF: API reference for gene regulatory network inference
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::grn {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_CORRELATION_THRESHOLD = Real(0.3);
    constexpr Real EPSILON = Real(1e-15);
    constexpr Index DEFAULT_N_BINS = 10;
    constexpr Index DEFAULT_N_TREES = 100;
    constexpr Index DEFAULT_SUBSAMPLE = 500;
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size SIMD_THRESHOLD = 32;
    constexpr Size PREFETCH_DISTANCE = 16;
}

// =============================================================================
// GRN Method Types
// =============================================================================

enum class GRNMethod {
    Correlation,
    PartialCorrelation,
    MutualInformation,
    GENIE3,
    Combined
};

// =============================================================================
// GRN Inference
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: infer_grn
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Infer gene regulatory network from expression data.
 *
 * PARAMETERS:
 *     expression   [in]  Expression matrix (cells x genes, CSR)
 *     n_cells      [in]  Number of cells
 *     n_genes      [in]  Number of genes
 *     network      [out] GRN adjacency matrix [n_genes * n_genes]
 *     method       [in]  Inference method
 *     threshold    [in]  Correlation threshold
 *
 * PRECONDITIONS:
 *     - network has capacity >= n_genes * n_genes
 *
 * POSTCONDITIONS:
 *     - network[i * n_genes + j] contains edge weight from gene i to j
 *
 * COMPLEXITY:
 *     Time:  O(n_genes^2 * n_cells) for correlation
 *     Space: O(n_genes^2) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over gene pairs
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void infer_grn(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Index n_cells,                          // Number of cells
    Index n_genes,                          // Number of genes
    Real* network,                            // Output GRN [n_genes^2]
    GRNMethod method = GRNMethod::Correlation, // Inference method
    Real threshold = config::DEFAULT_CORRELATION_THRESHOLD // Correlation threshold
);

/* -----------------------------------------------------------------------------
 * FUNCTION: partial_correlation
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute partial correlation matrix (controlling for other genes).
 *
 * PARAMETERS:
 *     expression   [in]  Expression matrix (cells x genes, CSR)
 *     n_cells      [in]  Number of cells
 *     n_genes      [in]  Number of genes
 *     partial_corr [out] Partial correlation matrix [n_genes * n_genes]
 *
 * PRECONDITIONS:
 *     - partial_corr has capacity >= n_genes * n_genes
 *
 * POSTCONDITIONS:
 *     - partial_corr[i * n_genes + j] contains partial correlation
 *
 * COMPLEXITY:
 *     Time:  O(n_genes^3 * n_cells)
 *     Space: O(n_genes^2) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void partial_correlation(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Index n_cells,                          // Number of cells
    Index n_genes,                          // Number of genes
    Real* partial_corr                       // Output partial correlations [n_genes^2]
);

} // namespace scl::kernel::grn

