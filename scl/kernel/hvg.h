// =============================================================================
// FILE: scl/kernel/hvg.h
// BRIEF: API reference for highly variable gene selection
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::hvg {

/* -----------------------------------------------------------------------------
 * FUNCTION: select_by_dispersion
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Select highly variable genes by dispersion (variance/mean ratio).
 *
 * PARAMETERS:
 *     matrix          [in]  Sparse expression matrix (genes x cells or cells x genes)
 *     n_top           [in]  Number of top genes to select
 *     out_indices     [out] Indices of selected genes [n_top]
 *     out_mask        [out] Binary mask (1 = selected) [n_genes]
 *     out_dispersions [out] Dispersion values [n_genes]
 *
 * PRECONDITIONS:
 *     - out_indices.len >= n_top
 *     - out_mask.len >= n_genes
 *     - out_dispersions.len >= n_genes
 *
 * POSTCONDITIONS:
 *     - out_indices contains indices of n_top genes with highest dispersion
 *     - out_mask[i] = 1 if gene i is selected
 *     - out_dispersions[i] = var[i] / mean[i] (or 0 if mean <= epsilon)
 *
 * ALGORITHM:
 *     1. Compute gene-wise mean and variance (parallel over genes)
 *     2. Compute dispersion with 4-way SIMD unroll
 *     3. Partial sort to select top k genes
 *
 * COMPLEXITY:
 *     Time:  O(nnz + n_genes * log(n_top))
 *     Space: O(n_genes) for intermediate buffers
 *
 * THREAD SAFETY:
 *     Safe - parallel_for over genes with no shared mutable state
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void select_by_dispersion(
    const Sparse<T, IsCSR>& matrix,    // Expression matrix
    Size n_top,                        // Number of genes to select
    Array<Index> out_indices,          // Output: selected gene indices [n_top]
    Array<uint8_t> out_mask,           // Output: selection mask [n_genes]
    Array<Real> out_dispersions        // Output: dispersion values [n_genes]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: select_by_vst
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Select highly variable genes using variance-stabilizing transformation.
 *
 * PARAMETERS:
 *     matrix        [in]  Sparse expression matrix
 *     clip_vals     [in]  Per-gene clipping values [n_genes]
 *     n_top         [in]  Number of top genes to select
 *     out_indices   [out] Indices of selected genes [n_top]
 *     out_mask      [out] Binary mask (1 = selected) [n_genes]
 *     out_variances [out] Variance values after clipping [n_genes]
 *
 * PRECONDITIONS:
 *     - clip_vals.len >= n_genes
 *     - out_indices.len >= n_top
 *     - out_mask.len >= n_genes
 *     - out_variances.len >= n_genes
 *
 * POSTCONDITIONS:
 *     - out_indices contains indices of n_top genes with highest clipped variance
 *     - out_mask[i] = 1 if gene i is selected
 *     - out_variances[i] = variance after clipping to clip_vals[i]
 *
 * ALGORITHM:
 *     1. For each gene, clip values to clip_val before computing variance
 *     2. Compute mean and variance (parallel over genes)
 *     3. Partial sort to select top k genes
 *
 * COMPLEXITY:
 *     Time:  O(nnz + n_genes * log(n_top))
 *     Space: O(n_genes)
 *
 * THREAD SAFETY:
 *     Safe - parallel over genes
 *
 * NUMERICAL NOTES:
 *     Clipping prevents high-expression outlier genes from dominating
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void select_by_vst(
    const Sparse<T, IsCSR>& matrix,    // Expression matrix
    Array<const Real> clip_vals,       // Per-gene clip values [n_genes]
    Size n_top,                        // Number of genes to select
    Array<Index> out_indices,          // Output: selected gene indices [n_top]
    Array<uint8_t> out_mask,           // Output: selection mask [n_genes]
    Array<Real> out_variances          // Output: clipped variance [n_genes]
);

// =============================================================================
// Internal Functions (detail namespace)
// =============================================================================

namespace detail {

/* -----------------------------------------------------------------------------
 * FUNCTION: dispersion_simd
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute dispersion = var / mean with SIMD optimization.
 *
 * PARAMETERS:
 *     means          [in]  Gene means [n]
 *     vars           [in]  Gene variances [n]
 *     out_dispersion [out] Dispersion values [n]
 *
 * ALGORITHM:
 *     4-way SIMD unroll with prefetch.
 *     Uses masked division to handle mean <= epsilon.
 * -------------------------------------------------------------------------- */
void dispersion_simd(
    Array<const Real> means,           // Gene means [n]
    Array<const Real> vars,            // Gene variances [n]
    Array<Real> out_dispersion         // Output dispersion [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: normalize_dispersion_simd
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Z-score normalize dispersions within a mean range.
 *
 * PARAMETERS:
 *     dispersions [in,out] Dispersion values, normalized in-place
 *     min_mean    [in]     Minimum mean for inclusion
 *     max_mean    [in]     Maximum mean for inclusion
 *     means       [in]     Gene means
 *
 * MUTABILITY:
 *     INPLACE - modifies dispersions
 *
 * POSTCONDITIONS:
 *     - Genes outside [min_mean, max_mean]: dispersion = -infinity
 *     - Others: z-score normalized
 * -------------------------------------------------------------------------- */
void normalize_dispersion_simd(
    Array<Real> dispersions,           // Dispersions [n], modified in-place
    Real min_mean,                     // Minimum mean threshold
    Real max_mean,                     // Maximum mean threshold
    Array<const Real> means            // Gene means [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: select_top_k_partial
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Select top k elements using partial sort.
 *
 * PARAMETERS:
 *     scores      [in]  Score values to rank by [n]
 *     k           [in]  Number to select
 *     out_indices [out] Top k indices [k]
 *     out_mask    [out] Selection mask [n]
 *
 * ALGORITHM:
 *     Uses scl::algo::partial_sort for O(n + k log k) selection.
 * -------------------------------------------------------------------------- */
void select_top_k_partial(
    Array<const Real> scores,          // Scores to rank [n]
    Size k,                            // Number to select
    Array<Index> out_indices,          // Output indices [k]
    Array<uint8_t> out_mask            // Output mask [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_moments
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute mean and variance for each gene.
 *
 * PARAMETERS:
 *     matrix    [in]  Sparse expression matrix
 *     out_means [out] Gene means [n_genes]
 *     out_vars  [out] Gene variances [n_genes]
 *     ddof      [in]  Delta degrees of freedom (0 or 1)
 *
 * ALGORITHM:
 *     Uses scl::vectorize::sum and sum_squared for SIMD accumulation.
 *     Parallel over genes.
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void compute_moments(
    const Sparse<T, IsCSR>& matrix,    // Expression matrix
    Array<Real> out_means,             // Output means [n_genes]
    Array<Real> out_vars,              // Output variances [n_genes]
    int ddof                           // Degrees of freedom correction
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_clipped_moments
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute mean and variance with per-gene value clipping.
 *
 * PARAMETERS:
 *     matrix    [in]  Sparse expression matrix
 *     clip_vals [in]  Per-gene clipping values [n_genes]
 *     out_means [out] Gene means [n_genes]
 *     out_vars  [out] Gene variances [n_genes]
 *
 * ALGORITHM:
 *     For each gene: clip values to clip_val, then compute moments.
 *     Parallel over genes.
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void compute_clipped_moments(
    const Sparse<T, IsCSR>& matrix,    // Expression matrix
    Array<const Real> clip_vals,       // Per-gene clip values [n_genes]
    Array<Real> out_means,             // Output means [n_genes]
    Array<Real> out_vars               // Output variances [n_genes]
);

} // namespace detail

} // namespace scl::kernel::hvg
