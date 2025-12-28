// =============================================================================
// FILE: scl/kernel/spatial_pattern.h
// BRIEF: API reference for spatial pattern detection (SpatialDE-style)
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::spatial_pattern {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_NEIGHBORS = 3;
    constexpr Size DEFAULT_N_NEIGHBORS = 15;
    constexpr Size MAX_ITERATIONS = 100;
    constexpr Real BANDWIDTH_SCALE = Real(0.3);
}

// =============================================================================
// Spatial Pattern Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: spatially_variable_genes
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Identify spatially variable genes using spatial autocorrelation.
 *
 * PARAMETERS:
 *     expression    [in]  Expression matrix (cells x genes, CSR)
 *     coordinates   [in]  Spatial coordinates [n_cells * n_dims]
 *     n_cells       [in]  Number of cells
 *     n_genes       [in]  Number of genes
 *     n_dims        [in]  Number of spatial dimensions
 *     sv_scores     [out] Spatial variation scores [n_genes]
 *     bandwidth     [in]  Spatial bandwidth for kernel
 *
 * PRECONDITIONS:
 *     - sv_scores has capacity >= n_genes
 *
 * POSTCONDITIONS:
 *     - sv_scores[g] contains spatial variation score for gene g
 *
 * COMPLEXITY:
 *     Time:  O(n_genes * n_cells^2)
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over genes
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void spatially_variable_genes(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    const Real* coordinates,                 // Spatial coordinates [n_cells * n_dims]
    Index n_cells,                          // Number of cells
    Index n_genes,                          // Number of genes
    Size n_dims,                            // Number of dimensions
    Array<Real> sv_scores,                  // Output SV scores [n_genes]
    Real bandwidth                          // Spatial bandwidth
);

/* -----------------------------------------------------------------------------
 * FUNCTION: spatial_gradient
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute spatial gradient of gene expression.
 *
 * PARAMETERS:
 *     expression    [in]  Expression matrix (cells x genes, CSR)
 *     coordinates   [in]  Spatial coordinates [n_cells * n_dims]
 *     gene_index    [in]  Gene index
 *     n_cells       [in]  Number of cells
 *     n_dims        [in]  Number of spatial dimensions
 *     gradients     [out] Gradient vectors [n_cells * n_dims]
 *
 * PRECONDITIONS:
 *     - gradients has capacity >= n_cells * n_dims
 *
 * POSTCONDITIONS:
 *     - gradients[i * n_dims + d] contains gradient component d for cell i
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * n_neighbors)
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void spatial_gradient(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    const Real* coordinates,                // Spatial coordinates [n_cells * n_dims]
    Index gene_index,                        // Gene index
    Index n_cells,                          // Number of cells
    Size n_dims,                            // Number of dimensions
    Real* gradients                          // Output gradients [n_cells * n_dims]
);

} // namespace scl::kernel::spatial_pattern

