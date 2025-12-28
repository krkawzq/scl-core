// =============================================================================
// FILE: scl/kernel/niche.h
// BRIEF: API reference for cellular niche analysis
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::niche {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size DEFAULT_K = 15;
    constexpr Size PARALLEL_THRESHOLD = 500;
}

// =============================================================================
// Niche Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: niche_composition
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute cell type composition in spatial neighborhoods.
 *
 * PARAMETERS:
 *     spatial_graph [in]  Spatial neighbor graph (CSR)
 *     cell_types    [in]  Cell type labels [n_cells]
 *     n_cells       [in]  Number of cells
 *     n_types       [in]  Number of cell types
 *     composition   [out] Niche composition [n_cells * n_types]
 *
 * PRECONDITIONS:
 *     - composition has capacity >= n_cells * n_types
 *
 * POSTCONDITIONS:
 *     - composition[i * n_types + t] contains fraction of type t in niche of cell i
 *
 * COMPLEXITY:
 *     Time:  O(nnz + n_cells * n_types)
 *     Space: O(n_types) auxiliary per cell
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void niche_composition(
    const Sparse<T, IsCSR>& spatial_graph,  // Spatial graph [n_cells x n_cells]
    Array<const Index> cell_types,          // Cell type labels [n_cells]
    Index n_cells,                           // Number of cells
    Index n_types,                           // Number of cell types
    Real* composition                         // Output composition [n_cells * n_types]
);

} // namespace scl::kernel::niche

