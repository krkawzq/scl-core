// =============================================================================
// FILE: scl/kernel/lineage.h
// BRIEF: API reference for lineage tracing and fate mapping
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"

namespace scl::kernel::lineage {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Index NO_PARENT = -1;
    constexpr Size MIN_CLONE_SIZE = 2;
}

// =============================================================================
// Lineage Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: lineage_coupling
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute coupling matrix between clones and cell types.
 *
 * PARAMETERS:
 *     clone_ids      [in]  Clone IDs for each cell [n_cells]
 *     cell_types     [in]  Cell type labels [n_cells]
 *     coupling_matrix [out] Coupling matrix [n_clones * n_types]
 *     n_clones       [in]  Number of clones
 *     n_types        [in]  Number of cell types
 *
 * PRECONDITIONS:
 *     - coupling_matrix has capacity >= n_clones * n_types
 *
 * POSTCONDITIONS:
 *     - coupling_matrix[c * n_types + t] contains fraction of clone c that is type t
 *
 * COMPLEXITY:
 *     Time:  O(n_cells)
 *     Space: O(n_clones * n_types) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized with atomic accumulation
 * -------------------------------------------------------------------------- */
void lineage_coupling(
    Array<const Index> clone_ids,            // Clone IDs [n_cells]
    Array<const Index> cell_types,           // Cell type labels [n_cells]
    Real* coupling_matrix,                   // Output coupling [n_clones * n_types]
    Size n_clones,                            // Number of clones
    Size n_types                              // Number of cell types
);

/* -----------------------------------------------------------------------------
 * FUNCTION: fate_bias
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute fate bias of clones toward different cell types.
 *
 * PARAMETERS:
 *     clone_ids      [in]  Clone IDs [n_cells]
 *     cell_types     [in]  Cell type labels [n_cells]
 *     n_clones       [in]  Number of clones
 *     n_types        [in]  Number of cell types
 *     fate_bias      [out] Fate bias matrix [n_clones * n_types]
 *
 * PRECONDITIONS:
 *     - fate_bias has capacity >= n_clones * n_types
 *
 * POSTCONDITIONS:
 *     - fate_bias[c * n_types + t] contains bias of clone c toward type t
 *
 * COMPLEXITY:
 *     Time:  O(n_cells + n_clones * n_types)
 *     Space: O(n_clones * n_types) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized
 * -------------------------------------------------------------------------- */
void fate_bias(
    Array<const Index> clone_ids,            // Clone IDs [n_cells]
    Array<const Index> cell_types,            // Cell type labels [n_cells]
    Size n_clones,                            // Number of clones
    Size n_types,                              // Number of cell types
    Real* fate_bias                            // Output fate bias [n_clones * n_types]
);

} // namespace scl::kernel::lineage

