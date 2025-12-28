// =============================================================================
// FILE: scl/kernel/tissue.h
// BRIEF: API reference for tissue architecture and organization analysis
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"

namespace scl::kernel::tissue {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_CELLS_PER_LAYER = 5;
    constexpr Size DEFAULT_N_NEIGHBORS = 15;
    constexpr Size MAX_ITERATIONS = 100;
    constexpr Real PI = Real(3.14159265358979323846);
}

// =============================================================================
// Tissue Analysis Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: layer_assignment
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Assign cells to tissue layers based on spatial coordinates.
 *
 * PARAMETERS:
 *     coordinates  [in]  Spatial coordinates [n_cells * n_dims]
 *     n_cells      [in]  Number of cells
 *     n_dims       [in]  Number of spatial dimensions
 *     layer_labels [out] Layer labels [n_cells]
 *     n_layers     [in]  Number of layers to assign
 *
 * PRECONDITIONS:
 *     - layer_labels has capacity >= n_cells
 *     - n_layers > 0
 *
 * POSTCONDITIONS:
 *     - layer_labels[i] contains layer ID for cell i
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * n_dims)
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
void layer_assignment(
    const Real* coordinates,                // Spatial coordinates [n_cells * n_dims]
    Size n_cells,                            // Number of cells
    Size n_dims,                              // Number of dimensions
    Array<Index> layer_labels,               // Output layer labels [n_cells]
    Index n_layers                           // Number of layers
);

/* -----------------------------------------------------------------------------
 * FUNCTION: zonation_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute zonation score along a spatial axis.
 *
 * PARAMETERS:
 *     coordinates  [in]  Spatial coordinates [n_cells * n_dims]
 *     expression   [in]  Gene expression values [n_cells]
 *     n_cells      [in]  Number of cells
 *     axis         [in]  Spatial axis index (0, 1, or 2)
 *     scores       [out] Zonation scores [n_cells]
 *
 * PRECONDITIONS:
 *     - scores has capacity >= n_cells
 *     - axis < n_dims
 *
 * POSTCONDITIONS:
 *     - scores[i] contains zonation score along specified axis
 *
 * COMPLEXITY:
 *     Time:  O(n_cells)
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized
 * -------------------------------------------------------------------------- */
void zonation_score(
    const Real* coordinates,                // Spatial coordinates [n_cells * n_dims]
    const Real* expression,                  // Gene expression [n_cells]
    Size n_cells,                            // Number of cells
    Size axis,                                // Spatial axis
    Array<Real> scores                        // Output zonation scores [n_cells]
);

} // namespace scl::kernel::tissue

