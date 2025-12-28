// =============================================================================
// FILE: scl/kernel/clonotype.h
// BRIEF: API reference for TCR/BCR clonal analysis
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"

namespace scl::kernel::clonotype {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Index NO_CLONE = -1;
    constexpr Size MIN_CLONE_SIZE = 2;
}

// =============================================================================
// Clonotype Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: clonal_diversity
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute clonal diversity metrics (Shannon entropy, Simpson index, Gini).
 *
 * PARAMETERS:
 *     clone_ids   [in]  Clone IDs for each cell [n_cells]
 *     n_cells     [in]  Number of cells
 *     shannon_entropy [out] Shannon entropy
 *     simpson_index   [out] Simpson diversity index
 *     gini_coeff      [out] Gini coefficient
 *
 * PRECONDITIONS:
 *     - All clone IDs are valid
 *
 * POSTCONDITIONS:
 *     - shannon_entropy contains H = -sum(p_i * log(p_i))
 *     - simpson_index contains 1 - sum(p_i^2)
 *     - gini_coeff contains inequality measure
 *
 * COMPLEXITY:
 *     Time:  O(n_cells)
 *     Space: O(n_clones) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized
 * -------------------------------------------------------------------------- */
void clonal_diversity(
    Array<const Index> clone_ids,           // Clone IDs [n_cells]
    Size n_cells,                            // Number of cells
    Real& shannon_entropy,                   // Output Shannon entropy
    Real& simpson_index,                     // Output Simpson index
    Real& gini_coeff                         // Output Gini coefficient
);

/* -----------------------------------------------------------------------------
 * FUNCTION: clone_expansion
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Identify expanded clones based on size threshold.
 *
 * PARAMETERS:
 *     clone_ids      [in]  Clone IDs [n_cells]
 *     n_cells        [in]  Number of cells
 *     expanded_clones [out] Expanded clone IDs [n_clones]
 *     clone_sizes     [out] Clone sizes [n_clones]
 *     min_size        [in]  Minimum size for expansion
 *     max_results     [in]  Maximum number of results
 *
 * PRECONDITIONS:
 *     - expanded_clones has capacity >= max_results
 *     - clone_sizes has capacity >= max_results
 *
 * POSTCONDITIONS:
 *     - Returns number of expanded clones
 *     - Output arrays contain expanded clone information
 *
 * COMPLEXITY:
 *     Time:  O(n_cells + n_clones)
 *     Space: O(n_clones) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
Index clone_expansion(
    Array<const Index> clone_ids,           // Clone IDs [n_cells]
    Size n_cells,                            // Number of cells
    Index* expanded_clones,                  // Output expanded clones [max_results]
    Size* clone_sizes,                       // Output clone sizes [max_results]
    Size min_size = config::MIN_CLONE_SIZE,  // Minimum size
    Index max_results                        // Maximum results
);

/* -----------------------------------------------------------------------------
 * FUNCTION: clone_phenotype_association
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Test association between clones and phenotypes.
 *
 * PARAMETERS:
 *     clone_ids      [in]  Clone IDs [n_cells]
 *     phenotypes     [in]  Phenotype labels [n_cells]
 *     n_clones       [in]  Number of clones
 *     n_phenotypes   [in]  Number of phenotypes
 *     association_matrix [out] Association matrix [n_clones * n_phenotypes]
 *
 * PRECONDITIONS:
 *     - association_matrix has capacity >= n_clones * n_phenotypes
 *
 * POSTCONDITIONS:
 *     - association_matrix[c * n_phenotypes + p] contains association strength
 *
 * COMPLEXITY:
 *     Time:  O(n_cells + n_clones * n_phenotypes)
 *     Space: O(n_clones * n_phenotypes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized
 * -------------------------------------------------------------------------- */
void clone_phenotype_association(
    Array<const Index> clone_ids,           // Clone IDs [n_cells]
    Array<const Index> phenotypes,           // Phenotype labels [n_cells]
    Size n_clones,                            // Number of clones
    Size n_phenotypes,                        // Number of phenotypes
    Real* association_matrix                  // Output associations [n_clones * n_phenotypes]
);

} // namespace scl::kernel::clonotype

