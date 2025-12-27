#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
// FILE: scl/kernel/niche.hpp
// BRIEF: Cellular neighborhood and microenvironment analysis
//
// STRATEGIC POSITION: Sparse + Nonlinear (Tier 2 - Spatial)
// - Spatial transcriptomics analysis
// - Combines spatial + neighbors
// - Microenvironment characterization
//
// APPLICATIONS:
// - Tumor microenvironment
// - Tissue architecture
// - Cell-cell interactions
// - Niche clustering
//
// KEY OPERATIONS:
// - Neighborhood composition
// - Co-localization scoring
// - Niche detection
// =============================================================================

namespace scl::kernel::niche {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Index DEFAULT_N_NEIGHBORS = 10;
    constexpr Real DEFAULT_RADIUS = Real(50.0);  // Spatial radius
}

// =============================================================================
// Neighborhood Composition (TODO: Implementation)
// =============================================================================

// TODO: Compute neighborhood cell type composition
template <typename T, bool IsCSR>
void neighborhood_composition(
    const Sparse<Index, IsCSR>& spatial_neighbors,
    Array<const Index> cell_type_labels,
    Index n_cell_types,
    Sparse<Real, !IsCSR>& composition  // Cells × Cell types
);

// TODO: Enrichment/depletion of cell types in neighborhoods
template <typename T, bool IsCSR>
void neighborhood_enrichment(
    const Sparse<Real, !IsCSR>& composition,
    Array<const Index> cell_type_labels,
    Sparse<Real, !IsCSR>& enrichment_scores,
    Array<Real> p_values
);

// =============================================================================
// Niche Clustering (TODO: Implementation)
// =============================================================================

// TODO: Cluster cells by neighborhood similarity
template <typename T, bool IsCSR>
void niche_clustering(
    const Sparse<Real, !IsCSR>& composition,
    Index n_niches,
    Array<Index> niche_labels
);

// TODO: Identify recurrent niche patterns
template <typename T, bool IsCSR>
void identify_niche_patterns(
    const Sparse<Real, !IsCSR>& composition,
    Index min_frequency,
    std::vector<std::vector<Index>>& niche_patterns
);

// =============================================================================
// Cell-Cell Contact (TODO: Implementation)
// =============================================================================

// TODO: Compute cell-cell contact frequency
void cell_cell_contact(
    const Sparse<Index, true>& spatial_neighbors,
    Array<const Index> cell_type_labels,
    Index n_cell_types,
    Sparse<Real, true>& contact_matrix  // Cell type × Cell type
);

// TODO: Permutation test for contact significance
void contact_significance(
    const Sparse<Real, true>& observed_contacts,
    const Sparse<Index, true>& spatial_neighbors,
    Array<const Index> cell_type_labels,
    Index n_permutations,
    Sparse<Real, true>& p_values
);

// =============================================================================
// Spatial Communication (TODO: Implementation)
// =============================================================================

// TODO: Spatially-constrained communication
template <typename T, bool IsCSR>
void spatial_communication(
    const Sparse<T, IsCSR>& expression,
    const Sparse<Index, IsCSR>& spatial_neighbors,
    const std::vector<std::pair<Index, Index>>& lr_pairs,
    Sparse<Real, true>& spatial_lr_scores
);

// =============================================================================
// Co-localization (TODO: Implementation)
// =============================================================================

// TODO: Co-localization score for cell type pairs
void colocalization_score(
    Array<const Index> cell_type_labels,
    const Sparse<Index, true>& spatial_neighbors,
    Index type_a,
    Index type_b,
    Real& colocalization,
    Real& p_value,
    Index n_permutations = 1000
);

// TODO: Global co-localization matrix
void colocalization_matrix(
    Array<const Index> cell_type_labels,
    const Sparse<Index, true>& spatial_neighbors,
    Index n_cell_types,
    Sparse<Real, true>& coloc_matrix
);

// =============================================================================
// Helper Functions (TODO: Implementation)
// =============================================================================

namespace detail {

// TODO: Compute composition vector for single cell
void single_cell_composition(
    Index cell_idx,
    const Index* neighbor_indices,
    Size n_neighbors,
    const Index* all_labels,
    Index n_types,
    Real* composition
);

} // namespace detail

} // namespace scl::kernel::niche

