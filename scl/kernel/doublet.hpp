#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <vector>
#include <random>

// =============================================================================
// FILE: scl/kernel/doublet.hpp
// BRIEF: Doublet detection for quality control
//
// STRATEGIC POSITION: Sparse + Nonlinear (Tier 3 - Biology QC)
// - Quality control for single-cell data
// - Utilizes neighbors module
// - Sparse simulation and scoring
// - Nonlinear scoring functions
//
// APPLICATIONS:
// - Detect doublets (two cells captured together)
// - Quality control filtering
// - Improve downstream analysis quality
//
// METHODS:
// - Simulate artificial doublets
// - Score cells by similarity to simulated doublets
// - Neighbor-based doublet enrichment
// =============================================================================

namespace scl::kernel::doublet {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_SIM_FRACTION = Real(1.0);  // Simulate 100% of #cells
    constexpr Index DEFAULT_N_NEIGHBORS = 30;
    constexpr Real DEFAULT_THRESHOLD = Real(0.25);    // Score threshold
}

// =============================================================================
// Core Functions (TODO: Implementation)
// =============================================================================

// TODO: Simulate artificial doublets
// Combine random pairs of cells
template <typename T, bool IsCSR>
void simulate_doublets(
    const Sparse<T, IsCSR>& X,
    Real sim_fraction,
    Sparse<T, IsCSR>& doublets,             // Output: simulated doublets
    std::vector<std::pair<Index, Index>>& parent_pairs,  // Parent cell pairs
    uint64_t seed = 42
);

// TODO: Compute doublet scores
// Score each cell by proximity to simulated doublets
template <typename T, bool IsCSR>
void doublet_score(
    const Sparse<T, IsCSR>& X,
    const Sparse<T, IsCSR>& doublets,
    const Sparse<Index, IsCSR>& knn_indices,    // From neighbors module
    Array<Real> scores                           // Output per cell
);

// TODO: Compute doublet neighbor proportion
// Fraction of neighbors that are simulated doublets
template <typename T, bool IsCSR>
void doublet_neighbor_proportion(
    const Sparse<Index, IsCSR>& knn_indices,
    Index n_real_cells,
    Index n_doublets,
    Array<Real> proportions                      // Output per cell
);

// TODO: Call doublets based on score threshold
void call_doublets(
    Array<const Real> scores,
    Real threshold,
    Array<bool> is_doublet                       // Output: true if doublet
);

// =============================================================================
// Advanced Methods (TODO: Implementation)
// =============================================================================

// TODO: Iterative doublet detection
// Refine doublet calls through multiple rounds
template <typename T, bool IsCSR>
void iterative_doublet_detection(
    const Sparse<T, IsCSR>& X,
    Index n_iterations,
    Array<Real> final_scores,
    Array<bool> is_doublet
);

// TODO: Expected doublet rate estimation
Real estimate_doublet_rate(
    Index n_cells,
    Real capture_rate = Real(0.01)              // Typical ~1% doublet rate
);

// TODO: Doublet enrichment in clusters
// Identify clusters with elevated doublet scores
void cluster_doublet_enrichment(
    Array<const Real> doublet_scores,
    Array<const Index> cluster_labels,
    Index n_clusters,
    Array<Real> cluster_enrichment,
    Array<Real> cluster_pvalues
);

// =============================================================================
// Simulation Strategies (TODO: Implementation)
// =============================================================================

// TODO: Simulate doublets with expression model
// Model doublet expression as sum of parent cells
template <typename T, bool IsCSR>
void simulate_doublets_additive(
    const Sparse<T, IsCSR>& X,
    Real sim_fraction,
    Sparse<T, IsCSR>& doublets,
    uint64_t seed = 42
);

// TODO: Simulate doublets from specific clusters
// Preferentially create inter-cluster doublets
template <typename T, bool IsCSR>
void simulate_doublets_clusters(
    const Sparse<T, IsCSR>& X,
    Array<const Index> cluster_labels,
    Real sim_fraction,
    Sparse<T, IsCSR>& doublets,
    uint64_t seed = 42
);

// =============================================================================
// Helper Functions (TODO: Implementation)
// =============================================================================

namespace detail {

// TODO: Combine two cells to create doublet
template <typename T>
void combine_cells(
    const T* cell1_values,
    const Index* cell1_indices,
    Size cell1_nnz,
    const T* cell2_values,
    const Index* cell2_indices,
    Size cell2_nnz,
    std::vector<T>& doublet_values,
    std::vector<Index>& doublet_indices
);

// TODO: Compute distance to nearest doublet
template <typename T>
Real distance_to_nearest_doublet(
    Index cell_idx,
    const Sparse<T, true>& knn_indices,
    Index n_real_cells
);

// TODO: Bootstrap confidence interval for doublet scores
void doublet_score_confidence(
    Array<const Real> scores,
    Index n_bootstrap,
    Array<Real> lower_bound,
    Array<Real> upper_bound,
    uint64_t seed = 42
);

} // namespace detail

} // namespace scl::kernel::doublet

