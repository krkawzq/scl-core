#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
// FILE: scl/kernel/velocity.hpp
// BRIEF: RNA velocity analysis primitives
//
// STRATEGIC POSITION: Sparse + Nonlinear (Tier 1 - Dynamics)
// - Dynamic trajectory analysis
// - Spliced/unspliced expression modeling
// - Velocity graph construction
//
// APPLICATIONS:
// - RNA velocity estimation
// - Cell fate prediction
// - Latent time inference
// - Velocity projection
//
// KEY OPERATIONS:
// - Splicing kinetics
// - Velocity graph (transition probabilities)
// - Embedding projection
// =============================================================================

namespace scl::kernel::velocity {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_MIN_R2 = Real(0.01);
    constexpr Index DEFAULT_N_NEIGHBORS = 30;
}

// =============================================================================
// Splice Ratio (TODO: Implementation)
// =============================================================================

// TODO: Compute spliced/unspliced ratio
template <typename T, bool IsCSR>
void splice_ratio(
    const Sparse<T, IsCSR>& spliced,
    const Sparse<T, IsCSR>& unspliced,
    Sparse<Real, IsCSR>& ratio
);

// TODO: Compute velocity (ds/dt)
template <typename T, bool IsCSR>
void compute_velocity(
    const Sparse<T, IsCSR>& spliced,
    const Sparse<T, IsCSR>& unspliced,
    Sparse<Real, IsCSR>& velocity,
    Array<Real> gamma  // Degradation rates per gene
);

// =============================================================================
// Velocity Graph (TODO: Implementation)
// =============================================================================

// TODO: Construct velocity transition graph
template <typename T, bool IsCSR>
void velocity_graph(
    const Sparse<T, IsCSR>& velocity,
    const Sparse<Index, IsCSR>& knn_indices,
    const Sparse<Real, IsCSR>& knn_distances,
    Sparse<Real, IsCSR>& transition_probs
);

// TODO: Cosine similarity-based velocity graph
template <typename T, bool IsCSR>
void velocity_graph_cosine(
    const Sparse<T, IsCSR>& velocity,
    const Sparse<Index, IsCSR>& knn_indices,
    Sparse<Real, IsCSR>& transition_probs
);

// =============================================================================
// Velocity Embedding (TODO: Implementation)
// =============================================================================

// TODO: Project velocity to low-dimensional embedding
template <typename T, bool IsCSR>
void velocity_embedding(
    const Sparse<T, IsCSR>& velocity,
    const Sparse<Real, !IsCSR>& embedding,  // e.g., UMAP coordinates
    const Sparse<Real, !IsCSR>& pca_components,
    Sparse<Real, !IsCSR>& velocity_embedded
);

// TODO: Grid-based velocity visualization
template <typename T, bool IsCSR>
void velocity_grid(
    const Sparse<Real, !IsCSR>& embedding,
    const Sparse<Real, !IsCSR>& velocity_embedded,
    Index grid_size,
    Sparse<Real, !IsCSR>& grid_velocity
);

// =============================================================================
// Cell Fate Probability (TODO: Implementation)
// =============================================================================

// TODO: Compute absorption probabilities to terminal states
template <typename T, bool IsCSR>
void cell_fate_probability(
    const Sparse<T, IsCSR>& transition_graph,
    Array<const Index> terminal_cells,
    Sparse<Real, !IsCSR>& fate_probs  // Cells × Terminal states
);

// TODO: Predict future state via random walk
template <typename T, bool IsCSR>
void predict_future_state(
    const Sparse<T, IsCSR>& transition_graph,
    Index n_steps,
    Sparse<Real, IsCSR>& propagated_graph
);

// =============================================================================
// Latent Time (TODO: Implementation)
// =============================================================================

// TODO: Infer latent time from velocity
template <typename T, bool IsCSR>
void latent_time(
    const Sparse<T, IsCSR>& spliced,
    const Sparse<T, IsCSR>& unspliced,
    const Sparse<Real, IsCSR>& velocity_graph,
    Array<Real> latent_time_out
);

// TODO: Root cell-based latent time
template <typename T, bool IsCSR>
void latent_time_rooted(
    const Sparse<Real, IsCSR>& velocity_graph,
    Index root_cell,
    Array<Real> latent_time_out
);

// =============================================================================
// Helper Functions (TODO: Implementation)
// =============================================================================

namespace detail {

// TODO: Fit splicing kinetics
void fit_kinetics(
    const Real* spliced,
    const Real* unspliced,
    Size n_cells,
    Real& gamma,
    Real& r2
);

// TODO: Compute velocity confidence
Real velocity_confidence(
    const Real* velocity,
    const Index* neighbor_indices,
    Size n_neighbors
);

} // namespace detail

} // namespace scl::kernel::velocity

