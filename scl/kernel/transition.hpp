#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
// FILE: scl/kernel/transition.hpp
// BRIEF: Cell state transition analysis (CellRank-style)
//
// STRATEGIC POSITION: Sparse + Nonlinear (Tier 1 - Dynamics)
// - Markov chain analysis on sparse graphs
// - Terminal state prediction
// - Transition dynamics quantification
//
// APPLICATIONS:
// - CellRank-style analysis
// - Terminal state identification
// - Fate probability computation
// - Metastable state detection
//
// THEORY:
// Model cell state transitions as Markov chain
// =============================================================================

namespace scl::kernel::transition {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_TOLERANCE = Real(1e-6);
    constexpr Index DEFAULT_MAX_ITER = 1000;
}

// =============================================================================
// Transition Matrix (TODO: Implementation)
// =============================================================================

// TODO: Build transition matrix from velocity or neighbors
template <typename T, bool IsCSR>
void transition_matrix(
    const Sparse<T, IsCSR>& velocity_graph,
    Sparse<Real, IsCSR>& transition_mat  // Row-stochastic
);

// TODO: Symmetrize transition matrix
template <typename T, bool IsCSR>
void symmetrize_transition(
    const Sparse<T, IsCSR>& transition_mat,
    Sparse<Real, IsCSR>& symmetric_mat
);

// =============================================================================
// Absorption Probability (TODO: Implementation)
// =============================================================================

// TODO: Compute absorption probabilities to terminal states
template <typename T, bool IsCSR>
void absorption_probability(
    const Sparse<T, IsCSR>& transition_mat,
    Array<const bool> is_terminal,
    Sparse<Real, !IsCSR>& absorption_probs  // Cells × Terminal clusters
);

// TODO: Identify terminal states automatically
template <typename T, bool IsCSR>
void identify_terminal_states(
    const Sparse<T, IsCSR>& transition_mat,
    Array<bool> is_terminal,
    Real threshold = Real(0.95)
);

// =============================================================================
// Hitting Time (TODO: Implementation)
// =============================================================================

// TODO: Mean first passage time
template <typename T, bool IsCSR>
void hitting_time(
    const Sparse<T, IsCSR>& transition_mat,
    Array<const Index> target_states,
    Array<Real> mean_hitting_time
);

// TODO: Expected time to absorption
template <typename T, bool IsCSR>
void time_to_absorption(
    const Sparse<T, IsCSR>& transition_mat,
    Array<const bool> is_absorbing,
    Array<Real> expected_time
);

// =============================================================================
// Stationary Distribution (TODO: Implementation)
// =============================================================================

// TODO: Compute stationary distribution via power iteration
template <typename T, bool IsCSR>
void stationary_distribution(
    const Sparse<T, IsCSR>& transition_mat,
    Array<Real> stationary,
    Index max_iter = config::DEFAULT_MAX_ITER,
    Real tol = config::DEFAULT_TOLERANCE
);

// TODO: Compute fundamental matrix (transient states)
template <typename T, bool IsCSR>
void fundamental_matrix(
    const Sparse<T, IsCSR>& transition_mat,
    Array<const bool> is_transient,
    Sparse<Real, IsCSR>& fundamental
);

// =============================================================================
// Metastable States (TODO: Implementation)
// =============================================================================

// TODO: Detect metastable states via eigendecomposition
template <typename T, bool IsCSR>
void metastable_states(
    const Sparse<T, IsCSR>& transition_mat,
    Index n_states,
    Array<Index> state_labels,
    Sparse<Real, !IsCSR>& membership_probs
);

// TODO: PCCA+ clustering for metastable states
template <typename T, bool IsCSR>
void pcca_clustering(
    const Sparse<T, IsCSR>& transition_mat,
    Index n_clusters,
    Array<Index> cluster_labels
);

// =============================================================================
// Lineage Drivers (TODO: Implementation)
// =============================================================================

// TODO: Identify genes driving state transitions
template <typename T, bool IsCSR>
void lineage_drivers(
    const Sparse<T, IsCSR>& expression,
    const Sparse<Real, !IsCSR>& absorption_probs,
    Index lineage,
    std::vector<Index>& driver_genes,
    std::vector<Real>& driver_scores
);

// =============================================================================
// Helper Functions (TODO: Implementation)
// =============================================================================

namespace detail {

// TODO: Check if matrix is row-stochastic
template <typename T, bool IsCSR>
bool is_stochastic(
    const Sparse<T, IsCSR>& mat,
    Real tol = Real(1e-6)
);

// TODO: Power iteration for dominant eigenvector
template <typename T, bool IsCSR>
void power_iteration(
    const Sparse<T, IsCSR>& mat,
    Array<Real> eigenvector,
    Index max_iter,
    Real tol
);

} // namespace detail

} // namespace scl::kernel::transition

