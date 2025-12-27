#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
// FILE: scl/kernel/pseudotime.hpp
// BRIEF: Pseudotime inference for trajectory analysis
//
// STRATEGIC POSITION: Sparse + Nonlinear (Tier 1 - Trajectory)
// - Developmental biology core analysis
// - Graph-based trajectory inference
// - Depends on: diffusion, components, neighbors
//
// APPLICATIONS:
// - Developmental trajectories
// - Cell differentiation ordering
// - Branch point detection
// - Root cell selection
//
// METHODS:
// - Diffusion Pseudotime (DPT, Haghverdi et al.)
// - Graph shortest path pseudotime
// - Branch detection and segmentation
// =============================================================================

namespace scl::kernel::pseudotime {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Index DEFAULT_N_DCS = 10;      // Diffusion components
    constexpr Real DEFAULT_THRESHOLD = Real(0.1);
}

// =============================================================================
// Diffusion Pseudotime (TODO: Implementation)
// =============================================================================

// TODO: Diffusion Pseudotime (DPT)
template <typename T, bool IsCSR>
void diffusion_pseudotime(
    const Sparse<T, IsCSR>& transition_matrix,
    Index root_cell,
    Array<Real> pseudotime,
    Index n_dcs = config::DEFAULT_N_DCS
);

// TODO: Multi-branch DPT with multiple roots
template <typename T, bool IsCSR>
void multibranch_dpt(
    const Sparse<T, IsCSR>& transition_matrix,
    Array<const Index> root_cells,
    std::vector<Array<Real>>& branch_pseudotimes
);

// =============================================================================
// Graph-Based Pseudotime (TODO: Implementation)
// =============================================================================

// TODO: Shortest path pseudotime
template <typename T, bool IsCSR>
void graph_pseudotime(
    const Sparse<T, IsCSR>& adjacency,
    Index root_cell,
    Array<Real> pseudotime
);

// TODO: Geodesic distance pseudotime
template <typename T, bool IsCSR>
void geodesic_pseudotime(
    const Sparse<T, IsCSR>& adjacency,
    Index root_cell,
    Array<Real> pseudotime
);

// =============================================================================
// Root Cell Selection (TODO: Implementation)
// =============================================================================

// TODO: Automatic root cell selection
template <typename T, bool IsCSR>
Index select_root_cell(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> gene_expression,  // Marker gene expression
    const std::vector<Index>& candidate_cells = {}
);

// TODO: Multiple root detection
template <typename T, bool IsCSR>
void detect_roots(
    const Sparse<T, IsCSR>& adjacency,
    std::vector<Index>& root_cells,
    Index max_roots = 3
);

// =============================================================================
// Branch Detection (TODO: Implementation)
// =============================================================================

// TODO: Detect branch points
template <typename T, bool IsCSR>
void detect_branches(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> pseudotime,
    std::vector<Index>& branch_points,
    Real threshold = config::DEFAULT_THRESHOLD
);

// TODO: Segment trajectory into branches
template <typename T, bool IsCSR>
void segment_trajectory(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> pseudotime,
    Array<const Index> branch_points,
    Array<Index> segment_labels  // Output: which segment each cell belongs to
);

// TODO: Branch assignment with probabilities
template <typename T, bool IsCSR>
void branch_assignment_probabilistic(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> pseudotime,
    Sparse<Real, !IsCSR>& branch_probabilities  // Cells × Branches
);

// =============================================================================
// Trajectory Refinement (TODO: Implementation)
// =============================================================================

// TODO: Smooth pseudotime along graph
template <typename T, bool IsCSR>
void smooth_pseudotime(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> pseudotime,
    Index n_iterations = 10
);

// TODO: Refine trajectory backbone
template <typename T, bool IsCSR>
void refine_backbone(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> pseudotime,
    std::vector<Index>& backbone_cells
);

// =============================================================================
// Helper Functions (TODO: Implementation)
// =============================================================================

namespace detail {

// TODO: Compute diffusion map
template <typename T, bool IsCSR>
void diffusion_map(
    const Sparse<T, IsCSR>& transition_matrix,
    Index n_components,
    Sparse<Real, !IsCSR>& diffusion_components
);

// TODO: Detect bifurcation points
bool is_bifurcation(
    Index cell,
    const Sparse<Real, true>& adjacency,
    const Real* pseudotime,
    Real threshold
);

} // namespace detail

} // namespace scl::kernel::pseudotime

