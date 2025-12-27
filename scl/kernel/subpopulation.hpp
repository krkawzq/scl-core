#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"

// =============================================================================
// FILE: scl/kernel/subpopulation.hpp
// BRIEF: Subpopulation analysis and cluster refinement
//
// APPLICATIONS:
// - Recursive sub-clustering
// - Cluster stability assessment
// - Rare cell detection
// =============================================================================

namespace scl::kernel::subpopulation {

// TODO: Recursive sub-clustering
template <typename T, bool IsCSR>
void subclustering(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> parent_labels,
    Index parent_cluster,
    Array<Index> subcluster_labels
);

// TODO: Cluster stability via bootstrap
template <typename T, bool IsCSR>
void cluster_stability(
    const Sparse<T, IsCSR>& expression,
    Index n_bootstraps,
    Array<Real> stability_scores
);

// TODO: Cluster purity
void cluster_purity(
    Array<const Index> cluster_labels,
    Array<const Index> true_labels,
    Array<Real> purity_per_cluster
);

// TODO: Rare cell detection
template <typename T, bool IsCSR>
void rare_cell_detection(
    const Sparse<T, IsCSR>& expression,
    const Sparse<Index, IsCSR>& neighbors,
    Array<Real> rarity_scores
);

// TODO: Population balance
void population_balance(
    Array<const Index> labels,
    Array<const Index> condition,
    Sparse<Real, true>& balance_matrix
);

} // namespace scl::kernel::subpopulation

