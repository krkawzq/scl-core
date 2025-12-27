#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
// FILE: scl/kernel/communication.hpp
// BRIEF: Cell-cell communication analysis (CellChat/CellPhoneDB-style)
//
// STRATEGIC POSITION: Sparse + Nonlinear (Tier 2 - Biology)
// - Ligand-receptor interaction scoring
// - Sparse expression + permutation tests
// - Pathway-level aggregation
//
// APPLICATIONS:
// - Identify active L-R pairs
// - Cell type communication networks
// - Signaling pathway analysis
// - Sender-receiver scoring
//
// KEY COMPUTATIONS:
// - Product of ligand and receptor expression
// - Permutation-based significance
// - Network construction
// =============================================================================

namespace scl::kernel::communication {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Index DEFAULT_N_PERM = 1000;
    constexpr Real DEFAULT_PVAL_THRESHOLD = Real(0.05);
}

// =============================================================================
// L-R Scoring (TODO: Implementation)
// =============================================================================

// TODO: Score ligand-receptor pairs
template <typename T, bool IsCSR>
void lr_score(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> cell_type_labels,
    const std::vector<std::pair<Index, Index>>& lr_pairs,  // (ligand, receptor)
    Sparse<Real, true>& scores  // Cell types × Cell types × LR pairs
);

// TODO: Mean expression product scoring
template <typename T, bool IsCSR>
Real lr_score_mean_product(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> sender_cells,
    Array<const Index> receiver_cells,
    Index ligand_gene,
    Index receptor_gene
);

// =============================================================================
// Communication Probability (TODO: Implementation)
// =============================================================================

// TODO: Compute communication probability
template <typename T, bool IsCSR>
void communication_probability(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> cell_type_labels,
    const std::vector<std::pair<Index, Index>>& lr_pairs,
    Sparse<Real, true>& comm_probs,
    Index n_permutations = config::DEFAULT_N_PERM
);

// TODO: Spatially-informed communication
template <typename T, bool IsCSR>
void spatial_communication(
    const Sparse<T, IsCSR>& expression,
    const Sparse<Real, IsCSR>& spatial_graph,
    const std::vector<std::pair<Index, Index>>& lr_pairs,
    Sparse<Real, true>& spatial_comm_scores
);

// =============================================================================
// Pathway Activity (TODO: Implementation)
// =============================================================================

// TODO: Aggregate L-R scores to pathway level
void pathway_activity(
    const Sparse<Real, true>& lr_scores,
    const std::vector<std::vector<Index>>& pathways,  // LR indices per pathway
    Sparse<Real, true>& pathway_scores
);

// TODO: Identify active pathways
void active_pathways(
    const Sparse<Real, true>& pathway_scores,
    Real threshold,
    std::vector<Index>& active_pathway_indices
);

// =============================================================================
// Sender-Receiver Scoring (TODO: Implementation)
// =============================================================================

// TODO: Score cell types as senders
template <typename T, bool IsCSR>
void sender_score(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> cell_type_labels,
    const std::vector<Index>& ligand_genes,
    Array<Real> sender_scores  // Per cell type
);

// TODO: Score cell types as receivers
template <typename T, bool IsCSR>
void receiver_score(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> cell_type_labels,
    const std::vector<Index>& receptor_genes,
    Array<Real> receiver_scores
);

// =============================================================================
// Communication Network (TODO: Implementation)
// =============================================================================

// TODO: Build cell type communication network
void communication_network(
    const Sparse<Real, true>& comm_probs,
    Real threshold,
    Sparse<Real, true>& network  // Directed graph: senders × receivers
);

// TODO: Network centrality for cell types
void communication_centrality(
    const Sparse<Real, true>& comm_network,
    Array<Real> in_centrality,   // Receiver importance
    Array<Real> out_centrality   // Sender importance
);

// =============================================================================
// Helper Functions (TODO: Implementation)
// =============================================================================

namespace detail {

// TODO: Permutation test for L-R significance
Real lr_permutation_test(
    Real observed_score,
    const Real* expression_ligand,
    const Real* expression_receptor,
    const Index* sender_indices,
    const Index* receiver_indices,
    Size n_senders,
    Size n_receivers,
    Index n_permutations,
    uint64_t seed
);

} // namespace detail

} // namespace scl::kernel::communication

