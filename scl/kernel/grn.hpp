#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
// FILE: scl/kernel/grn.hpp
// BRIEF: Gene regulatory network inference
//
// STRATEGIC POSITION: Sparse + Nonlinear (Tier 2 - Regulatory)
// - Network inference from expression
// - Sparse network construction
// - TF-target prediction
//
// APPLICATIONS:
// - SCENIC-style analysis
// - Regulatory network construction
// - TF activity inference
// - Regulon scoring
// =============================================================================

namespace scl::kernel::grn {

// =============================================================================
// Correlation Network (TODO: Implementation)
// =============================================================================

// TODO: Build correlation-based network
template <typename T, bool IsCSR>
void correlation_network(
    const Sparse<T, IsCSR>& expression,
    Real threshold,
    Sparse<Real, true>& network
);

// TODO: Partial correlation network
template <typename T, bool IsCSR>
void partial_correlation_network(
    const Sparse<T, IsCSR>& expression,
    Real threshold,
    Sparse<Real, true>& network
);

// =============================================================================
// TF-Target Scoring (TODO: Implementation)
// =============================================================================

// TODO: Score TF-target relationships
template <typename T, bool IsCSR>
void tf_target_score(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> tf_genes,
    Array<const Index> target_genes,
    Sparse<Real, true>& tf_target_scores
);

// TODO: GENIE3-style importance scoring
template <typename T, bool IsCSR>
void genie3_importance(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> tf_genes,
    Sparse<Real, true>& importance_matrix
);

// =============================================================================
// Regulon Activity (TODO: Implementation)
// =============================================================================

// TODO: Compute regulon activity scores
template <typename T, bool IsCSR>
void regulon_activity(
    const Sparse<T, IsCSR>& expression,
    const std::vector<std::vector<Index>>& regulons,
    Sparse<Real, !IsCSR>& activity_scores
);

// =============================================================================
// Helper Functions (TODO: Implementation)
// =============================================================================

namespace detail {

// TODO: Mutual information for gene pairs
template <typename T>
Real mutual_information_genes(
    const T* gene1_expr,
    const T* gene2_expr,
    Size n_cells,
    Index n_bins = 10
);

} // namespace detail

} // namespace scl::kernel::grn

