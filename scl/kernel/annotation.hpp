#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"

// =============================================================================
// FILE: scl/kernel/annotation.hpp
// BRIEF: Cell type annotation from reference
//
// APPLICATIONS:
// - Automated cell type annotation
// - Reference mapping
// - Transfer learning
// =============================================================================

namespace scl::kernel::annotation {

// TODO: Reference mapping
template <typename T, bool IsCSR>
void reference_mapping(
    const Sparse<T, IsCSR>& query_expression,
    const Sparse<T, IsCSR>& reference_expression,
    Array<const Index> reference_labels,
    const Sparse<Index, IsCSR>& query_to_ref_neighbors,
    Array<Index> query_labels,
    Array<Real> confidence_scores
);

// TODO: Correlation-based assignment
template <typename T, bool IsCSR>
void correlation_assignment(
    const Sparse<T, IsCSR>& query_expression,
    const Sparse<T, IsCSR>& reference_profiles,  // Cell type × Genes
    Array<Index> assigned_labels,
    Array<Real> correlation_scores
);

// TODO: Consensus annotation
void consensus_annotation(
    const std::vector<Array<Index>>& predictions,
    Array<Index> consensus_labels,
    Array<Real> consensus_confidence
);

// TODO: Novel type detection
template <typename T, bool IsCSR>
void detect_novel_types(
    const Sparse<T, IsCSR>& query_expression,
    Array<const Real> confidence_scores,
    Real threshold,
    Array<bool> is_novel
);

} // namespace scl::kernel::annotation

