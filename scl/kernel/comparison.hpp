#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"

// =============================================================================
// FILE: scl/kernel/comparison.hpp
// BRIEF: Group comparison and differential abundance
//
// APPLICATIONS:
// - Case-control studies
// - Composition analysis
// - Differential abundance testing
// =============================================================================

namespace scl::kernel::comparison {

// TODO: Composition analysis
void composition_analysis(
    Array<const Index> cell_types,
    Array<const Index> conditions,
    Sparse<Real, true>& proportions,
    Sparse<Real, true>& p_values
);

// TODO: Abundance test
void abundance_test(
    Array<const Index> cluster_labels,
    Array<const Index> condition,
    Array<Real> fold_changes,
    Array<Real> p_values
);

// TODO: Differential abundance (DA)
void differential_abundance(
    Array<const Index> cluster_labels,
    Array<const Index> sample_ids,
    Array<const Index> conditions,
    Array<Real> da_scores,
    Array<Real> p_values
);

// TODO: Condition response
template <typename T, bool IsCSR>
void condition_response(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> conditions,
    Sparse<Real, !IsCSR>& response_scores
);

// TODO: Effect size calculation
Real effect_size(
    Array<const Real> group1,
    Array<const Real> group2
);

} // namespace scl::kernel::comparison

