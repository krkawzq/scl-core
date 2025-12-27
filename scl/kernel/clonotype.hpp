#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"

// =============================================================================
// FILE: scl/kernel/clonotype.hpp
// BRIEF: TCR/BCR clonal analysis
//
// APPLICATIONS:
// - Clonal expansion analysis
// - Clonal diversity
// - Clone-phenotype association
// =============================================================================

namespace scl::kernel::clonotype {

// TODO: Clone size distribution
void clone_size_distribution(
    Array<const Index> clone_ids,
    std::vector<Size>& clone_sizes
);

// TODO: Clonal diversity indices
void clonal_diversity(
    Array<const Index> clone_ids,
    Real& shannon_diversity,
    Real& simpson_diversity,
    Real& gini_index
);

// TODO: Clone dynamics
void clone_dynamics(
    Array<const Index> clone_ids_t1,
    Array<const Index> clone_ids_t2,
    std::vector<Real>& expansion_rates
);

// TODO: Shared clonotypes
void shared_clonotypes(
    Array<const Index> clone_ids_sample1,
    Array<const Index> clone_ids_sample2,
    std::vector<Index>& shared_clones,
    Real& jaccard_index
);

// TODO: Clone-phenotype association
template <typename T, bool IsCSR>
void clone_phenotype(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> clone_ids,
    Sparse<Real, !IsCSR>& clone_profiles
);

} // namespace scl::kernel::clonotype

