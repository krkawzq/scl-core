#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"

// =============================================================================
// FILE: scl/kernel/state.hpp
// BRIEF: Cell state scoring (stemness, differentiation, proliferation)
//
// APPLICATIONS:
// - Stemness quantification
// - Differentiation potential (CytoTRACE-style)
// - Proliferation/stress scoring
// =============================================================================

namespace scl::kernel::state {

// TODO: Stemness score
template <typename T, bool IsCSR>
void stemness_score(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> stemness_genes,
    Array<Real> scores
);

// TODO: Differentiation potential (CytoTRACE-style)
template <typename T, bool IsCSR>
void differentiation_potential(
    const Sparse<T, IsCSR>& expression,
    Array<Real> potency_scores
);

// TODO: Proliferation score
template <typename T, bool IsCSR>
void proliferation_score(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> proliferation_genes,
    Array<Real> scores
);

// TODO: Stress score
template <typename T, bool IsCSR>
void stress_score(
    const Sparse<T, IsCSR>& expression,
    Array<const Index> stress_genes,
    Array<Real> scores
);

// TODO: State entropy (plasticity)
template <typename T, bool IsCSR>
void state_entropy(
    const Sparse<T, IsCSR>& expression,
    Array<Real> entropy_scores
);

} // namespace scl::kernel::state

