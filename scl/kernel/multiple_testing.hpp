#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"

// =============================================================================
// FILE: scl/kernel/multiple_testing.hpp
// BRIEF: Multiple testing correction
//
// APPLICATIONS:
// - FDR control
// - P-value adjustment
// - Q-value estimation
// =============================================================================

namespace scl::kernel::multiple_testing {

// TODO: Benjamini-Hochberg FDR
void benjamini_hochberg(
    Array<const Real> p_values,
    Array<Real> adjusted_p_values,
    Real fdr_level = Real(0.05)
);

// TODO: Bonferroni correction
void bonferroni(
    Array<const Real> p_values,
    Array<Real> adjusted_p_values
);

// TODO: Storey q-value
void storey_qvalue(
    Array<const Real> p_values,
    Array<Real> q_values,
    Real lambda = Real(0.5)
);

// TODO: Local FDR
void local_fdr(
    Array<const Real> p_values,
    Array<Real> lfdr
);

// TODO: Empirical FDR (permutation-based)
void empirical_fdr(
    Array<const Real> observed_scores,
    const std::vector<Array<Real>>& permuted_scores,
    Array<Real> fdr
);

} // namespace scl::kernel::multiple_testing

