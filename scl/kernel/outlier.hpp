#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"

// =============================================================================
// FILE: scl/kernel/outlier.hpp
// BRIEF: Outlier and anomaly detection
//
// APPLICATIONS:
// - Quality control
// - Ambient RNA detection
// - Empty droplet detection
// - Outlier gene identification
// =============================================================================

namespace scl::kernel::outlier {

// TODO: Isolation score
template <typename T, bool IsCSR>
void isolation_score(
    const Sparse<T, IsCSR>& data,
    Array<Real> scores
);

// TODO: Local Outlier Factor (LOF)
template <typename T, bool IsCSR>
void local_outlier_factor(
    const Sparse<T, IsCSR>& data,
    const Sparse<Index, IsCSR>& neighbors,
    const Sparse<Real, IsCSR>& distances,
    Array<Real> lof_scores
);

// TODO: Ambient RNA detection
template <typename T, bool IsCSR>
void ambient_detection(
    const Sparse<T, IsCSR>& expression,
    Array<Real> ambient_scores
);

// TODO: Empty droplet detection
template <typename T, bool IsCSR>
void empty_drops(
    const Sparse<T, IsCSR>& raw_counts,
    Array<bool> is_empty,
    Real fdr_threshold = Real(0.01)
);

// TODO: Outlier gene detection
template <typename T, bool IsCSR>
void outlier_genes(
    const Sparse<T, IsCSR>& expression,
    std::vector<Index>& outlier_gene_indices,
    Real threshold = Real(3.0)
);

} // namespace scl::kernel::outlier

