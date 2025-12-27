#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"

// =============================================================================
// FILE: scl/kernel/metrics.hpp
// BRIEF: Quality metrics for clustering and integration
//
// APPLICATIONS:
// - Clustering evaluation
// - Integration quality
// - Batch mixing assessment
// =============================================================================

namespace scl::kernel::metrics {

// TODO: Silhouette score
template <typename T, bool IsCSR>
Real silhouette_score(
    const Sparse<T, IsCSR>& distances,
    Array<const Index> labels
);

// TODO: Adjusted Rand Index (ARI)
Real adjusted_rand_index(
    Array<const Index> labels1,
    Array<const Index> labels2
);

// TODO: Normalized Mutual Information (NMI)
Real normalized_mutual_information(
    Array<const Index> labels1,
    Array<const Index> labels2
);

// TODO: Graph connectivity
template <typename T, bool IsCSR>
Real graph_connectivity(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Index> labels
);

// TODO: Batch entropy
void batch_entropy(
    const Sparse<Index, IsCSR>& neighbors,
    Array<const Index> batch_labels,
    Array<Real> entropy_scores
);

// TODO: Local inverse Simpson's index (LISI)
void lisi(
    const Sparse<Index, IsCSR>& neighbors,
    Array<const Index> labels,
    Array<Real> lisi_scores
);

} // namespace scl::kernel::metrics

