#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"

// =============================================================================
// FILE: scl/kernel/alignment.hpp
// BRIEF: Multi-modal data alignment and integration
//
// APPLICATIONS:
// - Batch correction
// - Multi-modal integration
// - MNN-based alignment
// =============================================================================

namespace scl::kernel::alignment {

// TODO: Mutual nearest neighbors (MNN) pairs
template <typename T, bool IsCSR>
void mnn_pairs(
    const Sparse<T, IsCSR>& data1,
    const Sparse<T, IsCSR>& data2,
    Index k,
    std::vector<std::pair<Index, Index>>& mnn_pairs_out
);

// TODO: Anchor finding
template <typename T, bool IsCSR>
void find_anchors(
    const Sparse<T, IsCSR>& data1,
    const Sparse<T, IsCSR>& data2,
    Index k,
    std::vector<std::pair<Index, Index>>& anchors
);

// TODO: Label transfer via anchors
void transfer_labels(
    const std::vector<std::pair<Index, Index>>& anchors,
    Array<const Index> source_labels,
    Array<Index> target_labels
);

// TODO: Integration quality score
template <typename T, bool IsCSR>
Real integration_score(
    const Sparse<T, IsCSR>& integrated_data,
    Array<const Index> batch_labels,
    const Sparse<Index, IsCSR>& neighbors
);

// TODO: Batch mixing metric
void batch_mixing(
    Array<const Index> batch_labels,
    const Sparse<Index, IsCSR>& neighbors,
    Array<Real> mixing_scores
);

} // namespace scl::kernel::alignment

