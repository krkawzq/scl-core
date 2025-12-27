#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"

// =============================================================================
// FILE: scl/kernel/sampling.hpp
// BRIEF: Advanced sampling strategies for large datasets
//
// APPLICATIONS:
// - Geometric sketching
// - Density-preserving downsampling
// - Landmark selection
// =============================================================================

namespace scl::kernel::sampling {

// TODO: Geometric sketching
template <typename T, bool IsCSR>
void geometric_sketching(
    const Sparse<T, IsCSR>& data,
    Size target_size,
    std::vector<Index>& selected_indices,
    uint64_t seed = 42
);

// TODO: Density-preserving sampling
template <typename T, bool IsCSR>
void density_preserving(
    const Sparse<T, IsCSR>& data,
    const Sparse<Index, IsCSR>& neighbors,
    Size target_size,
    std::vector<Index>& selected_indices
);

// TODO: Landmark selection
template <typename T, bool IsCSR>
void landmark_selection(
    const Sparse<T, IsCSR>& data,
    Size n_landmarks,
    std::vector<Index>& landmark_indices
);

// TODO: Representative cells
template <typename T, bool IsCSR>
void representative_cells(
    const Sparse<T, IsCSR>& data,
    Array<const Index> cluster_labels,
    Size per_cluster,
    std::vector<Index>& representatives
);

// TODO: Balanced sampling
void balanced_sampling(
    Array<const Index> labels,
    Size target_size,
    std::vector<Index>& selected_indices,
    uint64_t seed = 42
);

} // namespace scl::kernel::sampling

