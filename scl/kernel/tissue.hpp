#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"

// =============================================================================
// FILE: scl/kernel/tissue.hpp
// BRIEF: Tissue architecture and organization analysis
//
// APPLICATIONS:
// - Tissue structure quantification
// - Layer assignment
// - Zonation scoring
// =============================================================================

namespace scl::kernel::tissue {

// TODO: Tissue architecture quantification
template <typename T, bool IsCSR>
void tissue_architecture(
    const Sparse<Real, !IsCSR>& coordinates,
    Array<const Index> cell_types,
    Sparse<Real, true>& architecture_features
);

// TODO: Layer assignment
template <typename T, bool IsCSR>
void layer_assignment(
    const Sparse<Real, !IsCSR>& coordinates,
    Index n_layers,
    Array<Index> layer_labels
);

// TODO: Zonation score
template <typename T, bool IsCSR>
void zonation_score(
    const Sparse<T, IsCSR>& expression,
    const Sparse<Real, !IsCSR>& coordinates,
    Array<Real> zonation_scores
);

// TODO: Morphological features
void morphological_features(
    const Sparse<Real, !IsCSR>& coordinates,
    Array<const Index> labels,
    Sparse<Real, !IsCSR>& features
);

// TODO: Tissue module detection
template <typename T, bool IsCSR>
void tissue_module(
    const Sparse<T, IsCSR>& expression,
    const Sparse<Real, !IsCSR>& coordinates,
    std::vector<std::vector<Index>>& modules
);

} // namespace scl::kernel::tissue

