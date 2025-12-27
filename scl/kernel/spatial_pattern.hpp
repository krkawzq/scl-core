#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"

// =============================================================================
// FILE: scl/kernel/spatial_pattern.hpp
// BRIEF: Spatial pattern detection (SpatialDE-style)
//
// APPLICATIONS:
// - Spatially variable genes
// - Spatial gradients
// - Periodic patterns
// - Domain identification
// =============================================================================

namespace scl::kernel::spatial_pattern {

// TODO: Detect spatially variable genes
template <typename T, bool IsCSR>
void spatial_variability(
    const Sparse<T, IsCSR>& expression,
    const Sparse<Real, !IsCSR>& coordinates,
    std::vector<Index>& variable_genes,
    std::vector<Real>& variability_scores
);

// TODO: Spatial gradient detection
template <typename T, bool IsCSR>
void spatial_gradient(
    const Sparse<T, IsCSR>& expression,
    const Sparse<Real, !IsCSR>& coordinates,
    Index gene,
    Array<Real> gradient_direction,
    Real& gradient_strength
);

// TODO: Periodic pattern detection
template <typename T, bool IsCSR>
void periodic_pattern(
    const Sparse<T, IsCSR>& expression,
    const Sparse<Real, !IsCSR>& coordinates,
    std::vector<Index>& periodic_genes
);

// TODO: Boundary detection
template <typename T, bool IsCSR>
void boundary_detection(
    const Sparse<T, IsCSR>& expression,
    const Sparse<Real, !IsCSR>& coordinates,
    Array<Real> boundary_scores
);

// TODO: Spatial domain identification
template <typename T, bool IsCSR>
void spatial_domain(
    const Sparse<T, IsCSR>& expression,
    const Sparse<Real, !IsCSR>& coordinates,
    Index n_domains,
    Array<Index> domain_labels
);

} // namespace scl::kernel::spatial_pattern

