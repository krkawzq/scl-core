#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/impute.hpp
// BRIEF: Sparse-aware imputation methods
//
// STRATEGIC POSITION: Sparse + Nonlinear (Core Battlefield)
// - Leverages existing neighbors module
// - Maintains sparsity (only imputes dropouts, doesn't fill entire matrix)
// - Nonlinear weighted aggregation
// - Unique advantage: most libraries convert to dense
//
// KEY INSIGHT:
// Only impute zero values (likely dropouts), preserve non-zeros
// Output remains sparse (slightly denser than input)
// =============================================================================

namespace scl::kernel::impute {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_THRESHOLD = Real(0.0);
    constexpr Real DISTANCE_EPSILON = Real(1e-12);
    constexpr Size PREFETCH_DISTANCE = 16;
}

// =============================================================================
// KNN-Based Imputation (TODO: Implementation)
// =============================================================================

// TODO: Implement KNN imputation that preserves sparsity
template <typename T, bool IsCSR>
void knn_impute(
    const Sparse<T, IsCSR>& X,
    const Sparse<Index, IsCSR>& knn_indices,    // From neighbors module
    const Sparse<Real, IsCSR>& knn_distances,
    Sparse<T, IsCSR>& X_imputed,
    Real threshold = config::DEFAULT_THRESHOLD
);

// TODO: Implement weighted KNN imputation with custom weights
template <typename T, bool IsCSR>
void knn_impute_weighted(
    const Sparse<T, IsCSR>& X,
    const Sparse<Index, IsCSR>& knn_indices,
    const Sparse<Real, IsCSR>& knn_weights,     // Custom weights
    Sparse<T, IsCSR>& X_imputed,
    Real threshold = config::DEFAULT_THRESHOLD
);

// TODO: Implement MAGIC-like diffusion imputation
template <typename T, bool IsCSR>
void diffusion_impute(
    const Sparse<T, IsCSR>& X,
    const Sparse<Real, IsCSR>& transition_matrix,  // From diffusion module
    Index n_steps,
    Sparse<T, IsCSR>& X_imputed
);

// =============================================================================
// Helper Functions (TODO: Implementation)
// =============================================================================

namespace detail {

// TODO: Compute distance-based weights from distances
SCL_FORCE_INLINE Real distance_to_weight(Real distance, Real bandwidth);

// TODO: Identify positions to impute (zeros in sparse matrix)
template <typename T, bool IsCSR>
void identify_dropout_positions(
    const Sparse<T, IsCSR>& X,
    Real threshold,
    std::vector<std::pair<Index, Index>>& positions
);

// TODO: Weighted average of neighbor values
template <typename T>
T weighted_neighbor_average(
    const T* neighbor_values,
    const Real* weights,
    Size n_neighbors
);

} // namespace detail

} // namespace scl::kernel::impute

