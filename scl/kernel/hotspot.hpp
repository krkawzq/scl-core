#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/simd.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/hotspot.hpp
// BRIEF: Local spatial statistics and hotspot analysis
//
// STRATEGIC POSITION: Sparse + Nonlinear (Tier 2 - Spatial Extension)
// - Extends existing spatial.hpp module
// - Local spatial autocorrelation
// - Hotspot/coldspot detection
// - Spatial cluster classification
//
// APPLICATIONS:
// - Detect spatially variable genes (local Moran's I)
// - Find hot spots of gene expression (Gi*)
// - Classify spatial patterns (HH, HL, LH, LL)
// - Spatial outlier detection
//
// METHODS:
// - LISA (Local Indicators of Spatial Association)
// - Getis-Ord Gi* (hot/cold spot detection)
// - Local Geary's C
// - Spatial cluster classification
// =============================================================================

namespace scl::kernel::hotspot {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real SIGNIFICANCE_LEVEL = Real(0.05);
    constexpr Index DEFAULT_PERMUTATIONS = 999;
}

// =============================================================================
// Spatial Pattern Types
// =============================================================================

enum class SpatialPattern {
    NOT_SIGNIFICANT = 0,
    HIGH_HIGH = 1,          // High value, high neighbors
    LOW_LOW = 2,            // Low value, low neighbors
    HIGH_LOW = 3,           // High value, low neighbors (outlier)
    LOW_HIGH = 4            // Low value, high neighbors (outlier)
};

// =============================================================================
// LISA - Local Moran's I (TODO: Implementation)
// =============================================================================

// TODO: Compute local Moran's I for each location
template <typename T, bool IsCSR>
void local_morans_i(
    const Sparse<T, IsCSR>& spatial_weights,
    Array<const Real> values,
    Array<Real> local_i,                    // Local Moran's I per location
    Array<Real> p_values,                   // P-values (optional permutation)
    Index n_permutations = 0                // 0 = no permutation test
);

// TODO: Classify LISA patterns (HH, HL, LH, LL)
void classify_lisa_patterns(
    Array<const Real> values,
    Array<const Real> local_i,
    Array<const Real> p_values,
    Real significance_level,
    Array<SpatialPattern> patterns
);

// =============================================================================
// Getis-Ord Gi* Statistic (TODO: Implementation)
// =============================================================================

// TODO: Compute Gi* statistic for hotspot detection
template <typename T, bool IsCSR>
void getis_ord_g_star(
    const Sparse<T, IsCSR>& spatial_weights,
    Array<const Real> values,
    Array<Real> g_star,                     // Gi* statistic per location
    Array<Real> z_scores,                   // Z-scores
    Array<Real> p_values                    // P-values
);

// TODO: Identify hotspots and coldspots
void identify_hotspots(
    Array<const Real> z_scores,
    Real significance_level,
    Array<int8_t> classification            // -1=cold, 0=not sig, 1=hot
);

// =============================================================================
// Local Geary's C (TODO: Implementation)
// =============================================================================

// TODO: Compute local Geary's C
template <typename T, bool IsCSR>
void local_gearys_c(
    const Sparse<T, IsCSR>& spatial_weights,
    Array<const Real> values,
    Array<Real> local_c,                    // Local Geary's C per location
    Array<Real> p_values
);

// =============================================================================
// Spatial Clustering (TODO: Implementation)
// =============================================================================

// TODO: Detect spatial clusters using local statistics
template <typename T, bool IsCSR>
void detect_spatial_clusters(
    const Sparse<T, IsCSR>& spatial_weights,
    Array<const Real> values,
    Real significance_level,
    Array<Index> cluster_labels,            // Output cluster IDs
    Index& n_clusters
);

// TODO: Spatial scan statistic (detect arbitrary shaped clusters)
template <typename T, bool IsCSR>
void spatial_scan_statistic(
    const Sparse<T, IsCSR>& spatial_weights,
    Array<const Real> values,
    Size max_cluster_size,
    std::vector<std::vector<Index>>& clusters,
    std::vector<Real>& cluster_scores
);

// =============================================================================
// Multi-Feature Hotspot Analysis (TODO: Implementation)
// =============================================================================

// TODO: Compute LISA for multiple features (genes) in parallel
template <typename T, bool IsCSR>
void multivariate_lisa(
    const Sparse<T, IsCSR>& spatial_weights,
    const Sparse<T, !IsCSR>& feature_matrix,  // Features × Locations
    Sparse<Real, !IsCSR>& local_i_matrix,      // Results per feature
    Sparse<Real, !IsCSR>& p_value_matrix
);

// TODO: Identify spatially variable features
template <typename T, bool IsCSR>
void spatial_variable_features(
    const Sparse<T, IsCSR>& spatial_weights,
    const Sparse<T, !IsCSR>& feature_matrix,
    Real significance_threshold,
    std::vector<Index>& variable_features,
    std::vector<Real>& spatial_scores
);

// =============================================================================
// Helper Functions (TODO: Implementation)
// =============================================================================

namespace detail {

// TODO: Compute local statistic for single location
template <typename T>
Real compute_local_moran(
    Index location,
    const Real* values,
    const Index* neighbor_indices,
    const T* neighbor_weights,
    Size n_neighbors,
    Real mean,
    Real variance
);

// TODO: Compute Gi* for single location
template <typename T>
Real compute_g_star(
    Index location,
    const Real* values,
    const Index* neighbor_indices,
    const T* neighbor_weights,
    Size n_neighbors,
    Real global_mean,
    Real global_std,
    Index n_total
);

// TODO: Permutation test for local statistic
Real permutation_test_local(
    Real observed_statistic,
    const std::vector<Real>& null_distribution
);

// TODO: Classify single location pattern
SpatialPattern classify_pattern(
    Real value,
    Real spatial_lag,
    Real mean,
    Real local_i,
    Real p_value,
    Real significance_level
);

} // namespace detail

} // namespace scl::kernel::hotspot

