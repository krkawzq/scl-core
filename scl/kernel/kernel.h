// =============================================================================
// FILE: scl/kernel/kernel.h
// BRIEF: API reference for sparse kernel methods including KDE, kernel functions, and kernel ops
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::kernel {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_BANDWIDTH = Real(1.0);
    constexpr Real MIN_BANDWIDTH = Real(1e-10);
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size SIMD_THRESHOLD = 32;
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Real LOG_MIN = Real(1e-300);
    constexpr Index DEFAULT_K_NEIGHBORS = 15;
    constexpr Index NYSTROM_MAX_ITER = 50;
}

// =============================================================================
// Kernel Types
// =============================================================================

enum class KernelType {
    Gaussian,
    Epanechnikov,
    Cosine,
    Linear,
    Polynomial,
    Laplacian,
    Cauchy,
    Sigmoid,
    Uniform,
    Triangular
};

// =============================================================================
// Kernel Density Estimation
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: kernel_density_estimation
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute kernel density estimate at query points.
 *
 * PARAMETERS:
 *     points      [in]  Data points [n_points * n_dims]
 *     queries     [in]  Query points [n_queries * n_dims]
 *     n_points    [in]  Number of data points
 *     n_queries   [in]  Number of query points
 *     n_dims      [in]  Number of dimensions
 *     densities   [out] Density estimates [n_queries]
 *     bandwidth   [in]  Kernel bandwidth
 *     kernel_type [in]  Kernel function type
 *
 * PRECONDITIONS:
 *     - densities has capacity >= n_queries
 *
 * POSTCONDITIONS:
 *     - densities[i] contains KDE at query point i
 *
 * COMPLEXITY:
 *     Time:  O(n_queries * n_points * n_dims)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over queries
 * -------------------------------------------------------------------------- */
void kernel_density_estimation(
    const Real* points,                      // Data points [n_points * n_dims]
    const Real* queries,                     // Query points [n_queries * n_dims]
    Size n_points,                           // Number of data points
    Size n_queries,                          // Number of query points
    Size n_dims,                             // Number of dimensions
    Array<Real> densities,                    // Output densities [n_queries]
    Real bandwidth = config::DEFAULT_BANDWIDTH, // Bandwidth
    KernelType kernel_type = KernelType::Gaussian // Kernel type
);

/* -----------------------------------------------------------------------------
 * FUNCTION: kernel_matrix
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute kernel matrix between two point sets.
 *
 * PARAMETERS:
 *     points1      [in]  First point set [n1 * n_dims]
 *     points2      [in]  Second point set [n2 * n_dims]
 *     n1           [in]  Number of points in first set
 *     n2           [in]  Number of points in second set
 *     n_dims       [in]  Number of dimensions
 *     kernel_mat   [out] Kernel matrix [n1 * n2]
 *     bandwidth    [in]  Kernel bandwidth
 *     kernel_type  [in]  Kernel function type
 *
 * PRECONDITIONS:
 *     - kernel_mat has capacity >= n1 * n2
 *
 * POSTCONDITIONS:
 *     - kernel_mat[i * n2 + j] contains kernel value K(points1[i], points2[j])
 *
 * COMPLEXITY:
 *     Time:  O(n1 * n2 * n_dims)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over matrix elements
 * -------------------------------------------------------------------------- */
void kernel_matrix(
    const Real* points1,                     // First point set [n1 * n_dims]
    const Real* points2,                     // Second point set [n2 * n_dims]
    Size n1,                                 // Number of points in set 1
    Size n2,                                 // Number of points in set 2
    Size n_dims,                             // Number of dimensions
    Real* kernel_mat,                         // Output kernel matrix [n1 * n2]
    Real bandwidth = config::DEFAULT_BANDWIDTH, // Bandwidth
    KernelType kernel_type = KernelType::Gaussian // Kernel type
);

} // namespace scl::kernel::kernel

