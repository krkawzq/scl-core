# kernel.hpp

> scl/kernel/kernel.hpp · Kernel density estimation and kernel matrix computation

## Overview

This file provides kernel methods for density estimation and kernel matrix computation. It supports multiple kernel types including Gaussian, Epanechnikov, Cosine, Linear, Polynomial, Laplacian, Cauchy, Sigmoid, Uniform, and Triangular kernels.

**Header**: `#include "scl/kernel/kernel.hpp"`

---

## Main APIs

### kernel_density_estimation

::: source_code file="scl/kernel/kernel.hpp" symbol="kernel_density_estimation" collapsed
:::

**Algorithm Description**

Compute kernel density estimate at query points using kernel function:

1. For each query point q:
   - Compute distance to all data points
   - Apply kernel function K(||q - p|| / h) where h is bandwidth
   - Sum kernel values and normalize by number of data points
   - Store density estimate

2. Parallel processing over query points for efficiency

3. Supports multiple kernel types with optimized implementations

**Edge Cases**

- **Empty data points**: Returns zero densities for all queries
- **Empty queries**: Returns immediately without computation
- **Zero bandwidth**: Clamped to MIN_BANDWIDTH (1e-10) to avoid division by zero
- **Very large distances**: Kernel values approach zero for compact kernels

**Data Guarantees (Preconditions)**

- `points` must be valid pointer or nullptr (if n_points == 0)
- `queries` must be valid pointer or nullptr (if n_queries == 0)
- `densities` must have capacity >= n_queries
- Data layout: points[i * n_dims + j] is dimension j of point i
- Query layout: queries[i * n_dims + j] is dimension j of query i

**Complexity Analysis**

- **Time**: O(n_queries * n_points * n_dims) - each query compares to all points
- **Space**: O(1) auxiliary - only temporary distance calculations

**Example**

```cpp
#include "scl/kernel/kernel.hpp"

// Prepare data
const Real* points = /* data points [n_points * n_dims] */;
const Real* queries = /* query points [n_queries * n_dims] */;
Size n_points = 1000;
Size n_queries = 100;
Size n_dims = 10;

// Pre-allocate output
Array<Real> densities(n_queries);

// Compute KDE with Gaussian kernel
scl::kernel::kernel::kernel_density_estimation(
    points, queries,
    n_points, n_queries, n_dims,
    densities,
    bandwidth = 1.0,
    kernel_type = scl::kernel::kernel::KernelType::Gaussian
);

// densities[i] now contains KDE at query point i
for (Size i = 0; i < n_queries; ++i) {
    std::cout << "Query " << i << " density: " << densities[i] << "\n";
}
```

---

### kernel_matrix

::: source_code file="scl/kernel/kernel.hpp" symbol="kernel_matrix" collapsed
:::

**Algorithm Description**

Compute kernel matrix K between two point sets where K[i, j] = kernel(points1[i], points2[j]):

1. For each pair (i, j) of points from sets 1 and 2:
   - Compute distance between points1[i] and points2[j]
   - Apply kernel function with bandwidth
   - Store result in kernel_mat[i * n2 + j]

2. Parallel processing over matrix elements for efficiency

3. Supports all kernel types with optimized distance computations

**Edge Cases**

- **Empty point set 1**: Returns zero matrix
- **Empty point set 2**: Returns zero matrix
- **Zero bandwidth**: Clamped to MIN_BANDWIDTH to avoid numerical issues
- **Identical point sets**: Computes self-kernel matrix (symmetric for some kernels)

**Data Guarantees (Preconditions)**

- `points1` must be valid pointer or nullptr (if n1 == 0)
- `points2` must be valid pointer or nullptr (if n2 == 0)
- `kernel_mat` must have capacity >= n1 * n2
- Data layout: points[i * n_dims + j] is dimension j of point i
- Output layout: kernel_mat[i * n2 + j] is kernel value K(points1[i], points2[j])

**Complexity Analysis**

- **Time**: O(n1 * n2 * n_dims) - compute all pairwise kernel values
- **Space**: O(1) auxiliary - only temporary distance calculations

**Example**

```cpp
#include "scl/kernel/kernel.hpp"

// Prepare two point sets
const Real* points1 = /* first set [n1 * n_dims] */;
const Real* points2 = /* second set [n2 * n_dims] */;
Size n1 = 100;
Size n2 = 200;
Size n_dims = 10;

// Pre-allocate kernel matrix
Array<Real> kernel_mat(n1 * n2);

// Compute kernel matrix with Epanechnikov kernel
scl::kernel::kernel::kernel_matrix(
    points1, points2,
    n1, n2, n_dims,
    kernel_mat.ptr,
    bandwidth = 1.5,
    kernel_type = scl::kernel::kernel::KernelType::Epanechnikov
);

// kernel_mat[i * n2 + j] contains K(points1[i], points2[j])
// Use for kernel methods like SVM, kernel PCA, etc.
```

---

## Kernel Types

The following kernel types are supported:

- **Gaussian**: exp(-0.5 * (||x-y||/h)^2) - Smooth, unbounded support
- **Epanechnikov**: max(0, 1 - (||x-y||/h)^2) - Compact support, efficient
- **Cosine**: cos(π * ||x-y|| / (2*h)) - Compact support, smooth
- **Linear**: max(0, 1 - ||x-y||/h) - Compact support, simple
- **Polynomial**: (1 + x·y)^d - For polynomial kernels (requires inner product)
- **Laplacian**: exp(-||x-y||/h) - Exponential decay
- **Cauchy**: 1 / (1 + (||x-y||/h)^2) - Heavy-tailed distribution
- **Sigmoid**: tanh(α * x·y + c) - Neural network style
- **Uniform**: 1 if ||x-y|| < h, else 0 - Step function
- **Triangular**: max(0, 1 - ||x-y||/h) - Linear decay

---

## Configuration

### Default Parameters

```cpp
namespace scl::kernel::kernel::config {
    constexpr Real DEFAULT_BANDWIDTH = Real(1.0);
    constexpr Real MIN_BANDWIDTH = Real(1e-10);
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size SIMD_THRESHOLD = 32;
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Real LOG_MIN = Real(1e-300);
    constexpr Index DEFAULT_K_NEIGHBORS = 15;
    constexpr Index NYSTROM_MAX_ITER = 50;
}
```

---

## Notes

**Bandwidth Selection**: Choose bandwidth based on data scale. Too small bandwidth causes overfitting (spiky density), too large causes oversmoothing. Use cross-validation or rule-of-thumb methods (e.g., Silverman's rule) for optimal selection.

**Kernel Choice**: Gaussian is most common for smooth densities. Epanechnikov is optimal for mean squared error but has compact support. Choose based on application requirements.

**Performance**: Parallelization is enabled when n_queries >= PARALLEL_THRESHOLD (500) or n1 * n2 >= PARALLEL_THRESHOLD for kernel matrix.

---

## See Also

- [Neighbors](/cpp/kernels/neighbors) - K-nearest neighbors
- [BBKNN](/cpp/kernels/bbknn) - Batch-balanced KNN
