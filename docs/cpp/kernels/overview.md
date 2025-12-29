---
title: Kernels Overview
description: Computational kernels and algorithm modules
---

# Kernels Overview

SCL-Core provides 70+ computational kernel modules organized by functionality. Each kernel is optimized with SIMD, parallelization, and cache-friendly algorithms.

## Module Organization

Kernels are organized into the following categories:

### Data Processing
- **normalize**: Normalization operations (total count, L1, L2, log transform)
- **scale**: Scaling and standardization
- **log1p**: Log(1+x) transformation
- **softmax**: Softmax activation
- **impute**: Missing value imputation

### Neighbor Search
- **neighbors**: K-nearest neighbors search
- **bbknn**: Batch-balanced KNN
- **spatial**: Spatial neighbor search

### Statistical Analysis
- **stat**: Statistical tests (t-test, Mann-Whitney U, permutation tests)
- **metrics**: Distance metrics and similarity measures
- **correlation**: Correlation computation
- **mwu**: Mann-Whitney U test
- **ttest**: T-test implementation

### Spatial Analysis
- **spatial**: Spatial autocorrelation (Moran's I, etc.)
- **spatial_pattern**: Spatial pattern detection
- **hotspot**: Hotspot detection

### Graph Algorithms
- **louvain**: Louvain community detection
- **leiden**: Leiden algorithm
- **components**: Connected components
- **centrality**: Centrality measures
- **gnn**: Graph neural network operations

### Feature Selection
- **hvg**: Highly variable gene selection
- **markers**: Marker gene detection
- **feature**: Feature selection utilities

### Clustering and Classification
- **subpopulation**: Subpopulation identification
- **doublet**: Doublet detection
- **outlier**: Outlier detection

### Dimensionality Reduction
- **projection**: Projection methods (PCA, etc.)
- **diffusion**: Diffusion maps
- **pseudotime**: Pseudotime inference

### Other Operations
- **algebra**: Matrix algebra operations
- **gram**: Gram matrix computation
- **merge**: Matrix merging
- **slice**: Matrix slicing utilities
- **reorder**: Reordering operations
- **sampling**: Sampling strategies

## Common Patterns

### In-Place Operations

Many kernels support in-place operations:

```cpp
#include "scl/kernel/normalize.hpp"

// In-place normalization
kernel::normalize::normalize_rows_inplace(matrix, target_sum);

// In-place log transform
kernel::log1p::log1p_inplace(matrix);
```

### Output Buffers

Operations that write to output buffers:

```cpp
#include "scl/kernel/normalize.hpp"

// Allocate output buffer
auto output = memory::aligned_alloc<Real>(n);
Array<Real> output_view = {output.get(), n};

// Compute row sums
kernel::normalize::compute_row_sums(matrix, output_view);
```

### Sparse Matrix Operations

Most kernels work with sparse matrices:

```cpp
#include "scl/kernel/neighbors.hpp"

// K-nearest neighbors on sparse matrix
CSR neighbors = kernel::neighbors::knn_sparse(
    matrix, k, metric
);
```

## Performance Characteristics

### Automatic Parallelization

Kernels automatically parallelize when data size exceeds thresholds:

```cpp
// Parallelized automatically for large datasets
kernel::normalize::normalize_rows_inplace(large_matrix, target_sum);
```

### SIMD Optimization

Kernels use SIMD for vectorized operations:

```cpp
// SIMD-accelerated sum computation
Real sum = kernel::normalize::compute_row_sum(row_data);
```

### Memory Efficiency

Kernels are designed for memory efficiency:

```cpp
// In-place operations avoid extra allocations
kernel::log1p::log1p_inplace(matrix);  // No extra memory
```

## Usage Examples

### Normalization

```cpp
#include "scl/kernel/normalize.hpp"

// Total count normalization
kernel::normalize::normalize_total_inplace(matrix, 1e4);

// L2 normalization
kernel::normalize::normalize_rows_inplace(matrix, NormMode::L2);

// Compute row sums
Array<Real> row_sums = {sums_ptr, matrix.rows()};
kernel::normalize::compute_row_sums(matrix, row_sums);
```

### Neighbor Search

```cpp
#include "scl/kernel/neighbors.hpp"

// K-nearest neighbors
CSR knn = kernel::neighbors::knn_sparse(
    matrix, k=15, metric=DistanceMetric::Euclidean
);

// Batch-balanced KNN
CSR bbknn_result = kernel::bbknn::bbknn(
    matrices, batch_labels, k=15
);
```

### Statistical Tests

```cpp
#include "scl/kernel/stat/ttest.hpp"

// T-test
Array<Real> group1 = {data1, n1};
Array<Real> group2 = {data2, n2};
Real t_stat = kernel::stat::ttest::t_test(group1, group2);
```

### Spatial Analysis

```cpp
#include "scl/kernel/spatial.hpp"

// Spatial neighbors
Array<Real> coordinates = {coords_ptr, n * 2};  // x, y coordinates
CSR spatial_neighbors = kernel::spatial::spatial_neighbors(
    coordinates, n, radius
);

// Moran's I
Real morans_i = kernel::spatial::morans_i(
    matrix, spatial_neighbors, gene_idx
);
```

## Configuration

### Kernel-Specific Configuration

Many kernels have configuration namespaces:

```cpp
namespace scl::kernel::normalize::config {
    constexpr Size PREFETCH_DISTANCE = 64;
    constexpr Size PARALLEL_THRESHOLD = 256;
}
```

### Adjusting Thresholds

```cpp
// Modify parallel threshold for specific kernel
namespace scl::kernel::normalize::config {
    constexpr Size PARALLEL_THRESHOLD = 512;  // Custom threshold
}
```

## Best Practices

### 1. Use In-Place Operations When Possible

```cpp
// Good: In-place, no extra memory
kernel::normalize::normalize_rows_inplace(matrix, target_sum);

// Avoid: Extra allocation
auto normalized = kernel::normalize::normalize_rows(matrix, target_sum);
```

### 2. Pre-allocate Output Buffers

```cpp
// Good: Pre-allocated buffer
auto output = memory::aligned_alloc<Real>(n);
Array<Real> output_view = {output.get(), n};
kernel::normalize::compute_row_sums(matrix, output_view);

// Avoid: Kernel allocates internally
auto output = kernel::normalize::compute_row_sums(matrix);  // May allocate
```

### 3. Choose Appropriate Sparse Format

```cpp
// Row-based operations → CSR
CSR matrix = CSR::create(rows, cols, nnz);
kernel::normalize::normalize_rows_inplace(matrix, target_sum);

// Column-based operations → CSC
CSC matrix = CSC::create(rows, cols, nnz);
kernel::normalize::normalize_cols_inplace(matrix, target_sum);
```

### 4. Batch Operations

```cpp
// Process multiple matrices
std::vector<CSR> matrices = {...};
for (auto& matrix : matrices) {
    kernel::normalize::normalize_rows_inplace(matrix, target_sum);
}
```

## Related Documentation

- [Normalization](./normalize.md) - Normalization operations
- [Neighbors](./neighbors.md) - Neighbor search
- [Spatial Analysis](./spatial.md) - Spatial operations
- [Statistical Tests](./statistics.md) - Statistical analysis

