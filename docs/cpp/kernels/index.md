# Kernels

The `scl/kernel/` directory contains 400+ computational operators organized by functionality.

## Overview

Kernels provide:

- **Sparse Tools** - Matrix conversion, validation, cleanup
- **Normalization** - Row/column normalization, scaling
- **Statistics** - Statistical tests, metrics
- **Neighbors** - KNN, batch-balanced KNN
- **Clustering** - Leiden, Louvain community detection
- **Spatial** - Spatial analysis, hotspot detection
- **Enrichment** - Gene set enrichment analysis

## Categories

### Sparse Tools

Matrix infrastructure and utilities:

```cpp
#include "scl/kernel/sparse.hpp"

// Convert to contiguous CSR/CSC
auto arrays = scl::kernel::sparse::to_contiguous_arrays(matrix);

// Convert to COO format
auto coo = scl::kernel::sparse::to_coo_arrays(matrix);

// Validate matrix structure
auto result = scl::kernel::sparse::validate(matrix);

// Get memory info
auto info = scl::kernel::sparse::memory_info(matrix);

// Eliminate zeros
scl::kernel::sparse::eliminate_zeros(matrix, tolerance);

// Prune small values
scl::kernel::sparse::prune(matrix, threshold, keep_structure);
```

### Normalization

Row/column normalization and scaling:

```cpp
#include "scl/kernel/normalize.hpp"

// Compute row norms
scl::kernel::normalize::row_norms(matrix, NormMode::L2, output);

// Normalize rows in-place
scl::kernel::normalize::normalize_rows_inplace(matrix, NormMode::L2);

// Scale matrix
scl::kernel::normalize::scale(matrix, factor);
```

### Statistics

Statistical tests and metrics:

```cpp
#include "scl/kernel/ttest.hpp"
#include "scl/kernel/mwu.hpp"

// T-test
auto result = scl::kernel::ttest::ttest(group1, group2);

// Mann-Whitney U test
auto result = scl::kernel::mwu::mann_whitney_u(group1, group2);
```

### Neighbors

K-nearest neighbors:

```cpp
#include "scl/kernel/neighbors.hpp"

// Compute KNN
scl::kernel::neighbors::knn(data, k, indices, distances);

// Batch-balanced KNN
scl::kernel::neighbors::bbknn(data, batch_labels, k, indices, distances);
```

### Clustering

Community detection:

```cpp
#include "scl/kernel/leiden.hpp"
#include "scl/kernel/louvain.hpp"

// Leiden clustering
auto labels = scl::kernel::leiden::leiden(graph, resolution);

// Louvain clustering
auto labels = scl::kernel::louvain::louvain(graph, resolution);
```

## Design Patterns

### Functional API

Pure functions with no side effects:

```cpp
// Pure function - returns result
auto result = compute_something(input);

// In-place modification - clearly named
modify_something_inplace(data);
```

### Template-Based Polymorphism

Works with any compatible type:

```cpp
template <CSRLike MatrixT>
void process_matrix(const MatrixT& matrix) {
    // Works with any CSR-like type
}
```

### Explicit Parallelization

```cpp
// Parallel by default for large inputs
parallel_for(Size(0), n, [&](size_t i) {
    process(data[i]);
});
```

## Performance

### SIMD Optimization

Hot paths use SIMD:

```cpp
namespace s = scl::simd;
const s::Tag d;

for (size_t i = 0; i < n; i += s::Lanes(d)) {
    auto v = s::Load(d, data + i);
    // SIMD operations
}
```

### Minimal Allocations

Use workspaces instead of allocating:

```cpp
// Pre-allocate workspace
WorkspacePool<Real> pool(num_threads, workspace_size);

// Reuse in parallel loop
parallel_for(Size(0), n, [&](size_t i, size_t thread_rank) {
    Real* workspace = pool.get(thread_rank);
    // Use workspace
});
```

## Next Steps

Explore specific kernel categories:

- [Sparse Tools](/cpp/kernels/sparse-tools) - Matrix utilities
- [Normalization](/cpp/kernels/normalization) - Normalization and scaling
- [Statistics](/cpp/kernels/statistics) - Statistical tests
- [Neighbors](/cpp/kernels/neighbors) - KNN algorithms
- [Clustering](/cpp/kernels/clustering) - Community detection

---

::: tip High Performance
All kernels are optimized for performance with SIMD, parallelization, and minimal allocations.
:::

