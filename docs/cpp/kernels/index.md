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

### Softmax

Softmax normalization with temperature scaling:

```cpp
#include "scl/kernel/softmax.hpp"

// Standard softmax
scl::kernel::softmax::softmax_inplace(values, len);

// With temperature
scl::kernel::softmax::softmax_inplace(values, len, 0.5);

// Log-softmax
scl::kernel::softmax::log_softmax_inplace(values, len);

// Sparse matrix
scl::kernel::softmax::softmax_inplace(matrix);
```

### Statistics

Statistical tests and metrics:

```cpp
#include "scl/kernel/ttest.hpp"
#include "scl/kernel/mwu.hpp"

// T-test
scl::kernel::ttest::ttest(matrix, group_ids, t_stats, p_values, log2_fc);

// Mann-Whitney U test
scl::kernel::mwu::mwu_test(matrix, group_ids, u_stats, p_values, log2_fc);
```

### Group Aggregation

Per-group statistics:

```cpp
#include "scl/kernel/group.hpp"

// Compute group means and variances
scl::kernel::group::group_stats(
    matrix, group_ids, n_groups, group_sizes, means, vars
);
```

### Matrix Operations

Merging and slicing:

```cpp
#include "scl/kernel/merge.hpp"
#include "scl/kernel/slice.hpp"

// Vertical stack
auto result = scl::kernel::merge::vstack(matrix1, matrix2);

// Horizontal stack
auto result = scl::kernel::merge::hstack(matrix1, matrix2);

// Slice rows
auto sliced = scl::kernel::slice::slice_primary(matrix, keep_indices);

// Filter columns
auto filtered = scl::kernel::slice::filter_secondary(matrix, mask);
```

### Neighbors

K-nearest neighbors:

```cpp
#include "scl/kernel/neighbors.hpp"

// Pre-compute norms
scl::kernel::neighbors::compute_norms(matrix, norms_sq);

// Compute KNN
scl::kernel::neighbors::knn(matrix, norms_sq, k, indices, distances);
```

### Batch Balanced KNN

Batch-aware KNN for integrating data across batches:

```cpp
#include "scl/kernel/bbknn.hpp"

// Pre-compute norms (optional)
scl::kernel::bbknn::compute_norms(matrix, norms_sq);

// Compute BBKNN (k neighbors from each batch)
scl::kernel::bbknn::bbknn(
    matrix, batch_labels, n_batches, k, indices, distances, norms_sq
);
```

### MMD

Maximum Mean Discrepancy for distribution comparison:

```cpp
#include "scl/kernel/mmd.hpp"

// Compare two distributions
scl::kernel::mmd::mmd_rbf(mat_x, mat_y, output, gamma);
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
- [Softmax](/cpp/kernels/softmax) - Softmax normalization with temperature scaling
- [Mann-Whitney U](/cpp/kernels/mwu) - Non-parametric statistical test
- [T-test](/cpp/kernels/ttest) - Parametric statistical test
- [Neighbors](/cpp/kernels/neighbors) - KNN algorithms
- [BBKNN](/cpp/kernels/bbknn) - Batch Balanced KNN for batch integration
- [MMD](/cpp/kernels/mmd) - Maximum Mean Discrepancy for distribution comparison
- [Merge](/cpp/kernels/merge) - Matrix merging operations
- [Slice](/cpp/kernels/slice) - Matrix slicing operations
- [Group](/cpp/kernels/group) - Group aggregation statistics
- [Statistics](/cpp/kernels/statistics) - Statistical tests
- [Clustering](/cpp/kernels/clustering) - Community detection

---

::: tip High Performance
All kernels are optimized for performance with SIMD, parallelization, and minimal allocations.
:::

