---
title: Neighbor Search
description: K-nearest neighbors computation for sparse matrices
---

# Neighbor Search

The `neighbors` kernel provides efficient K-nearest neighbors (KNN) search for sparse matrices with adaptive algorithms optimized for different sparsity patterns.

## Overview

Neighbor search is fundamental for:
- Building neighborhood graphs
- Computing similarity metrics
- Dimensionality reduction (UMAP, t-SNE)
- Clustering algorithms

The implementation uses adaptive algorithms that automatically select the best strategy based on sparsity patterns.

## Functions

### Sparse Dot Product

#### `sparse_dot_adaptive`

Compute dot product between two sparse vectors with adaptive algorithm selection.

```cpp
template <typename T>
T sparse_dot_adaptive(
    const Index* idx1, const T* val1, Size n1,
    const Index* idx2, const T* val2, Size n2
);
```

**Parameters**:
- `idx1`, `val1`, `n1` [in]: First sparse vector (indices, values, length)
- `idx2`, `val2`, `n2` [in]: Second sparse vector (indices, values, length)

**Returns**: Dot product of the two vectors

**Algorithm Selection**:
- **Linear merge**: For similar-length vectors
- **Binary search**: For vectors with large length ratio
- **Galloping search**: For very large length ratios

**Example**:
```cpp
#include "scl/kernel/neighbors.hpp"

// Get two rows from sparse matrix
auto row1_indices = matrix.row_indices(0);
auto row1_values = matrix.row_values(0);
auto row2_indices = matrix.row_indices(1);
auto row2_values = matrix.row_values(1);

// Compute dot product
Real similarity = kernel::neighbors::detail::sparse_dot_adaptive(
    row1_indices.ptr, row1_values.ptr, row1_indices.size,
    row2_indices.ptr, row2_values.ptr, row2_indices.size
);
```

### Distance Computation

#### `euclidean_distance_sparse`

Compute Euclidean distance between two sparse vectors.

```cpp
template <typename T>
Real euclidean_distance_sparse(
    const Index* idx1, const T* val1, Size n1,
    const Index* idx2, const T* val2, Size n2
);
```

**Mathematical Operation**: `√(Σ(xᵢ - yᵢ)²)`

**Example**:
```cpp
Real dist = kernel::neighbors::euclidean_distance_sparse(
    row1_indices.ptr, row1_values.ptr, row1_indices.size,
    row2_indices.ptr, row2_values.ptr, row2_indices.size
);
```

#### `cosine_distance_sparse`

Compute cosine distance between two sparse vectors.

```cpp
template <typename T>
Real cosine_distance_sparse(
    const Index* idx1, const T* val1, Size n1,
    const Index* idx2, const T* val2, Size n2
);
```

**Mathematical Operation**: `1 - (x·y) / (||x|| ||y||)`

## Adaptive Algorithms

### Linear Merge

For vectors with similar lengths, use linear merge:

```cpp
// O(n1 + n2) time
// Best for: Similar-length vectors
T dot_linear(
    const Index* idx1, const T* val1, Size n1,
    const Index* idx2, const T* val2, Size n2
);
```

**Optimizations**:
- 8-way skip for non-overlapping ranges
- 4-way skip optimization
- Prefetching for memory access

### Binary Search

For vectors with large length ratio, use binary search:

```cpp
// O(n_small * log(n_large)) time
// Best for: n_small << n_large
T dot_binary(
    const Index* idx_small, const T* val_small, Size n_small,
    const Index* idx_large, const T* val_large, Size n_large
);
```

**Optimizations**:
- Search space reduction after each match
- Prefetching for small vector access

### Galloping Search

For very large length ratios, use exponential search (galloping):

```cpp
// O(n_small * log(n_large)) time with better constants
// Best for: n_small << n_large (very sparse)
T dot_gallop(
    const Index* idx_small, const T* val_small, Size n_small,
    const Index* idx_large, const T* val_large, Size n_large
);
```

**Optimizations**:
- Exponential search (galloping) before binary search
- Reduces number of comparisons

## Configuration

```cpp
namespace scl::kernel::neighbors::config {
    constexpr Size PREFETCH_DISTANCE = 32;
    constexpr Size CHUNK_SIZE = 32;
    constexpr Size RATIO_THRESHOLD = 32;      // Switch to binary search
    constexpr Size GALLOP_THRESHOLD = 256;    // Switch to galloping
}
```

## Performance Considerations

### Algorithm Selection

The adaptive algorithm automatically selects the best strategy:

```cpp
// Automatic selection based on length ratio
if (n1 > n2) {
    // Ensure n1 <= n2
    swap(idx1, idx2);
    swap(val1, val2);
    swap(n1, n2);
}

if (n2 / n1 > RATIO_THRESHOLD) {
    if (n2 > GALLOP_THRESHOLD) {
        return dot_gallop(...);  // Very sparse
    } else {
        return dot_binary(...);   // Moderately sparse
    }
} else {
    return dot_linear(...);       // Similar density
}
```

### Early Exit Optimization

Disjoint range check for O(1) early exit:

```cpp
// O(1) check: Are ranges disjoint?
if (idx1[n1-1] < idx2[0] || idx2[n2-1] < idx1[0]) {
    return T(0);  // No overlap, dot product is zero
}
```

### Prefetching

Aggressive prefetching for indirect memory access:

```cpp
// Prefetch ahead for both vectors
if (i + PREFETCH_DISTANCE < n1) {
    SCL_PREFETCH_READ(&idx1[i + PREFETCH_DISTANCE], 0);
    SCL_PREFETCH_READ(&val1[i + PREFETCH_DISTANCE], 0);
}
```

## Common Patterns

### Building KNN Graph

```cpp
CSR build_knn_graph(const CSR& matrix, Index k) {
    Index n = matrix.rows();
    
    // Allocate output
    auto knn = CSR::create(n, n, n * k);
    
    // For each cell, find k nearest neighbors
    for (Index i = 0; i < n; ++i) {
        // Compute distances to all other cells
        std::vector<std::pair<Real, Index>> distances;
        for (Index j = 0; j < n; ++j) {
            if (i == j) continue;
            
            Real dist = compute_distance(matrix, i, j);
            distances.push_back({dist, j});
        }
        
        // Select k nearest
        std::partial_sort(
            distances.begin(),
            distances.begin() + k,
            distances.end()
        );
        
        // Add to KNN graph
        for (Index ki = 0; ki < k; ++ki) {
            knn.set(i, distances[ki].second, distances[ki].first);
        }
    }
    
    return knn;
}
```

### Batch Processing

```cpp
// Process neighbors in batches
constexpr Size BATCH_SIZE = 256;
for (Index batch_start = 0; batch_start < n; batch_start += BATCH_SIZE) {
    Index batch_end = std::min(batch_start + BATCH_SIZE, n);
    
    // Process batch
    for (Index i = batch_start; i < batch_end; ++i) {
        find_neighbors(matrix, i, k);
    }
}
```

## Related Documentation

- [Kernels Overview](./overview.md) - General kernel usage
- [Spatial Analysis](./spatial.md) - Spatial neighbor search
- [Sparse Matrices](../core/sparse.md) - Sparse matrix operations
