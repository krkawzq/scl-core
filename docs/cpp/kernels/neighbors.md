# Neighbors

K-nearest neighbors (KNN) computation with sparse matrix optimization and Cauchy-Schwarz pruning.

## Overview

Neighbors operations provide:

- **KNN search** - Find K nearest neighbors for each sample
- **Euclidean distance** - L2 distance computation
- **Sparse optimization** - Efficient sparse dot product
- **Pruning** - Cauchy-Schwarz lower bound for early termination

## Basic Usage

### knn

Find K nearest neighbors for each row in a sparse matrix.

```cpp
#include "scl/kernel/neighbors.hpp"
#include "scl/core/sparse.hpp"

Sparse<Real, true> matrix = /* ... */;  // n_samples x n_features

// Pre-compute squared norms
Array<Real> norms_sq(matrix.primary_dim());
scl::kernel::neighbors::compute_norms(matrix, norms_sq);

// Find K nearest neighbors
Size k = 15;
Array<Index> indices(matrix.primary_dim() * k);
Array<Real> distances(matrix.primary_dim() * k);

scl::kernel::neighbors::knn(
    matrix,
    norms_sq,
    k,
    indices,
    distances
);
```

**Parameters:**
- `matrix` [in] - Sparse matrix (n_samples x n_features)
- `norms_sq` [in] - Pre-computed squared norms from `compute_norms()`
- `k` [in] - Number of neighbors to find
- `out_indices` [out] - Neighbor indices, shape (n_samples * k)
- `out_distances` [out] - Neighbor distances, shape (n_samples * k)

**Preconditions:**
- `norms_sq.len >= matrix.primary_dim()`
- `norms_sq` contains valid squared norms from `compute_norms()`
- `out_indices.len >= matrix.primary_dim() * k`
- `out_distances.len >= matrix.primary_dim() * k`
- `k > 0`

**Postconditions:**
- For each sample i:
  - `out_indices[i*k : i*k+k]` contains indices of k nearest neighbors
  - `out_distances[i*k : i*k+k]` contains Euclidean distances to neighbors
  - Neighbors are sorted by distance (ascending)
  - Self (i) is excluded from neighbors
- If fewer than k neighbors exist: remaining slots filled with index=-1 and distance=infinity

**Algorithm:**
For each sample i in parallel:
1. Maintain max-heap of size k for nearest neighbors
2. For each candidate j != i:
   a. Cauchy-Schwarz pruning: skip if |norm_i - norm_j| >= current_max
   b. Compute sparse dot product using adaptive strategy:
      - Linear merge: for similar-size vectors
      - Binary search: for ratio >= 32
      - Galloping: for ratio >= 256
   c. Compute distance: sqrt(norm_i + norm_j - 2*dot)
   d. Update heap if distance < current max
3. Sort final heap to get ascending order

**Sparse dot optimizations:**
- 8-way/4-way skip for non-overlapping index ranges
- Prefetch in merge loop
- Early exit on disjoint ranges (O(1) check)

**Complexity:**
- Time: O(n^2 * avg_nnz) worst case, often much better with pruning
- Space: O(k) per thread for heap storage

**Thread Safety:**
Safe - parallelized over samples with thread-local workspace

**Numerical Notes:**
- Distance computed as sqrt(norm_i + norm_j - 2*dot)
- Negative values from numerical error clamped to 0
- Cauchy-Schwarz lower bound enables significant pruning

## Helper Functions

### compute_norms

Compute squared L2 norms for each row/column of a sparse matrix.

```cpp
Array<Real> norms_sq(matrix.primary_dim());
scl::kernel::neighbors::compute_norms(matrix, norms_sq);
```

**Parameters:**
- `matrix` [in] - Sparse matrix (CSR or CSC)
- `norms_sq` [out] - Pre-allocated buffer for squared norms

**Postconditions:**
- `norms_sq[i] = sum of squared values in row/column i`

**Algorithm:**
For each row in parallel:
- Use SIMD-optimized `scl::vectorize::sum_squared`

**Complexity:**
- Time: O(nnz)
- Space: O(1) auxiliary

**Thread Safety:**
Safe - parallelized over rows with no shared mutable state

**Throws:**
`SCL_CHECK_DIM` - if norms_sq size is insufficient

## Use Cases

### Building KNN Graph

Construct a KNN graph for downstream analysis:

```cpp
Sparse<Real, true> data = /* ... */;  // cells x genes

// Compute norms
Array<Real> norms_sq(data.primary_dim());
scl::kernel::neighbors::compute_norms(data, norms_sq);

// Find neighbors
Size k = 15;
Array<Index> knn_indices(data.primary_dim() * k);
Array<Real> knn_distances(data.primary_dim() * k);

scl::kernel::neighbors::knn(
    data, norms_sq, k, knn_indices, knn_distances
);

// Build adjacency matrix or graph structure
// Use knn_indices and knn_distances to construct graph
```

### UMAP / t-SNE Preprocessing

Prepare KNN graph for dimensionality reduction:

```cpp
// Standard preprocessing
Sparse<Real, true> normalized_data = /* ... */;

Array<Real> norms_sq(normalized_data.primary_dim());
scl::kernel::neighbors::compute_norms(normalized_data, norms_sq);

Size k = 15;
Array<Index> knn_indices(normalized_data.primary_dim() * k);
Array<Real> knn_distances(normalized_data.primary_dim() * k);

scl::kernel::neighbors::knn(
    normalized_data, norms_sq, k, knn_indices, knn_distances
);

// Pass to UMAP/t-SNE with KNN graph
```

### Clustering Preprocessing

Compute KNN for Leiden/Louvain clustering:

```cpp
// Normalize and compute KNN
Sparse<Real, true> data = /* ... */;
scl::kernel::normalize::normalize_rows_inplace(data, NormMode::L2);

Array<Real> norms_sq(data.primary_dim());
scl::kernel::neighbors::compute_norms(data, norms_sq);

Array<Index> knn_indices(data.primary_dim() * 15);
Array<Real> knn_distances(data.primary_dim() * 15);

scl::kernel::neighbors::knn(
    data, norms_sq, 15, knn_indices, knn_distances
);

// Convert to graph and cluster
// Use with scl::kernel::leiden or scl::kernel::louvain
```

## Performance

### Pruning Optimization

Cauchy-Schwarz lower bound enables early termination:
- Skip candidates where |norm_i - norm_j| >= current_max_distance
- Reduces computation by 50-90% in typical cases

### Sparse Dot Product

Adaptive strategy based on vector size ratio:
- Linear merge: O(nnz1 + nnz2) for similar sizes
- Binary search: O(nnz1 * log(nnz2)) for ratio >= 32
- Galloping: O(nnz1 * log(nnz2/nnz1)) for ratio >= 256

### SIMD Optimization

- SIMD-optimized norm computation
- Prefetch in merge loops
- 8-way/4-way skip for non-overlapping ranges

### Parallelization

- Parallelized over samples
- Thread-local heap storage
- No synchronization overhead

## Algorithm Details

### Distance Computation

Euclidean distance: sqrt(norm_i + norm_j - 2*dot)

Where:
- norm_i = ||x_i||^2 (pre-computed)
- norm_j = ||x_j||^2 (pre-computed)
- dot = x_i · x_j (computed via sparse dot product)

### Cauchy-Schwarz Pruning

Lower bound: |norm_i - norm_j| <= ||x_i - x_j||

If |norm_i - norm_j| >= current_max_distance:
- Skip candidate j (cannot be closer than current max)

### Heap Management

Max-heap of size k:
- Maintains k nearest neighbors seen so far
- O(log k) insertion
- Final sort: O(k log k)

## See Also

- [BBKNN](/cpp/kernels/bbknn) - Batch-balanced KNN
- [Spatial](/cpp/kernels/spatial) - Spatial neighbor search

