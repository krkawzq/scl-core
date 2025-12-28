# neighbors.hpp

> scl/kernel/neighbors.hpp · K-nearest neighbors computation using Euclidean distance

## Overview

This file provides efficient K-nearest neighbors (KNN) computation for sparse matrices using Euclidean distance. The implementation uses various optimizations including Cauchy-Schwarz pruning, adaptive sparse dot product strategies, and SIMD-optimized operations.

Key features:
- Euclidean distance-based KNN
- Pre-computed squared norms for efficiency
- Cauchy-Schwarz inequality for pruning
- Adaptive sparse dot product (linear merge, binary search, galloping)
- Max-heap for maintaining top-k neighbors
- Thread-safe parallelization over samples

**Header**: `#include "scl/kernel/neighbors.hpp"`

---

## Main APIs

### knn

Find K nearest neighbors for each row in a sparse matrix using Euclidean distance.

::: source_code file="scl/kernel/neighbors.hpp" symbol="knn" collapsed
:::

**Algorithm Description**

For each sample i in parallel:

1. Maintain max-heap of size k for nearest neighbors
2. For each candidate j != i:
   a. **Cauchy-Schwarz pruning**: Skip if `|norm_i - norm_j| >= current_max_distance`
     - This uses the lower bound from Cauchy-Schwarz: `|a-b| <= ||a-b||`
   b. **Sparse dot product**: Compute dot product using adaptive strategy:
     - **Linear merge**: For similar-size vectors (ratio < 32)
     - **Binary search**: For ratio >= 32 (larger vector much longer)
     - **Galloping**: For ratio >= 256 (extreme size difference)
   c. **Distance computation**: `distance = sqrt(norm_i + norm_j - 2*dot)`
     - Negative values from numerical error clamped to 0
   d. **Heap update**: If distance < current max, remove max and insert new neighbor
3. Sort final heap to get ascending order by distance

Sparse dot product optimizations:
- 8-way/4-way skip for non-overlapping index ranges
- Prefetch in merge loop for cache efficiency
- Early exit on disjoint ranges (O(1) check)

**Edge Cases**

- **Fewer than k neighbors exist**: Remaining slots filled with index=-1 and distance=infinity
- **Self-exclusion**: Sample i is excluded from its own neighbor list
- **All-zero samples**: Distance computed correctly (both norms = 0, distance = 0)
- **Identical samples**: Distance = 0, will be in neighbor list if allowed
- **Negative distance from numerical error**: Clamped to 0 before heap insertion
- **Empty matrix**: Returns immediately with all indices = -1

**Data Guarantees (Preconditions)**

- `norms_sq.len >= matrix.primary_dim()` (pre-computed norms)
- `norms_sq` contains valid squared norms from `compute_norms()`
- `out_indices.len >= matrix.primary_dim() * k`
- `out_distances.len >= matrix.primary_dim() * k`
- `k > 0`
- Matrix must be valid CSR/CSC format

**Complexity Analysis**

- **Time**: O(n^2 * avg_nnz) worst case, often much better with pruning
  - Best case (heavy pruning): O(n * k * log k)
  - With Cauchy-Schwarz pruning, many candidates skipped early
- **Space**: O(k) per thread for heap storage

**Example**

```cpp
#include "scl/kernel/neighbors.hpp"
#include "scl/core/sparse.hpp"

// Create sparse matrix (n_samples x n_features)
Sparse<Real, true> matrix(n_samples, n_features);
// Fill with data...

// Pre-compute squared norms
Array<Real> norms_sq(n_samples);
scl::kernel::neighbors::compute_norms(matrix, norms_sq);

// Pre-allocate output
Size k = 10;  // Number of neighbors
Array<Index> indices(n_samples * k);
Array<Real> distances(n_samples * k);

// Compute KNN
scl::kernel::neighbors::knn(
    matrix,
    norms_sq,
    k,
    indices,
    distances
);

// Results:
// For sample i:
// - indices[i*k : i*k+k] contains indices of k nearest neighbors
// - distances[i*k : i*k+k] contains Euclidean distances
// - Neighbors are sorted by distance (ascending)
// - Self (i) is excluded
```

---

## Utility Functions

### compute_norms

Compute squared L2 norms for each row/column of a sparse matrix. Must be called before `knn()`.

::: source_code file="scl/kernel/neighbors.hpp" symbol="compute_norms" collapsed
:::

**Complexity**

- Time: O(nnz) with SIMD-optimized sum_squared
- Space: O(1) auxiliary

**Example**

```cpp
Sparse<Real, true> matrix(n_samples, n_features);
Array<Real> norms_sq(n_samples);

// Compute norms before KNN
scl::kernel::neighbors::compute_norms(matrix, norms_sq);

// Now use norms_sq in knn()
scl::kernel::neighbors::knn(matrix, norms_sq, k, indices, distances);
```

---

## Numerical Notes

- **Distance formula**: `sqrt(norm_i + norm_j - 2*dot)` where `norm_i = ||x_i||^2`
- **Numerical stability**: Negative values from `norm_i + norm_j - 2*dot` clamped to 0 (should be non-negative by Cauchy-Schwarz)
- **Cauchy-Schwarz bound**: `|norm_i - norm_j| <= ||x_i - x_j||` used for pruning
- **Sparse dot product**: Adaptive strategy based on size ratio:
  - Linear merge: O(nnz1 + nnz2) when sizes similar
  - Binary search: O(nnz1 * log(nnz2)) when ratio >= 32
  - Galloping: O(nnz1 * log(nnz2/nnz1)) when ratio >= 256
- **Heap operations**: Max-heap maintained with O(log k) insert/remove

## See Also

- [BBKNN](/cpp/kernels/bbknn) - Batch-balanced KNN for batch integration
- [Normalization](/cpp/kernels/normalization) - Normalize matrices before computing distances
