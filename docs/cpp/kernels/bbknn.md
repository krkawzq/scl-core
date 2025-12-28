# bbknn.hpp

> scl/kernel/bbknn.hpp · Batch Balanced K-Nearest Neighbors for cross-batch integration

## Overview

This file provides batch-balanced KNN search for integrating data across multiple batches. Unlike standard KNN which finds k nearest neighbors globally, BBKNN finds k nearest neighbors from EACH batch, ensuring balanced representation across batches.

Key features:
- Batch-aware neighbor search
- Memory-efficient batch-grouped processing
- Optimized with Cauchy-Schwarz pruning and sparse dot product
- Thread-safe parallelization

**Header**: `#include "scl/kernel/bbknn.hpp"`

---

## Main APIs

### bbknn

::: source_code file="scl/kernel/bbknn.hpp" symbol="bbknn" collapsed
:::

**Algorithm Description**

Batch Balanced KNN finds k nearest neighbors from each batch for every sample:

1. **Batch Grouping**: Group all samples by their batch labels for cache locality
2. **Per-Sample Processing**: For each query sample i:
   - For each batch b:
     - Initialize a max-heap of size k to track k nearest neighbors
     - Iterate over all candidates in batch b
     - For each candidate j:
       - Compute lower bound using Cauchy-Schwarz inequality: `min_dist^2 = norm_i^2 + norm_j^2 - 2*sqrt(norm_i^2 * norm_j^2)`
       - If lower bound >= current max distance in heap, skip candidate (pruning)
       - Otherwise, compute exact Euclidean distance: `dist^2 = norm_i^2 + norm_j^2 - 2*dot(i,j)`
       - Use optimized sparse dot product with 8/4-way skip optimization
       - If distance < current max in heap, insert into heap
     - Extract k nearest neighbors sorted by distance
3. **Output Layout**: For sample i, batch b, neighbor j: `offset = i * (n_batches * k) + b * k + j`

**Edge Cases**

- **Empty matrix**: Returns immediately with all indices set to -1 and distances to infinity
- **Fewer than k neighbors in batch**: Remaining slots filled with index = -1, distance = infinity
- **Zero-norm rows**: Handled correctly (distance = norm_i^2 + norm_j^2 when dot product = 0)
- **Negative batch labels**: Samples with negative batch labels are ignored (not included in any batch)

**Data Guarantees (Preconditions)**

- Matrix must be valid CSR format with sorted indices within rows
- `batch_labels.len == matrix.primary_dim()`
- `batch_labels[i]` must be in range [0, n_batches) or negative (ignored)
- `out_indices.len >= n_samples * n_batches * k`
- `out_distances.len >= n_samples * n_batches * k`
- If `norms_sq` provided: `norms_sq.len >= n_samples`
- Matrix indices must be sorted within each row (CSR requirement)

**Complexity Analysis**

- **Time**: O(n_samples * avg_batch_size * (nnz_per_row + k*log(k)))
  - For each sample: iterate over candidates in each batch
  - Sparse dot product: O(nnz_per_row) average case
  - Heap operations: O(k*log(k)) per batch
- **Space**: O(n_threads * n_batches * k) for thread-local heaps

**Example**

```cpp
#include "scl/kernel/bbknn.hpp"
#include "scl/core/sparse.hpp"

// Create sparse matrix (cells x features)
Sparse<Real, true> expression = /* ... */;  // n_cells x n_features
Array<int32_t> batch_labels(n_cells);
// ... assign batch labels (0, 1, 2, ...) ...

Size n_batches = 3;
Size k = 15;  // Neighbors per batch

// Pre-compute norms for better performance (optional)
Array<Real> norms_sq(n_cells);
scl::kernel::bbknn::compute_norms(expression, norms_sq);

// Allocate output arrays
Array<Index> indices(n_cells * n_batches * k);
Array<Real> distances(n_cells * n_batches * k);

// Compute BBKNN
scl::kernel::bbknn::bbknn(
    expression,
    batch_labels,
    n_batches,
    k,
    indices,
    distances,
    norms_sq  // Optional: can omit to compute norms on-the-fly
);

// Access neighbors for cell i, batch b, neighbor j:
// Index neighbor_idx = indices[i * (n_batches * k) + b * k + j];
// Real dist = distances[i * (n_batches * k) + b * k + j];
```

---

### compute_norms

::: source_code file="scl/kernel/bbknn.hpp" symbol="compute_norms" collapsed
:::

**Algorithm Description**

Precompute squared L2 norms for all rows of a sparse matrix:

1. Parallel processing over rows
2. For each row i: compute `norm_i^2 = sum(matrix[i,:]^2)`
3. Uses SIMD-optimized `scl::vectorize::sum_squared` for efficiency

**Edge Cases**

- **Empty rows**: Return norm_sq = 0
- **Zero matrix**: All norms are zero

**Data Guarantees (Preconditions)**

- Matrix must be valid CSR format
- `norms_sq.len >= matrix.primary_dim()`

**Complexity Analysis**

- **Time**: O(nnz / n_threads) - parallelized over rows
- **Space**: O(1) auxiliary

**Example**

```cpp
Sparse<Real, true> matrix = /* ... */;
Array<Real> norms_sq(matrix.rows());

scl::kernel::bbknn::compute_norms(matrix, norms_sq);

// norms_sq[i] now contains ||matrix[i,:]||^2
```

---

### build_batch_groups

::: source_code file="scl/kernel/bbknn.hpp" symbol="build_batch_groups" collapsed
:::

**Algorithm Description**

Build memory-efficient batch-grouped index structure:

1. **First pass**: Count samples per batch (single pass through batch_labels)
2. **Prefix sum**: Compute offsets for each batch: `offsets[b] = sum(sizes[0..b-1])`
3. **Second pass**: Fill indices array, grouping samples by batch

**Edge Cases**

- **Negative batch labels**: Ignored (not included in any batch)
- **Empty batches**: Batch has size 0, but offset still computed correctly
- **All samples invalid**: Returns empty structure with total_size = 0

**Data Guarantees (Preconditions)**

- `batch_labels[i]` in range [0, n_batches) or negative (ignored)
- `n_batches > 0`

**Complexity Analysis**

- **Time**: O(n_samples) - two passes
- **Space**: O(n_samples + n_batches) for indices and offsets arrays

**Example**

```cpp
Array<int32_t> batch_labels(n_cells);
// ... assign batch labels ...

BatchGroups groups;
scl::kernel::bbknn::build_batch_groups(
    batch_labels,
    n_batches,
    groups
);

// Access samples in batch b:
Size batch_size = groups.batch_size(b);
const Index* batch_indices = groups.batch_data(b);

// Free when done
scl::kernel::bbknn::free_batch_groups(groups);
```

---

## Utility Functions

### free_batch_groups

Free memory allocated by `build_batch_groups`.

::: source_code file="scl/kernel/bbknn.hpp" symbol="free_batch_groups" collapsed
:::

**Complexity**

- Time: O(1)
- Space: O(1)

---

## Notes

**Output Layout**

For sample i, batch b, neighbor j:
```
offset = i * (n_batches * k) + b * k + j
indices[offset] = neighbor index (or -1 if not found)
distances[offset] = Euclidean distance (or infinity if not found)
```

**Optimization Details**

1. **Cauchy-Schwarz Pruning**: Lower bound enables early termination, reducing computation by 30-70%
2. **Sparse Dot Product**: 8/4-way skip optimization for non-overlapping regions
3. **Fixed-Size Heap**: Manual sift operations avoid dynamic allocation overhead
4. **Batch Grouping**: Improves cache locality by processing batch-contiguous samples

**Thread Safety**

All functions are thread-safe and parallelized:
- `bbknn`: Uses thread-local heap storage
- `compute_norms`: Parallel over rows
- `build_batch_groups`: Sequential (single-threaded)

## See Also

- [Neighbors](/cpp/kernels/neighbors) - Standard KNN search
- [Spatial](/cpp/kernels/spatial) - Spatial neighbor search
