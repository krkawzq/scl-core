# Sampling

Sampling and downsampling kernels for cell selection and data reduction.

## Overview

The `sampling` module provides diverse sampling strategies:

- **Geometric sketching**: Preserve rare populations with uniform manifold coverage
- **Density-preserving**: Maintain local density distribution
- **Landmark selection**: KMeans++-style diverse sampling
- **Representative cells**: Select cells closest to cluster centroids
- **Balanced/Stratified**: Equal representation across groups/strata
- **Uniform/Reservoir**: Simple random sampling

All operations are:
- Memory-efficient
- Reproducible with seeds
- Optimized for large datasets

## Core Functions

### geometric_sketching

Sample cells using geometric sketching to preserve rare populations.

```cpp
#include "scl/kernel/sampling.hpp"

Sparse<Real, true> data = /* expression matrix */;
Index* selected = new Index[target_size];
Size n_selected;

scl::kernel::sampling::geometric_sketching(
    data, target_size, selected, n_selected, 42
);
```

**Parameters:**
- `data` [in] - Expression matrix (cells x genes, CSR)
- `target_size` [in] - Desired number of cells to select
- `selected_indices` [out] - Indices of selected cells
- `n_selected` [out] - Actual number of cells selected
- `seed` [in] - Random seed for reproducibility (default: 42)

**Preconditions:**
- `selected_indices` has capacity >= min(target_size, data.rows())
- `target_size > 0`

**Postconditions:**
- `n_selected <= target_size`
- `selected_indices[0..n_selected)` contains selected cell indices
- Cells are sampled uniformly from geometric grid buckets

**Complexity:**
- Time: O(n * d + n log n) where n = cells, d = features
- Space: O(n + d) auxiliary

**Thread Safety:** Unsafe - sequential implementation

### density_preserving

Sample cells while preserving local density distribution.

```cpp
Sparse<Index, true> neighbors = /* KNN graph */;
scl::kernel::sampling::density_preserving(
    data, neighbors, target_size, selected, n_selected
);
```

**Parameters:**
- `data` [in] - Expression matrix
- `neighbors` [in] - KNN graph (CSR)
- `target_size` [in] - Desired number of cells
- `selected_indices` [out] - Indices of selected cells
- `n_selected` [out] - Actual number selected

**Preconditions:**
- `data.rows() == neighbors.rows()`
- `selected_indices` has capacity >= min(target_size, data.rows())

**Postconditions:**
- Cells from sparse regions are more likely to be selected
- Local density distribution is preserved in sample

**Complexity:**
- Time: O(n)
- Space: O(n) auxiliary

**Thread Safety:** Unsafe - sequential implementation

### landmark_selection

Select diverse landmark cells using KMeans++ initialization.

```cpp
scl::kernel::sampling::landmark_selection(
    data, n_landmarks, landmark_indices, n_selected, 42
);
```

**Parameters:**
- `data` [in] - Expression matrix
- `n_landmarks` [in] - Number of landmarks to select
- `landmark_indices` [out] - Indices of selected landmarks
- `n_selected` [out] - Actual number selected
- `seed` [in] - Random seed (default: 42)

**Preconditions:**
- `landmark_indices` has capacity >= min(n_landmarks, data.rows())

**Postconditions:**
- `n_selected = min(n_landmarks, data.rows())`
- Landmarks are maximally spread in expression space

**Complexity:**
- Time: O(n_landmarks * n * d) for sparse distance computation
- Space: O(n) auxiliary

**Thread Safety:** Unsafe - sequential KMeans++

### representative_cells

Select representative cells from each cluster.

```cpp
Array<const Index> cluster_labels = /* cluster assignments */;
Index* representatives = new Index[max_representatives];
Size n_selected;

scl::kernel::sampling::representative_cells(
    data, cluster_labels, per_cluster, representatives, n_selected, 42
);
```

**Parameters:**
- `data` [in] - Expression matrix
- `cluster_labels` [in] - Cluster assignment for each cell
- `per_cluster` [in] - Number of representatives per cluster
- `representatives` [out] - Indices of representative cells
- `n_selected` [out] - Total representatives selected
- `seed` [in] - Random seed (default: 42)

**Preconditions:**
- `data.rows() == cluster_labels.len`
- `representatives` has sufficient capacity

**Postconditions:**
- `n_selected = sum(min(per_cluster, cluster_size))` over clusters
- Representatives are closest cells to each cluster centroid

**Complexity:**
- Time: O(n * d + n_clusters * cluster_size * per_cluster)
- Space: O(n + d * n_clusters) auxiliary

**Thread Safety:** Unsafe - sequential implementation

## Balanced and Stratified Sampling

### balanced_sampling

Sample equal numbers from each group/label category.

```cpp
Array<const Index> labels = /* group labels */;
Index* selected = new Index[target_size];
Size n_selected;

scl::kernel::sampling::balanced_sampling(
    labels, target_size, selected, n_selected, 42
);
```

**Parameters:**
- `labels` [in] - Group labels for each element
- `target_size` [in] - Total desired sample size
- `selected_indices` [out] - Indices of selected elements
- `n_selected` [out] - Actual number selected
- `seed` [in] - Random seed (default: 42)

**Preconditions:**
- `selected_indices` has capacity >= target_size
- Labels are non-negative integers

**Postconditions:**
- Each non-empty group contributes roughly target_size / n_groups samples
- Remainder distributed to first groups

**Complexity:**
- Time: O(n)
- Space: O(n) auxiliary

**Thread Safety:** Unsafe - sequential implementation

### stratified_sampling

Sample from strata defined by binning a continuous variable.

```cpp
Array<const Real> values = /* continuous values */;
scl::kernel::sampling::stratified_sampling(
    values, n_strata, target_size, selected, n_selected, 42
);
```

**Parameters:**
- `values` [in] - Continuous values to stratify by
- `n_strata` [in] - Number of strata to create
- `target_size` [in] - Total desired sample size
- `selected_indices` [out] - Indices of selected elements
- `n_selected` [out] - Actual number selected
- `seed` [in] - Random seed (default: 42)

**Preconditions:**
- `values.len > 0`
- `n_strata > 0`

**Postconditions:**
- Elements are binned into n_strata equal-width strata
- `balanced_sampling` is applied to strata labels

**Complexity:**
- Time: O(n)
- Space: O(n) auxiliary

**Thread Safety:** Unsafe - sequential implementation

## Simple Sampling

### uniform_sampling

Simple uniform random sampling without replacement.

```cpp
scl::kernel::sampling::uniform_sampling(
    n, target_size, selected_indices, n_selected, 42
);
```

**Parameters:**
- `n` [in] - Total population size
- `target_size` [in] - Desired sample size
- `selected_indices` [out] - Indices of selected elements
- `n_selected` [out] - Actual number selected
- `seed` [in] - Random seed (default: 42)

**Preconditions:**
- `selected_indices` has capacity >= min(target_size, n)

**Postconditions:**
- `n_selected = min(target_size, n)`
- Each element has equal probability of selection

**Complexity:**
- Time: O(n) for initialization, O(target_size) for sampling
- Space: O(n) auxiliary

**Thread Safety:** Unsafe - sequential implementation

### importance_sampling

Sample elements with probability proportional to given weights.

```cpp
Array<const Real> weights = /* sampling weights */;
scl::kernel::sampling::importance_sampling(
    weights, target_size, selected_indices, n_selected, 42
);
```

**Parameters:**
- `weights` [in] - Sampling weights (non-negative)
- `target_size` [in] - Number of samples to draw
- `selected_indices` [out] - Indices of selected elements
- `n_selected` [out] - Actual number selected
- `seed` [in] - Random seed (default: 42)

**Preconditions:**
- `weights.len > 0`
- All weights >= 0

**Postconditions:**
- `n_selected = target_size`
- P(select i) proportional to weights[i]
- Same element may appear multiple times (with replacement)

**Complexity:**
- Time: O(n + target_size * log n)
- Space: O(n) auxiliary

**Thread Safety:** Unsafe - sequential implementation

### reservoir_sampling

Select k items uniformly at random from a stream using reservoir sampling.

```cpp
scl::kernel::sampling::reservoir_sampling(
    stream_size, reservoir_size, reservoir, n_selected, 42
);
```

**Parameters:**
- `stream_size` [in] - Total number of items in stream
- `reservoir_size` [in] - Number of items to select
- `reservoir` [out] - Indices of selected items
- `n_selected` [out] - Actual number selected
- `seed` [in] - Random seed (default: 42)

**Preconditions:**
- `reservoir` has capacity >= min(reservoir_size, stream_size)

**Postconditions:**
- `n_selected = min(reservoir_size, stream_size)`
- Each item has equal probability of being in reservoir

**Complexity:**
- Time: O(stream_size)
- Space: O(reservoir_size)

**Thread Safety:** Unsafe - sequential implementation

## Configuration

```cpp
namespace scl::kernel::sampling::config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size DEFAULT_BINS = 64;
    constexpr Size MAX_ITERATIONS = 1000;
    constexpr Real CONVERGENCE_TOL = Real(1e-6);
    constexpr Size PARALLEL_THRESHOLD = 256;
}
```

## Use Cases

### Preserving Rare Populations

```cpp
// Use geometric sketching to preserve rare cell types
Sparse<Real, true> expression = /* ... */;
Index* selected = new Index[10000];
Size n_selected;

scl::kernel::sampling::geometric_sketching(
    expression, 10000, selected, n_selected, 42
);
// Selected cells have uniform coverage of expression space
```

### Cluster Representatives

```cpp
// Select representative cells from each cluster
Array<const Index> clusters = /* cluster labels */;
Index* reps = new Index[n_clusters * 5];
Size n_reps;

scl::kernel::sampling::representative_cells(
    expression, clusters, 5, reps, n_reps, 42
);
// 5 representatives per cluster, closest to centroids
```

### Balanced Sampling

```cpp
// Sample equal numbers from each batch
Array<const Index> batches = /* batch labels */;
Index* selected = new Index[1000];
Size n_selected;

scl::kernel::sampling::balanced_sampling(
    batches, 1000, selected, n_selected, 42
);
// Each batch contributes roughly equal number of cells
```

### Stratified by Expression

```cpp
// Stratify by total UMI counts
Array<Real> total_counts = /* compute row sums */;
Index* selected = new Index[5000];
Size n_selected;

scl::kernel::sampling::stratified_sampling(
    total_counts, 10, 5000, selected, n_selected, 42
);
// Sample from 10 UMI count strata
```

## Performance

- **Memory efficient**: Minimal allocations for large datasets
- **Reproducible**: Deterministic with fixed seeds
- **Fast RNG**: Xoshiro128+ for high-quality randomness
- **Scalable**: Handles millions of cells efficiently

---

::: tip Method Selection
- **Geometric sketching**: Best for preserving rare populations
- **Density-preserving**: Best for maintaining local structure
- **Landmark selection**: Best for diverse coverage
- **Representative cells**: Best for cluster summarization
- **Balanced/Stratified**: Best for equal representation
- **Uniform**: Simplest, fastest for random sampling
:::

