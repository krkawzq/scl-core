# sampling.hpp

> scl/kernel/sampling.hpp · Sampling and downsampling kernels for cell selection

## Overview

This file provides various cell sampling strategies for selecting representative subsets of cells from large datasets. Different strategies preserve different properties: geometric coverage, density distribution, cluster representation, etc.

Key features:
- Geometric sketching for uniform manifold coverage
- Density-preserving sampling
- Landmark selection (KMeans++ style)
- Representative cell selection from clusters
- Balanced and stratified sampling
- Uniform and importance sampling

**Header**: `#include "scl/kernel/sampling.hpp"`

---

## Main APIs

### geometric_sketching

::: source_code file="scl/kernel/sampling.hpp" symbol="geometric_sketching" collapsed
:::

**Algorithm Description**

Sample cells using geometric sketching to preserve rare populations:

1. **Compute bounds**: Find min/max for each feature dimension
2. **Create grid**: Divide feature space into `DEFAULT_BINS` bins per dimension
3. **Hash assignment**: Assign each cell to a grid bucket via hash function
4. **Sort by bucket**: Sort cells by bucket ID using VQSort
5. **Proportional sampling**: Sample proportionally from each bucket to ensure uniform coverage

Geometric sketching ensures that cells are sampled uniformly across the data manifold, preserving rare cell types that might be missed by random sampling.

**Edge Cases**

- **Empty data**: Returns 0 selected cells
- **target_size >= n_cells**: Returns all cells
- **Very sparse data**: Some buckets may be empty
- **High-dimensional data**: Grid becomes sparse (curse of dimensionality)

**Data Guarantees (Preconditions)**

- Matrix must be valid CSR format (cells x features)
- `selected_indices` has capacity >= min(target_size, data.rows())
- `target_size > 0`

**Complexity Analysis**

- **Time**: O(n * d + n log n) where n = cells, d = features
  - O(n * d) for bounds computation and hashing
  - O(n log n) for sorting by bucket
  - O(n) for sampling
- **Space**: O(n + d) auxiliary for buckets and bounds

**Example**

```cpp
#include "scl/kernel/sampling.hpp"

Sparse<Real, true> data = /* ... */;  // Expression matrix (cells x genes)
Size target_size = 10000;  // Desired sample size
Index* selected_indices = /* allocate target_size */;
Size n_selected;

scl::kernel::sampling::geometric_sketching(
    data,
    target_size,
    selected_indices,
    n_selected,
    42  // seed
);

// selected_indices[0..n_selected) contains selected cell indices
// Cells are sampled uniformly from geometric grid buckets
```

---

### density_preserving

::: source_code file="scl/kernel/sampling.hpp" symbol="density_preserving" collapsed
:::

**Algorithm Description**

Sample cells while preserving local density distribution:

1. **Compute local density**: For each cell, count neighbors in KNN graph
2. **Compute weights**: `weight[i] = 1 / (density[i] + epsilon)` - inverse density
3. **Normalize weights**: Sum weights to 1
4. **Systematic sampling**: Sample with probability proportional to weights

This ensures that cells from sparse regions (rare cell types) are more likely to be selected, preserving the density distribution in the sample.

**Edge Cases**

- **Isolated cells**: Have very high weight (likely to be selected)
- **Dense clusters**: Have lower weight per cell
- **Empty KNN graph**: Falls back to uniform sampling

**Data Guarantees (Preconditions)**

- `data.rows() == neighbors.rows()`
- `selected_indices` has capacity >= min(target_size, data.rows())
- KNN graph must be valid CSR format

**Complexity Analysis**

- **Time**: O(n) - single pass through cells and neighbors
- **Space**: O(n) auxiliary for density and weights

**Example**

```cpp
Sparse<Real, true> data = /* ... */;
Sparse<Index, true> neighbors = /* ... */;  // KNN graph

scl::kernel::sampling::density_preserving(
    data,
    neighbors,
    target_size,
    selected_indices,
    n_selected
);

// Cells from sparse regions are more likely to be selected
```

---

### landmark_selection

::: source_code file="scl/kernel/sampling.hpp" symbol="landmark_selection" collapsed
:::

**Algorithm Description**

Select diverse landmark cells using KMeans++ initialization:

1. **First center**: Select uniformly at random
2. **Subsequent centers**: For each new center:
   - Compute squared distance from each cell to nearest existing center
   - Sample cell with probability proportional to squared distance
   - Add to landmarks
3. **Repeat**: Until n_landmarks selected

KMeans++ ensures landmarks are maximally spread in expression space, providing good coverage for dimensionality reduction or clustering initialization.

**Edge Cases**

- **n_landmarks >= n_cells**: Returns all cells
- **Single landmark**: Returns one random cell
- **Empty data**: Returns 0 landmarks

**Data Guarantees (Preconditions)**

- Matrix must be valid CSR format
- `landmark_indices` has capacity >= min(n_landmarks, data.rows())
- `n_landmarks > 0`

**Complexity Analysis**

- **Time**: O(n_landmarks * n * d) for sparse distance computation
  - For each landmark: O(n * d) to compute distances
- **Space**: O(n) auxiliary for distances

**Example**

```cpp
Size n_landmarks = 1000;
Index* landmark_indices = /* allocate n_landmarks */;
Size n_selected;

scl::kernel::sampling::landmark_selection(
    data,
    n_landmarks,
    landmark_indices,
    n_selected,
    42  // seed
);

// Landmarks are maximally spread in expression space
```

---

### representative_cells

::: source_code file="scl/kernel/sampling.hpp" symbol="representative_cells" collapsed
:::

**Algorithm Description**

Select representative cells from each cluster:

1. **For each cluster**:
   - Compute centroid as mean of all cells in cluster
   - Compute squared distance from each cell to centroid
   - Use partial_sort to find closest `per_cluster` cells
   - Add to representatives list

Representative cells are those closest to cluster centroids, useful for visualization or downstream analysis.

**Edge Cases**

- **Empty clusters**: Skipped
- **Small clusters**: Returns all cells if cluster_size < per_cluster
- **per_cluster = 0**: Returns 0 representatives

**Data Guarantees (Preconditions)**

- `data.rows() == cluster_labels.len`
- `representatives` has sufficient capacity
- Cluster labels must be non-negative integers

**Complexity Analysis**

- **Time**: O(n * d + n_clusters * cluster_size * per_cluster)
  - O(n * d) for centroid computation
  - O(cluster_size * per_cluster) per cluster for sorting
- **Space**: O(n + d * n_clusters) auxiliary

**Example**

```cpp
Array<Index> cluster_labels = /* ... */;  // Cluster assignment per cell
Size per_cluster = 10;  // Representatives per cluster
Index* representatives = /* allocate */;
Size n_selected;

scl::kernel::sampling::representative_cells(
    data,
    cluster_labels,
    per_cluster,
    representatives,
    n_selected,
    42  // seed
);

// Representatives are closest cells to each cluster centroid
```

---

## Utility Functions

### balanced_sampling

Sample equal numbers from each group/label category.

::: source_code file="scl/kernel/sampling.hpp" symbol="balanced_sampling" collapsed
:::

**Complexity**

- Time: O(n)
- Space: O(n) auxiliary

---

### stratified_sampling

Sample from strata defined by binning a continuous variable.

::: source_code file="scl/kernel/sampling.hpp" symbol="stratified_sampling" collapsed
:::

**Complexity**

- Time: O(n)
- Space: O(n) auxiliary

---

### uniform_sampling

Simple uniform random sampling without replacement.

::: source_code file="scl/kernel/sampling.hpp" symbol="uniform_sampling" collapsed
:::

**Complexity**

- Time: O(n) for initialization, O(target_size) for sampling
- Space: O(n) auxiliary

---

### importance_sampling

Sample elements with probability proportional to given weights (with replacement).

::: source_code file="scl/kernel/sampling.hpp" symbol="importance_sampling" collapsed
:::

**Complexity**

- Time: O(n + target_size * log n)
- Space: O(n) auxiliary

---

### reservoir_sampling

Select k items uniformly at random from a stream using reservoir sampling (Algorithm R).

::: source_code file="scl/kernel/sampling.hpp" symbol="reservoir_sampling" collapsed
:::

**Complexity**

- Time: O(stream_size)
- Space: O(reservoir_size)

---

## Notes

**Sampling Strategy Selection**

- **Geometric sketching**: For preserving rare populations and uniform coverage
- **Density-preserving**: For maintaining density distribution
- **Landmark selection**: For diverse, spread-out samples
- **Representative cells**: For cluster-based selection
- **Balanced sampling**: For equal representation across groups
- **Uniform sampling**: For simple random sampling

**Thread Safety**

Most functions are sequential (unsafe for parallel execution) due to:
- Random number generation dependencies
- Sorting operations
- Cumulative computations

**Use Cases**

- **Data reduction**: Reduce dataset size for faster computation
- **Visualization**: Select representative cells for plotting
- **Downstream analysis**: Prepare samples for specific algorithms
- **Quality control**: Sample for manual inspection

## See Also

- [Resample](/cpp/kernels/resample) - Count resampling operations
- [Statistics](/cpp/kernels/statistics) - Statistical analysis
