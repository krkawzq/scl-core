# subpopulation.hpp

> scl/kernel/subpopulation.hpp · Subpopulation analysis and cluster refinement

## Overview

This file provides functions for subpopulation analysis and cluster refinement in single-cell data. It includes recursive sub-clustering and cluster stability assessment using bootstrap resampling.

**Header**: `#include "scl/kernel/subpopulation.hpp"`

Key features:
- Recursive sub-clustering within existing clusters
- Cluster stability assessment via bootstrap
- Hierarchical cluster refinement

---

## Main APIs

### recursive_subclustering

::: source_code file="scl/kernel/subpopulation.hpp" symbol="recursive_subclustering" collapsed
:::

**Algorithm Description**

Perform recursive sub-clustering within clusters:

1. For each initial cluster:
   - If cluster size >= min_size and depth < max_depth:
     - Apply clustering algorithm (e.g., k-means, Leiden) to subdivide cluster
     - Recursively apply to each subcluster
     - Assign hierarchical labels: `subcluster_labels[i] = parent_cluster * base + subcluster_id`
   - Otherwise: keep original cluster label
2. Builds hierarchical cluster tree up to max_depth levels
3. Uses parallel processing for independent clusters

**Edge Cases**

- **max_depth = 0**: Returns original cluster labels unchanged
- **min_size too large**: No clusters are subdivided
- **Empty clusters**: Skipped in recursion
- **Single cell clusters**: Cannot be subdivided

**Data Guarantees (Preconditions)**

- `subcluster_labels` has capacity >= n_cells
- `cluster_labels` contains valid cluster indices
- Expression matrix must be valid CSR format
- `min_size >= 2` for meaningful subdivision

**Complexity Analysis**

- **Time**: O(max_depth * n_cells * log(n_cells)) - clustering at each level
- **Space**: O(n_cells) auxiliary space

**Example**

```cpp
#include "scl/kernel/subpopulation.hpp"

scl::Sparse<Real, true> expression = /* expression matrix */;
scl::Array<Index> cluster_labels = /* initial clusters */;
scl::Array<Index> subcluster_labels(n_cells);

scl::kernel::subpopulation::recursive_subclustering(
    expression, cluster_labels, n_cells,
    subcluster_labels, 3,  // max_depth
    10                     // min_size
);

// subcluster_labels contains hierarchical subcluster assignments
```

---

### cluster_stability

::: source_code file="scl/kernel/subpopulation.hpp" symbol="cluster_stability" collapsed
:::

**Algorithm Description**

Assess cluster stability using bootstrap resampling:

1. For each bootstrap iteration (parallel):
   - Sample cells with replacement (bootstrap sample)
   - Re-cluster bootstrap sample
   - Compute cluster overlap with original clustering
   - Accumulate stability metrics
2. For each cluster:
   - Stability score = average Jaccard similarity across bootstrap iterations
   - Higher scores indicate more stable clusters
3. Returns stability scores in [0, 1] range

**Edge Cases**

- **n_bootstrap = 0**: Returns zero stability scores
- **Small clusters**: May have low stability due to sampling variance
- **Perfect stability**: All bootstrap iterations yield identical clusters

**Data Guarantees (Preconditions)**

- `stability_scores` has capacity >= n_clusters
- `cluster_labels` contains valid cluster indices
- Expression matrix must be valid CSR format
- Random seed ensures reproducibility

**Complexity Analysis**

- **Time**: O(n_bootstrap * n_cells * log(n_cells)) - clustering per bootstrap
- **Space**: O(n_cells) auxiliary space per thread

**Example**

```cpp
scl::Array<Real> stability_scores(n_clusters);

scl::kernel::subpopulation::cluster_stability(
    expression, cluster_labels, n_cells,
    stability_scores,
    100,  // n_bootstrap
    42    // seed
);

// stability_scores[c] contains stability score for cluster c
// Higher scores (closer to 1.0) indicate more stable clusters
```

---

## Configuration

Default parameters are defined in `scl::kernel::subpopulation::config`:

- `EPSILON = 1e-10`: Numerical tolerance
- `MIN_CLUSTER_SIZE = 10`: Minimum cluster size for subdivision
- `DEFAULT_K = 5`: Default k for k-means sub-clustering
- `MAX_ITERATIONS = 100`: Maximum iterations for clustering algorithms
- `DEFAULT_BOOTSTRAP = 100`: Default number of bootstrap iterations

---

## Notes

- Recursive sub-clustering builds hierarchical cluster trees
- Cluster stability helps identify robust vs. unstable clusters
- Bootstrap resampling provides statistical confidence in cluster assignments
- Stability scores can be used to filter unreliable clusters

## See Also

- [Clustering Modules](./leiden) - For clustering algorithms
- [Metrics Module](./metrics) - For cluster quality metrics
