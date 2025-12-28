# Subpopulation Analysis

Cluster refinement and stability assessment for subpopulation identification.

## Overview

Subpopulation analysis kernels provide:

- **Recursive Subclustering** - Hierarchical cluster refinement
- **Cluster Stability** - Bootstrap-based stability assessment
- **Quality Control** - Identify robust vs. unstable clusters
- **Fine-grained Analysis** - Discover subpopulations within clusters

## Recursive Subclustering

### recursive_subclustering

Perform recursive sub-clustering within clusters:

```cpp
#include "scl/kernel/subpopulation.hpp"

Sparse<Real, true> expression = /* ... */;      // Expression matrix [n_cells x n_genes]
Array<Index> cluster_labels = /* ... */;        // Initial cluster labels [n_cells]
Index n_cells = expression.rows();
Array<Index> subcluster_labels(n_cells);        // Pre-allocated output

// Standard subclustering (max_depth=3, min_size=10)
scl::kernel::subpopulation::recursive_subclustering(
    expression, cluster_labels, n_cells, subcluster_labels);

// With custom parameters
scl::kernel::subpopulation::recursive_subclustering(
    expression, cluster_labels, n_cells, subcluster_labels,
    max_depth = 4,                              // Deeper hierarchy
    min_size = 20                               // Larger minimum cluster size
);
```

**Parameters:**
- `expression`: Expression matrix (cells × genes, CSR format)
- `cluster_labels`: Initial cluster labels, size = n_cells
- `n_cells`: Number of cells
- `subcluster_labels`: Output subcluster labels, must be pre-allocated, size = n_cells
- `max_depth`: Maximum recursion depth (default: 3)
- `min_size`: Minimum cluster size for splitting (default: 10)

**Postconditions:**
- `subcluster_labels` contains refined subcluster assignments
- Subclusters are hierarchical (depth indicated by label encoding)
- Original clusters are split into subpopulations when appropriate

**Algorithm:**
Recursive hierarchical clustering:
1. For each cluster:
   - If size < min_size: keep as leaf
   - Else: apply clustering algorithm to subdivide
   - Recursively process each subcluster
2. Continue until max_depth reached or clusters too small

**Complexity:**
- Time: O(max_depth * n_cells * log(n_cells)) per level
- Space: O(n_cells) auxiliary for labels and workspace

**Thread Safety:**
- Unsafe - recursive algorithm with shared state

**Use cases:**
- Fine-grained cell type identification
- Hierarchical cluster analysis
- Subpopulation discovery
- Cluster refinement for annotation

## Cluster Stability

### cluster_stability

Assess cluster stability using bootstrap resampling:

```cpp
Sparse<Real, true> expression = /* ... */;
Array<Index> cluster_labels = /* ... */;        // Cluster labels [n_cells]
Index n_clusters = /* number of unique clusters */;
Array<Real> stability_scores(n_clusters);       // Pre-allocated output

// Standard stability assessment (100 bootstrap iterations)
scl::kernel::subpopulation::cluster_stability(
    expression, cluster_labels, n_cells, stability_scores);

// With custom bootstrap iterations
scl::kernel::subpopulation::cluster_stability(
    expression, cluster_labels, n_cells, stability_scores,
    n_bootstrap = 200,                          // More iterations
    seed = 12345                                // Random seed
);
```

**Parameters:**
- `expression`: Expression matrix (cells × genes, CSR format)
- `cluster_labels`: Cluster labels, size = n_cells
- `n_cells`: Number of cells
- `stability_scores`: Output stability scores, must be pre-allocated, size = n_clusters
- `n_bootstrap`: Number of bootstrap iterations (default: 100)
- `seed`: Random seed for reproducibility (default: 42)

**Postconditions:**
- `stability_scores[c]` contains stability score for cluster c
- Scores typically in range [0, 1], higher = more stable
- Stable clusters are robust to data perturbation

**Algorithm:**
Bootstrap resampling approach:
1. For each bootstrap iteration:
   - Sample cells with replacement
   - Re-cluster sampled cells
   - Compute cluster overlap with original
2. Aggregate overlap scores across iterations
3. Stability = average agreement

**Complexity:**
- Time: O(n_bootstrap * n_cells * log(n_cells)) - dominated by re-clustering
- Space: O(n_cells) auxiliary for bootstrap samples and labels

**Thread Safety:**
- Safe - parallelized over bootstrap iterations
- Each iteration is independent

**Use cases:**
- Quality control for clustering results
- Identify robust vs. unstable clusters
- Guide cluster refinement decisions
- Validate clustering parameters

## Configuration

### Default Parameters

```cpp
namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_CLUSTER_SIZE = 10;
    constexpr Size DEFAULT_K = 5;
    constexpr Size MAX_ITERATIONS = 100;
    constexpr Size DEFAULT_BOOTSTRAP = 100;
}
```

**Minimum Cluster Size:**
- Clusters smaller than `MIN_CLUSTER_SIZE` are not split
- Prevents over-fragmentation
- Adjust based on dataset size

**Bootstrap Iterations:**
- More iterations = more reliable stability scores
- Default 100 is usually sufficient
- Increase for high-precision requirements

## Examples

### Hierarchical Subclustering

```cpp
#include "scl/kernel/subpopulation.hpp"

// Initial clustering
Sparse<Real, true> expression = /* ... */;
Array<Index> initial_labels = /* ... */;  // From Leiden/Louvain

// Refine with subclustering
Index n_cells = expression.rows();
Array<Index> refined_labels(n_cells);

scl::kernel::subpopulation::recursive_subclustering(
    expression, initial_labels, n_cells, refined_labels,
    max_depth = 3, min_size = 15);

// refined_labels now contains hierarchical subclusters
```

### Stability Assessment

```cpp
// Assess stability of clustering results
Array<Index> cluster_labels = /* ... */;
Index n_clusters = *std::max_element(cluster_labels.begin(),
                                     cluster_labels.end()) + 1;

Array<Real> stability(n_clusters);
scl::kernel::subpopulation::cluster_stability(
    expression, cluster_labels, n_cells, stability,
    n_bootstrap = 200);

// Filter unstable clusters
std::vector<Index> stable_clusters;
for (Index c = 0; c < n_clusters; ++c) {
    if (stability[c] > 0.7) {  // Threshold for stability
        stable_clusters.push_back(c);
    }
}

std::cout << "Found " << stable_clusters.size()
          << " stable clusters out of " << n_clusters << "\n";
```

### Combined Workflow

```cpp
// 1. Initial clustering
Array<Index> initial_labels(n_cells);
// ... perform initial clustering ...

// 2. Assess stability
Index n_clusters = /* ... */;
Array<Real> stability(n_clusters);
scl::kernel::subpopulation::cluster_stability(
    expression, initial_labels, n_cells, stability);

// 3. Refine stable clusters
Array<Index> refined_labels(n_cells);
scl::kernel::subpopulation::recursive_subclustering(
    expression, initial_labels, n_cells, refined_labels);

// 4. Filter by stability
// Use stability scores to identify reliable subclusters
```

## Performance Considerations

### Recursive Subclustering

- Computational cost scales with depth
- Large clusters require more computation
- Consider limiting max_depth for very large datasets

### Bootstrap Stability

- Parallelized over bootstrap iterations
- Each iteration requires full re-clustering
- Total time: O(n_bootstrap * clustering_time)
- Consider reducing n_bootstrap for initial exploration

---

::: tip Stability Threshold
Use stability scores to guide downstream analysis: focus on stable clusters for marker identification and annotation, treat unstable clusters with caution.
:::

