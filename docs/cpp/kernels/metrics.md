# Clustering Metrics

Clustering and integration quality metrics for evaluating analysis results.

## Overview

The `metrics` module provides comprehensive evaluation metrics:

- **Silhouette score**: Measure cluster quality and separation
- **ARI/NMI**: Compare clusterings (Adjusted Rand Index, Normalized Mutual Information)
- **Graph connectivity**: Measure cluster connectivity in graphs
- **Batch mixing**: Evaluate batch correction quality (LISI, batch entropy)

All operations are:
- Parallelized where applicable
- Statistically rigorous
- Memory-efficient

## Clustering Quality Metrics

### silhouette_score

Compute the mean Silhouette Coefficient across all samples.

```cpp
#include "scl/kernel/metrics.hpp"

Sparse<Real, true> distances = /* pairwise distance matrix */;
Array<const Index> labels = /* cluster assignments */;

Real score = scl::kernel::metrics::silhouette_score(distances, labels);
```

**Parameters:**
- `distances` [in] - Pairwise distance matrix (cells x cells, CSR)
- `labels` [in] - Cluster assignments for each cell

**Returns:**
- Mean silhouette score in range [-1, 1]
- 1 indicates dense, well-separated clusters
- -1 indicates incorrect clustering

**Preconditions:**
- `distances.rows() == labels.len`
- At least 2 cells and 2 clusters for meaningful result
- Distance values should be non-negative

**Postconditions:**
- Returns 0 if fewer than 2 cells or clusters
- Singleton clusters are excluded from computation

**Complexity:**
- Time: O(n * nnz_per_row * n_clusters)
- Space: O(n_clusters) auxiliary

**Thread Safety:** Unsafe - sequential implementation

### silhouette_samples

Compute Silhouette Coefficient for each individual sample.

```cpp
Array<Real> scores(n_cells);
scl::kernel::metrics::silhouette_samples(distances, labels, scores);
```

**Parameters:**
- `distances` [in] - Pairwise distance matrix
- `labels` [in] - Cluster assignments
- `scores` [out] - Per-sample silhouette scores

**Preconditions:**
- `distances.rows() == labels.len == scores.len`
- At least 2 cells and 2 clusters

**Postconditions:**
- `scores[i]` = silhouette for cell i, in [-1, 1]
- Cells in singleton clusters have score 0

**Complexity:**
- Time: O(n * nnz_per_row * n_clusters)
- Space: O(n_clusters * n_threads) for thread-local buffers

**Thread Safety:** Safe - parallelized over cells with WorkspacePool

## Clustering Comparison

### adjusted_rand_index

Compute the Adjusted Rand Index between two clusterings.

```cpp
Array<const Index> labels1 = /* first clustering */;
Array<const Index> labels2 = /* second clustering */;

Real ari = scl::kernel::metrics::adjusted_rand_index(labels1, labels2);
```

**Parameters:**
- `labels1` [in] - First clustering assignment
- `labels2` [in] - Second clustering assignment

**Returns:**
- ARI score in range [-1, 1]
- 1 indicates identical clusterings
- 0 indicates random labeling

**Preconditions:**
- `labels1.len == labels2.len`
- Labels must be non-negative integers

**Postconditions:**
- Returns 1.0 for identical clusterings
- Returns 0.0 on average for random clusterings

**Complexity:**
- Time: O(n + n_clusters1 * n_clusters2)
- Space: O(n_clusters1 * n_clusters2) for contingency table

**Thread Safety:** Unsafe - sequential implementation

### normalized_mutual_information

Compute Normalized Mutual Information between two clusterings.

```cpp
Real nmi = scl::kernel::metrics::normalized_mutual_information(labels1, labels2);
```

**Parameters:**
- `labels1` [in] - First clustering assignment
- `labels2` [in] - Second clustering assignment

**Returns:**
- NMI score in range [0, 1]
- 1 indicates perfect agreement

**Preconditions:**
- `labels1.len == labels2.len`
- Labels must be non-negative integers

**Postconditions:**
- Returns 1.0 for identical clusterings
- Returns 0.0 for independent clusterings

**Complexity:**
- Time: O(n + n_clusters1 * n_clusters2)
- Space: O(n_clusters1 * n_clusters2)

**Thread Safety:** Unsafe - sequential implementation

## Batch Integration Metrics

### batch_entropy

Compute normalized entropy of batch distribution in each cell's neighborhood.

```cpp
Sparse<Index, true> neighbors = /* KNN graph */;
Array<const Index> batch_labels = /* batch assignments */;
Array<Real> entropy_scores(n_cells);

scl::kernel::metrics::batch_entropy(neighbors, batch_labels, entropy_scores);
```

**Parameters:**
- `neighbors` [in] - KNN graph (cells x cells, CSR)
- `batch_labels` [in] - Batch assignment for each cell
- `entropy_scores` [out] - Per-cell normalized entropy

**Preconditions:**
- `neighbors.rows() == batch_labels.len == entropy_scores.len`
- Batch labels must be non-negative integers

**Postconditions:**
- `entropy_scores[i]` in [0, 1]
- 1 indicates perfect batch mixing (uniform distribution)
- 0 indicates single batch in neighborhood

**Complexity:**
- Time: O(n * k) where k = average neighbors per cell
- Space: O(n_batches * n_threads) for thread-local counters

**Thread Safety:** Safe - parallelized with WorkspacePool

### lisi

Compute Local Inverse Simpson's Index for measuring label diversity.

```cpp
Array<Real> lisi_scores(n_cells);
scl::kernel::metrics::lisi(neighbors, labels, lisi_scores);
```

**Parameters:**
- `neighbors` [in] - KNN graph
- `labels` [in] - Label assignments (batch or cell type)
- `lisi_scores` [out] - Per-cell LISI scores

**Preconditions:**
- `neighbors.rows() == labels.len == lisi_scores.len`
- Labels must be non-negative integers

**Postconditions:**
- `lisi_scores[i] >= 1`
- LISI = 1 when all neighbors have same label
- LISI approaches n_labels for uniform distribution

**Complexity:**
- Time: O(n * k) where k = average neighbors
- Space: O(n_labels * n_threads) for thread-local counters

**Thread Safety:** Safe - parallelized with WorkspacePool

### graph_connectivity

Measure cluster connectivity as fraction of clusters that are fully connected.

```cpp
Sparse<Real, true> adjacency = /* cell neighborhood graph */;
Array<const Index> labels = /* cluster assignments */;

Real connectivity = scl::kernel::metrics::graph_connectivity(adjacency, labels);
```

**Parameters:**
- `adjacency` [in] - Cell neighborhood graph
- `labels` [in] - Cluster assignments

**Returns:**
- Fraction of clusters that are connected, in [0, 1]

**Preconditions:**
- `adjacency.rows() == labels.len`
- Adjacency should be symmetric for undirected connectivity

**Postconditions:**
- Returns 1.0 if all clusters are fully connected
- Returns 0.0 if all clusters are fragmented

**Complexity:**
- Time: O(n + nnz)
- Space: O(n) for component IDs and BFS queue

**Thread Safety:** Unsafe - sequential BFS

## Additional Metrics

### fowlkes_mallows_index

Compute Fowlkes-Mallows Index measuring similarity between clusterings.

```cpp
Real fmi = scl::kernel::metrics::fowlkes_mallows_index(labels1, labels2);
```

**Returns:**
- FMI in range [0, 1], geometric mean of precision and recall

**Complexity:**
- Time: O(n + n_clusters1 * n_clusters2)
- Space: O(n_clusters1 * n_clusters2)

**Thread Safety:** Unsafe - sequential implementation

### v_measure

Compute V-measure, the harmonic mean of homogeneity and completeness.

```cpp
Real v_score = scl::kernel::metrics::v_measure(
    labels_true, labels_pred, 1.0
);
```

**Parameters:**
- `labels_true` [in] - Ground truth labels
- `labels_pred` [in] - Predicted cluster labels
- `beta` [in] - Weight for homogeneity vs completeness (default: 1.0)

**Returns:**
- V-measure in range [0, 1]

**Preconditions:**
- `labels_true.len == labels_pred.len`
- `beta >= 0` (beta=1 gives equal weight)

**Postconditions:**
- Returns 1.0 for perfect clustering
- `beta > 1` weights completeness more
- `beta < 1` weights homogeneity more

**Complexity:**
- Time: O(n + n_classes * n_clusters)
- Space: O(n_classes * n_clusters)

**Thread Safety:** Unsafe - sequential implementation

## Configuration

```cpp
namespace scl::kernel::metrics::config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Real LOG2_E = Real(1.4426950408889634);
    constexpr Size PARALLEL_THRESHOLD = 256;
}
```

## Use Cases

### Clustering Evaluation

```cpp
// Evaluate clustering quality
Sparse<Real, true> distances = /* pairwise distances */;
Array<const Index> labels = /* cluster labels */;

Real silhouette = scl::kernel::metrics::silhouette_score(distances, labels);
// Higher is better (closer to 1)

// Compare two clusterings
Array<const Index> labels2 = /* alternative clustering */;
Real ari = scl::kernel::metrics::adjusted_rand_index(labels, labels2);
Real nmi = scl::kernel::metrics::normalized_mutual_information(labels, labels2);
```

### Batch Integration Assessment

```cpp
// Evaluate batch correction
Sparse<Index, true> knn = /* KNN graph */;
Array<const Index> batches = /* batch labels */;
Array<Real> entropy(n_cells);

scl::kernel::metrics::batch_entropy(knn, batches, entropy);
Real mean_entropy = scl::kernel::metrics::mean_batch_entropy(knn, batches);
// Higher entropy = better batch mixing

// LISI for batch diversity
Array<Real> lisi(n_cells);
scl::kernel::metrics::lisi(knn, batches, lisi);
Real mean_lisi = scl::kernel::metrics::mean_lisi(knn, batches);
// Higher LISI = more diverse batches in neighborhoods
```

### Cluster Connectivity

```cpp
// Check if clusters are well-connected in graph
Sparse<Real, true> graph = /* cell graph */;
Array<const Index> clusters = /* cluster labels */;

Real connectivity = scl::kernel::metrics::graph_connectivity(graph, clusters);
// 1.0 = all clusters fully connected
// 0.0 = all clusters fragmented
```

## Performance

- **Parallelization**: Scales linearly with number of cells
- **Memory efficient**: Minimal allocations for large datasets
- **SIMD acceleration**: Vectorized operations for mean computations
- **Workspace pooling**: Thread-local buffers for parallel operations

---

::: tip Metric Selection
- **Silhouette**: Best for evaluating cluster quality without ground truth
- **ARI/NMI**: Best for comparing clusterings or evaluating against ground truth
- **LISI/Batch Entropy**: Best for evaluating batch correction quality
- **Graph Connectivity**: Best for evaluating spatial/spatial transcriptomics clusters
:::

