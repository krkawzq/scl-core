# metrics.hpp

> scl/kernel/metrics.hpp · Clustering and integration quality metrics

## Overview

Comprehensive suite of quality metrics for evaluating clustering results, batch integration quality, and label similarity. These metrics are essential for assessing algorithm performance, data integration success, and biological interpretation.

This file provides:
- Clustering quality metrics (silhouette, ARI, NMI, purity)
- Batch integration metrics (batch entropy, LISI)
- Graph-based connectivity measures
- Label similarity comparisons

**Header**: `#include "scl/kernel/metrics.hpp"`

---

## Main APIs

### silhouette_score

::: source_code file="scl/kernel/metrics.hpp" symbol="silhouette_score" collapsed
:::

**Algorithm Description**

Computes the mean Silhouette Coefficient across all samples, measuring how similar each sample is to its own cluster compared to other clusters:

1. **For each cell i** with cluster label c:
   - Compute a(i) = mean distance to other cells in the same cluster c
   - Compute b(i) = minimum of mean distances to cells in each other cluster
   - Compute silhouette: s(i) = (b(i) - a(i)) / max(a(i), b(i))
2. **Return mean**: Mean of all s(i) values

The silhouette score ranges from -1 to 1:
- **1**: Cell is well-clustered (much closer to own cluster than others)
- **0**: Cell is on boundary between clusters
- **-1**: Cell is closer to other clusters than own cluster

**Edge Cases**

- **Fewer than 2 clusters**: Returns 0
- **Fewer than 2 cells**: Returns 0
- **Singleton clusters**: Excluded from computation (score = 0 for singleton members)
- **Single cell per cluster**: Returns 0 (no meaningful comparison)
- **Disconnected distance matrix**: Handles gracefully, only considers connected components

**Data Guarantees (Preconditions)**

- Distance matrix rows must equal labels array length
- Distance values should be non-negative (negative values may cause issues)
- At least 2 cells and 2 clusters for meaningful result
- Distance matrix should be symmetric (though not strictly required)

**Complexity Analysis**

- **Time**: O(n * nnz_per_row * n_clusters) where n is number of cells, nnz_per_row is average non-zeros per row (neighbors), and n_clusters is number of clusters. For each cell, we examine distances to all neighbors and compute means for all clusters.
- **Space**: O(n_clusters) auxiliary space for storing per-cluster statistics

**Example**

```cpp
#include "scl/kernel/metrics.hpp"

Sparse<Real, true> distances = /* pairwise distance matrix [n_cells x n_cells] */;
Array<Index> labels = /* cluster assignments [n_cells] */;

// Compute mean silhouette score
Real score = scl::kernel::metrics::silhouette_score(distances, labels);

// Score ranges from -1 to 1
// Higher scores indicate better clustering
// score > 0.5: strong clustering
// score < 0: poor clustering (cells closer to other clusters)
```

---

### adjusted_rand_index

::: source_code file="scl/kernel/metrics.hpp" symbol="adjusted_rand_index" collapsed
:::

**Algorithm Description**

Computes the Adjusted Rand Index (ARI) between two clusterings, measuring similarity adjusted for chance:

1. **Build contingency table**: Count pairs of samples that are:
   - In same cluster in both clusterings (n_ij)
   - In same cluster only in first clustering (a_i)
   - In same cluster only in second clustering (b_j)
2. **Compute sums**:
   - sum_nij = sum of C(n_ij, 2) for all pairs of clusters
   - sum_ai = sum of C(a_i, 2) for all clusters in first clustering
   - sum_bj = sum of C(b_j, 2) for all clusters in second clustering
3. **Compute ARI**:
   - expected = (sum_ai * sum_bj) / C(n, 2)
   - ARI = (sum_nij - expected) / (mean - expected)
   - where mean = (sum_ai + sum_bj) / 2

ARI ranges from -1 to 1:
- **1**: Identical clusterings
- **0**: Random labeling (on average)
- **-1**: Maximum disagreement

**Edge Cases**

- **Identical clusterings**: Returns 1.0
- **Independent clusterings**: Returns approximately 0.0 (on average)
- **Empty labels**: Returns 0.0
- **Single cluster in both**: Returns 1.0 (trivial agreement)
- **Different number of clusters**: Handled correctly via contingency table

**Data Guarantees (Preconditions)**

- Labels1 and labels2 must have equal length
- All label values must be non-negative integers
- Label values don't need to be contiguous (will be mapped internally)

**Complexity Analysis**

- **Time**: O(n + n_clusters1 * n_clusters2) where n is number of samples, n_clusters1 and n_clusters2 are number of clusters in each clustering. Building contingency table is O(n), computing sums is O(n_clusters1 * n_clusters2).
- **Space**: O(n_clusters1 * n_clusters2) for contingency table storage

**Example**

```cpp
#include "scl/kernel/metrics.hpp"

Array<Index> labels1 = /* first clustering [n] */;
Array<Index> labels2 = /* second clustering [n] */;

// Compute ARI between two clusterings
Real ari = scl::kernel::metrics::adjusted_rand_index(labels1, labels2);

// ARI = 1.0: Perfect agreement
// ARI = 0.0: Random agreement (on average)
// ARI < 0: Less agreement than random
```

---

### normalized_mutual_information

::: source_code file="scl/kernel/metrics.hpp" symbol="normalized_mutual_information" collapsed
:::

**Algorithm Description**

Computes Normalized Mutual Information (NMI) between two clusterings:

1. **Build contingency table**: Count co-occurrences n_ij (samples in cluster i of first clustering and cluster j of second clustering)
2. **Compute entropies**:
   - H(labels1) = -sum_i (n_i / n) * log2(n_i / n)
   - H(labels2) = -sum_j (n_j / n) * log2(n_j / n)
3. **Compute mutual information**:
   - MI = sum_ij (n_ij / n) * log2((n_ij * n) / (n_i * n_j))
4. **Normalize**:
   - NMI = 2 * MI / (H(labels1) + H(labels2))

NMI ranges from 0 to 1:
- **1**: Perfect agreement
- **0**: Independent clusterings (no mutual information)

**Edge Cases**

- **Identical clusterings**: Returns 1.0
- **Independent clusterings**: Returns 0.0
- **Empty labels**: Returns 0.0
- **Single cluster in one clustering**: Returns 0.0 (no information)

**Data Guarantees (Preconditions)**

- Labels1 and labels2 must have equal length
- All label values must be non-negative integers

**Complexity Analysis**

- **Time**: O(n + n_clusters1 * n_clusters2) - building contingency table is O(n), computing entropies and MI is O(n_clusters1 * n_clusters2)
- **Space**: O(n_clusters1 * n_clusters2) for contingency table

**Example**

```cpp
#include "scl/kernel/metrics.hpp"

Array<Index> labels1 = /* first clustering */;
Array<Index> labels2 = /* second clustering */;

Real nmi = scl::kernel::metrics::normalized_mutual_information(labels1, labels2);

// NMI ranges from 0 to 1
// Higher values indicate better agreement
```

---

### graph_connectivity

::: source_code file="scl/kernel/metrics.hpp" symbol="graph_connectivity" collapsed
:::

**Algorithm Description**

Measures cluster connectivity as the fraction of clusters that are fully connected (single connected component) in the graph:

1. **For each cluster c**:
   - Extract subgraph induced by nodes with label c
   - Perform BFS (Breadth-First Search) to find connected components
   - Count number of connected components
2. **Count connected clusters**: Clusters with exactly one component
3. **Return ratio**: connected_clusters / total_clusters

This metric is important for single-cell analysis where clusters should form connected regions in the cell-cell similarity graph.

**Edge Cases**

- **All clusters connected**: Returns 1.0
- **All clusters fragmented**: Returns 0.0
- **Empty adjacency matrix**: Returns 0.0
- **Disconnected graph**: Each connected component in the full graph forms a separate cluster component
- **Singleton clusters**: Count as connected (single node = single component)

**Data Guarantees (Preconditions)**

- Adjacency matrix rows must equal labels array length
- Adjacency should be symmetric for undirected connectivity
- Graph should represent cell-cell similarity (e.g., KNN graph)

**Complexity Analysis**

- **Time**: O(n + nnz) where n is number of cells and nnz is number of edges. BFS for each cluster visits all nodes and edges in that cluster.
- **Space**: O(n) for component IDs and BFS queue storage

**Example**

```cpp
#include "scl/kernel/metrics.hpp"

Sparse<Real, true> adjacency = /* cell-cell similarity graph [n_cells x n_cells] */;
Array<Index> labels = /* cluster labels [n_cells] */;

Real connectivity = scl::kernel::metrics::graph_connectivity(adjacency, labels);

// connectivity = 1.0: All clusters are fully connected
// connectivity = 0.5: Half of clusters are connected
// Lower values indicate fragmented clusters
```

---

### batch_entropy

::: source_code file="scl/kernel/metrics.hpp" symbol="batch_entropy" collapsed
:::

**Algorithm Description**

Computes normalized entropy of batch distribution in each cell's neighborhood, measuring batch mixing quality:

1. **For each cell i** (in parallel):
   - Get neighborhood: neighbors in KNN graph (including self)
   - Count batch occurrences: count how many neighbors belong to each batch
   - Compute Shannon entropy: H = -sum_b (p_b * log2(p_b)) where p_b is proportion of neighbors in batch b
   - Normalize: normalized_entropy = H / log2(n_batches)
2. **Store per-cell scores**: entropy_scores[i] = normalized entropy for cell i

Normalized entropy ranges from 0 to 1:
- **1**: Perfect batch mixing (uniform distribution across batches in neighborhood)
- **0**: No mixing (all neighbors from single batch)

**Edge Cases**

- **Single batch**: All scores are 0 (log(n_batches) = 0, handled gracefully)
- **Perfect mixing**: All scores approach 1.0
- **No mixing**: Scores are 0.0
- **Small neighborhoods**: Entropy may be biased by neighborhood size

**Data Guarantees (Preconditions)**

- Neighbors matrix rows must equal batch_labels array length
- Neighbors matrix rows must equal entropy_scores array length
- Batch labels must be non-negative integers
- KNN graph should be symmetric (though not strictly required)

**Complexity Analysis**

- **Time**: O(n * k) where n is number of cells and k is average number of neighbors per cell. For each cell, we examine k neighbors.
- **Space**: O(n_batches * n_threads) for thread-local batch counters during parallel processing

**Example**

```cpp
#include "scl/kernel/metrics.hpp"

Sparse<Index, true> neighbors = /* KNN graph [n_cells x n_cells] */;
Array<Index> batch_labels = /* batch assignments [n_cells] */;
Array<Real> entropy_scores(n_cells);

scl::kernel::metrics::batch_entropy(neighbors, batch_labels, entropy_scores);

// entropy_scores[i] = normalized batch entropy in cell i's neighborhood
// Higher values indicate better batch mixing
```

---

### lisi

::: source_code file="scl/kernel/metrics.hpp" symbol="lisi" collapsed
:::

**Algorithm Description**

Computes Local Inverse Simpson's Index (LISI) for measuring label diversity in local neighborhoods:

1. **For each cell i** (in parallel):
   - Get neighborhood: neighbors in KNN graph (including self)
   - Count label occurrences: count how many neighbors have each label value
   - Compute Simpson's index: SI = sum_b (p_b^2) where p_b is proportion of neighbors with label b
   - Compute LISI: LISI = 1 / SI
2. **Store per-cell scores**: lisi_scores[i] = LISI for cell i

LISI ranges from 1 to n_labels:
- **1**: All neighbors have same label (no diversity)
- **n_labels**: Perfect diversity (uniform distribution across all labels)

Higher LISI values indicate greater label diversity in the neighborhood.

**Edge Cases**

- **Uniform distribution**: LISI approaches number of unique labels
- **Single label in neighborhood**: LISI = 1.0
- **Empty neighborhood**: Handled gracefully (self-only)
- **Tied proportions**: LISI computed correctly

**Data Guarantees (Preconditions)**

- Neighbors matrix rows must equal labels array length
- Neighbors matrix rows must equal lisi_scores array length
- Labels must be non-negative integers
- KNN graph should represent meaningful cell-cell similarity

**Complexity Analysis**

- **Time**: O(n * k) where n is number of cells and k is average neighbors per cell
- **Space**: O(n_labels * n_threads) for thread-local label counters during parallel processing

**Example**

```cpp
#include "scl/kernel/metrics.hpp"

Sparse<Index, true> neighbors = /* KNN graph */;
Array<Index> labels = /* labels (batch or cell type) [n_cells] */;
Array<Real> lisi_scores(n_cells);

scl::kernel::metrics::lisi(neighbors, labels, lisi_scores);

// lisi_scores[i] = LISI for cell i
// LISI = 1: No diversity (single label in neighborhood)
// Higher LISI: More diverse labels in neighborhood
```

---

### silhouette_samples

::: source_code file="scl/kernel/metrics.hpp" symbol="silhouette_samples" collapsed
:::

**Algorithm Description**

Computes Silhouette Coefficient for each individual sample (same algorithm as silhouette_score but returns per-sample scores):

1. **For each cell i** (in parallel):
   - Compute a(i) = mean distance to other cells in same cluster
   - Compute b(i) = minimum mean distance to cells in other clusters
   - Compute s(i) = (b(i) - a(i)) / max(a(i), b(i))
2. **Store per-sample scores**: scores[i] = s(i)

Each score ranges from -1 to 1, with same interpretation as silhouette_score.

**Edge Cases**

- **Singleton clusters**: Members have score 0
- **Single-cell clusters**: Score is 0
- **Fewer than 2 clusters**: All scores are 0

**Data Guarantees (Preconditions)**

- Distance matrix rows must equal labels array length
- Labels array length must equal scores array length
- At least 2 clusters for meaningful scores

**Complexity Analysis**

- **Time**: O(n * nnz_per_row * n_clusters) - same as silhouette_score but computes all per-sample values
- **Space**: O(n_clusters * n_threads) for thread-local buffers during parallel processing

**Example**

```cpp
#include "scl/kernel/metrics.hpp"

Sparse<Real, true> distances = /* pairwise distances */;
Array<Index> labels = /* cluster labels */;
Array<Real> scores(n_cells);

scl::kernel::metrics::silhouette_samples(distances, labels, scores);

// scores[i] = silhouette score for cell i
// Can identify poorly clustered cells (scores < 0)
```

---

## Utility Functions

### fowlkes_mallows_index

Measures similarity between clusterings as geometric mean of precision and recall.

::: source_code file="scl/kernel/metrics.hpp" symbol="fowlkes_mallows_index" collapsed
:::

**Complexity**
- Time: O(n + n_clusters1 * n_clusters2)
- Space: O(n_clusters1 * n_clusters2)

---

### v_measure

Harmonic mean of homogeneity and completeness.

::: source_code file="scl/kernel/metrics.hpp" symbol="v_measure" collapsed
:::

**Complexity**
- Time: O(n + n_classes * n_clusters)
- Space: O(n_classes * n_clusters)

---

### homogeneity_score

Measures if each cluster contains only members of a single class.

::: source_code file="scl/kernel/metrics.hpp" symbol="homogeneity_score" collapsed
:::

**Complexity**
- Time: O(n + n_classes * n_clusters)
- Space: O(n_classes * n_clusters)

---

### completeness_score

Measures if all members of a class are assigned to the same cluster.

::: source_code file="scl/kernel/metrics.hpp" symbol="completeness_score" collapsed
:::

**Complexity**
- Time: O(n + n_classes * n_clusters)
- Space: O(n_classes * n_clusters)

---

### purity_score

Fraction of correctly assigned samples (majority class per cluster).

::: source_code file="scl/kernel/metrics.hpp" symbol="purity_score" collapsed
:::

**Complexity**
- Time: O(n + n_classes * n_clusters)
- Space: O(n_classes * n_clusters)

---

### mean_lisi

Computes mean LISI score across all cells.

::: source_code file="scl/kernel/metrics.hpp" symbol="mean_lisi" collapsed
:::

**Complexity**
- Time: O(n * k)
- Space: O(n) for intermediate scores

---

### mean_batch_entropy

Computes mean batch entropy across all cells.

::: source_code file="scl/kernel/metrics.hpp" symbol="mean_batch_entropy" collapsed
:::

**Complexity**
- Time: O(n * k)
- Space: O(n) for intermediate scores

---

## Configuration

The namespace `scl::kernel::metrics::config` provides configuration constants:

- `EPSILON = 1e-10`: Numerical stability constant
- `LOG2_E = 1.4426950408889634`: log2(e) for entropy calculations
- `PARALLEL_THRESHOLD = 256`: Minimum size for parallel processing

## Notes

- **Metric interpretation**: Different metrics emphasize different aspects of clustering quality. Use multiple metrics for comprehensive evaluation.
- **Batch integration**: Batch entropy and LISI are specifically designed for evaluating batch correction and integration quality.
- **Computational cost**: Some metrics (like silhouette_score) can be expensive for large datasets. Consider sampling for very large datasets.
- **Label encoding**: All label-based metrics assume non-negative integer labels but don't require contiguous labeling.

## See Also

- [Louvain Clustering](./louvain) - Community detection algorithm
- [Leiden Clustering](./leiden) - Alternative clustering algorithm
- [Neighbors](./neighbors) - KNN graph construction for connectivity metrics
