# Batch 8 Kernels

> Batch 8: Clustering, Markers, Merging, and Metrics

## Overview

This batch contains kernels for:

- **Louvain Clustering** - Multi-level community detection algorithm
- **Marker Selection** - Gene marker identification and specificity scoring
- **Matrix Merging** - Horizontal and vertical matrix concatenation
- **Quality Metrics** - Clustering and integration quality evaluation

## Files

| File | Description | Main APIs |
|------|-------------|-----------|
| [louvain.hpp](./louvain) | Louvain Community Detection | `cluster`, `compute_modularity`, `community_sizes`, `get_community_members` |
| [markers.hpp](./markers) | Marker Gene Selection | `find_markers`, `specificity_score` |
| [merge.hpp](./merge) | Matrix Merging Operations | `vstack`, `hstack` |
| [metrics.hpp](./metrics) | Quality Metrics | `silhouette_score`, `adjusted_rand_index`, `normalized_mutual_information`, `graph_connectivity`, `batch_entropy`, `lisi` |

## Quick Start

### Louvain Clustering

```cpp
#include "scl/kernel/louvain.hpp"

Sparse<Real, true> adjacency = /* adjacency matrix */;
Array<Index> labels(n_nodes);

scl::kernel::louvain::cluster(adjacency, labels, resolution = 1.0);
```

### Marker Selection

```cpp
#include "scl/kernel/markers.hpp"

Sparse<Real, true> expression = /* expression matrix */;
Array<Index> cluster_labels = /* cluster assignments */;
Index* marker_genes = /* allocate */;
Real* marker_scores = /* allocate */;

scl::kernel::markers::find_markers(
    expression, cluster_labels, n_cells, n_genes, n_clusters,
    marker_genes, marker_scores, max_markers = 50
);
```

### Matrix Merging

```cpp
#include "scl/kernel/merge.hpp"

Sparse<Real, true> matrix1 = /* first matrix */;
Sparse<Real, true> matrix2 = /* second matrix */;

// Vertical stacking
auto vstacked = scl::kernel::merge::vstack(matrix1, matrix2);

// Horizontal stacking
auto hstacked = scl::kernel::merge::hstack(matrix1, matrix2);
```

### Quality Metrics

```cpp
#include "scl/kernel/metrics.hpp"

Sparse<Real, true> distances = /* pairwise distances */;
Array<Index> labels = /* cluster labels */;

// Silhouette score
Real score = scl::kernel::metrics::silhouette_score(distances, labels);

// Adjusted Rand Index
Array<Index> labels1 = /* first clustering */;
Array<Index> labels2 = /* second clustering */;
Real ari = scl::kernel::metrics::adjusted_rand_index(labels1, labels2);
```

## See Also

- [Leiden Clustering](./leiden) - Alternative community detection algorithm
- [Neighbors](./neighbors) - KNN graph construction for clustering
- [Statistics](./statistics) - Statistical tests and analysis

