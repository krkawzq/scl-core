---
title: Clustering Metrics
description: Quality metrics for clustering and integration evaluation
---

# Clustering Metrics

The `metrics` kernel provides efficient computation of clustering quality metrics.

## Overview

Clustering metrics are used for:
- Evaluating clustering quality
- Comparing different clusterings
- Integration quality assessment
- Batch mixing evaluation

## Functions

### `silhouette_score`

Compute silhouette score for clustering.

```cpp
template <typename T, bool IsCSR>
Real silhouette_score(
    const Sparse<T, IsCSR>& distances,
    Array<const Index> labels
);
```

**Parameters**:
- `distances` [in]: Pairwise distance matrix
- `labels` [in]: Cluster labels

**Returns**: Silhouette score (range: -1 to 1)

**Example**:
```cpp
#include "scl/kernel/metrics.hpp"

Array<Index> labels = {labels_ptr, n_cells};
Real score = kernel::metrics::silhouette_score(distances, labels);
```

### `adjusted_rand_index`

Compute Adjusted Rand Index (ARI).

```cpp
Real adjusted_rand_index(
    Array<const Index> labels1,
    Array<const Index> labels2
);
```

**Parameters**:
- `labels1` [in]: First clustering
- `labels2` [in]: Second clustering

**Returns**: ARI score (range: -1 to 1)

### `normalized_mutual_info`

Compute Normalized Mutual Information (NMI).

```cpp
Real normalized_mutual_info(
    Array<const Index> labels1,
    Array<const Index> labels2
);
```

## Configuration

```cpp
namespace scl::kernel::metrics::config {
    constexpr Real EPSILON = 1e-10;
    constexpr Size PARALLEL_THRESHOLD = 256;
}
```

## Related Documentation

- [Louvain](./louvain.md) - Clustering algorithms
- [Statistics](./statistics.md) - Statistical tests
