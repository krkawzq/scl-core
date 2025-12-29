---
title: Graph Centrality
description: Centrality measures for graph analysis
---

# Graph Centrality

The `centrality` kernel provides efficient computation of various graph centrality measures.

## Overview

Centrality measures identify important nodes in graphs:
- PageRank: Importance based on random walks
- Betweenness: Nodes on many shortest paths
- Closeness: Average distance to all nodes
- Eigenvector: Importance based on neighbors' importance

## Functions

### `pagerank`

Compute PageRank centrality.

```cpp
template <typename T, bool IsCSR>
void pagerank(
    const Sparse<T, IsCSR>& graph,
    Array<Real> scores,
    Real damping = config::DEFAULT_DAMPING,
    Index max_iter = config::DEFAULT_MAX_ITER
);
```

**Parameters**:
- `graph` [in]: Adjacency matrix
- `scores` [out]: PageRank scores
- `damping` [in]: Damping factor (default: 0.85)
- `max_iter` [in]: Maximum iterations (default: 100)

**Example**:
```cpp
#include "scl/kernel/centrality.hpp"

auto scores = memory::aligned_alloc<Real>(graph.rows());
Array<Real> scores_view = {scores.get(), static_cast<Size>(graph.rows())};

kernel::centrality::pagerank(graph, scores_view, damping=0.85);
```

## Configuration

```cpp
namespace scl::kernel::centrality::config {
    constexpr Real DEFAULT_DAMPING = 0.85;
    constexpr Index DEFAULT_MAX_ITER = 100;
    constexpr Real DEFAULT_TOLERANCE = 1e-6;
    constexpr Size PARALLEL_THRESHOLD = 256;
}
```

## Related Documentation

- [Louvain](./louvain.md) - Community detection
- [Kernels Overview](./overview.md) - General kernel usage
