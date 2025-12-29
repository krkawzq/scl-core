---
title: Leiden Clustering
description: High-performance Leiden algorithm for community detection
---

# Leiden Clustering

The `leiden` kernel implements the Leiden algorithm, an improved variant of Louvain for community detection with better guarantees on community quality.

## Overview

The Leiden algorithm improves upon Louvain by:
- Guaranteeing well-connected communities
- Better modularity optimization
- Faster convergence
- More stable results

## Functions

### `leiden`

Main function for Leiden community detection.

```cpp
template <typename T, bool IsCSR>
void leiden(
    const Sparse<T, IsCSR>& graph,
    Array<Index> communities,
    Real resolution = config::DEFAULT_RESOLUTION,
    Index max_iter = config::DEFAULT_MAX_ITER
);
```

**Parameters**:
- `graph` [in]: Adjacency matrix (sparse, symmetric)
- `communities` [out]: Community assignments
- `resolution` [in]: Resolution parameter (default: 1.0)
- `max_iter` [in]: Maximum iterations (default: 10)

**Example**:
```cpp
#include "scl/kernel/leiden.hpp"

CSR graph = build_neighbor_graph(data, k=15);
auto communities = memory::aligned_alloc<Index>(graph.rows());
Array<Index> comm_view = {communities.get(), static_cast<Size>(graph.rows())};

kernel::leiden::leiden(graph, comm_view, resolution=1.0);
```

## Configuration

```cpp
namespace scl::kernel::leiden::config {
    constexpr Real DEFAULT_RESOLUTION = 1.0;
    constexpr Index DEFAULT_MAX_ITER = 10;
    constexpr Real THETA = 0.05;  // Refinement parameter
    constexpr Size PARALLEL_THRESHOLD = 500;
}
```

## Related Documentation

- [Louvain](./louvain.md) - Louvain algorithm
- [Components](./components.md) - Connected components
