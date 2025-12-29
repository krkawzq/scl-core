---
title: Connected Components
description: Graph connectivity analysis and component detection
---

# Connected Components

The `components` kernel provides efficient algorithms for finding connected components in sparse graphs.

## Overview

Connected components analysis is used for:
- Finding disconnected subgraphs
- Graph connectivity assessment
- Preprocessing for clustering
- Quality control

## Functions

### `connected_components`

Find all connected components in a graph.

```cpp
template <typename T, bool IsCSR>
void connected_components(
    const Sparse<T, IsCSR>& graph,
    Array<Index> components
);
```

**Parameters**:
- `graph` [in]: Adjacency matrix
- `components` [out]: Component ID for each node

**Example**:
```cpp
#include "scl/kernel/components.hpp"

CSR graph = build_graph(data);
auto components = memory::aligned_alloc<Index>(graph.rows());
Array<Index> comp_view = {components.get(), static_cast<Size>(graph.rows())};

kernel::components::connected_components(graph, comp_view);
```

## Configuration

```cpp
namespace scl::kernel::components::config {
    constexpr Size PARALLEL_NODES_THRESHOLD = 1000;
    constexpr Size PARALLEL_EDGES_THRESHOLD = 10000;
    constexpr Size PREFETCH_DISTANCE = 8;
}
```

## Related Documentation

- [Louvain](./louvain.md) - Community detection
- [Kernels Overview](./overview.md) - General kernel usage
