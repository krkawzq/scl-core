---
title: Louvain Clustering
description: Louvain algorithm for community detection
---

# Louvain Clustering

The `louvain` kernel implements the Louvain algorithm for community detection in sparse graphs, optimized with parallelization and efficient data structures.

## Overview

The Louvain algorithm is a greedy optimization method for community detection that:
- Maximizes modularity to find communities
- Uses a two-phase iterative approach (local optimization + aggregation)
- Supports hierarchical community detection
- Handles weighted and unweighted graphs

## Functions

### `louvain`

Main function for Louvain community detection.

```cpp
template <typename T, bool IsCSR>
void louvain(
    const Sparse<T, IsCSR>& graph,
    Array<Index> communities,
    Real resolution = config::DEFAULT_RESOLUTION,
    Index max_iter = config::DEFAULT_MAX_ITER
);
```

**Parameters**:
- `graph` [in]: Adjacency matrix (sparse, symmetric)
- `communities` [out]: Community assignments (length = graph.rows())
- `resolution` [in]: Resolution parameter (higher = more communities, default: 1.0)
- `max_iter` [in]: Maximum iterations per level (default: 100)

**Returns**: Community assignments in `communities` array

**Example**:
```cpp
#include "scl/kernel/louvain.hpp"

// Build adjacency graph
CSR graph = build_neighbor_graph(data, k=15);

// Allocate output
auto communities = memory::aligned_alloc<Index>(graph.rows());
Array<Index> comm_view = {communities.get(), static_cast<Size>(graph.rows())};

// Run Louvain
kernel::louvain::louvain(
    graph, comm_view,
    resolution=1.0,  // Standard resolution
    max_iter=100
);

// Use communities
for (Index i = 0; i < graph.rows(); ++i) {
    Index comm = comm_view[i];
    // Process cell i in community comm
}
```

**Complexity**: O(nnz * log(n)) time per iteration, O(n) space

## Algorithm Details

### Modularity

Modularity measures the quality of a community partition:

```
Q = (1/2m) * Σᵢⱼ [Aᵢⱼ - (kᵢkⱼ/2m)] * δ(cᵢ, cⱼ)
```

where:
- `m`: Total edge weight
- `Aᵢⱼ`: Edge weight between nodes i and j
- `kᵢ`: Degree of node i
- `cᵢ`: Community of node i
- `δ`: Kronecker delta (1 if same community, 0 otherwise)

### Two-Phase Algorithm

1. **Local Optimization**: Move nodes to maximize modularity gain
2. **Aggregation**: Build new graph with communities as nodes

### Hierarchical Clustering

The algorithm can be run hierarchically:

```cpp
// Level 1: Initial communities
kernel::louvain::louvain(graph, level1_communities);

// Aggregate graph
CSR aggregated = aggregate_graph(graph, level1_communities);

// Level 2: Refine communities
kernel::louvain::louvain(aggregated, level2_communities);
```

## Configuration

```cpp
namespace scl::kernel::louvain::config {
    constexpr Real DEFAULT_RESOLUTION = 1.0;
    constexpr Index DEFAULT_MAX_ITER = 100;
    constexpr Real MODULARITY_EPSILON = 1e-8;
    constexpr Size PARALLEL_THRESHOLD = 1000;
    constexpr Index MAX_LEVELS = 100;
}
```

## Performance Considerations

### Parallelization

The algorithm parallelizes node moves when graph size exceeds threshold:

```cpp
if (n_nodes >= config::PARALLEL_THRESHOLD) {
    // Parallel node movement
    threading::parallel_for(0, n_nodes, [&](size_t i) {
        optimize_node(i);
    });
}
```

### Efficient Data Structures

- **Hash Table**: O(1) community weight lookup
- **Fibonacci Hashing**: Better distribution than linear probing
- **SIMD**: Vectorized weight computations

### Memory Efficiency

- Block allocation for community data
- Reuse of data structures across iterations
- Minimal temporary allocations

## Common Patterns

### Finding Optimal Resolution

```cpp
Real find_optimal_resolution(
    const CSR& graph,
    Index target_n_communities
) {
    Real lo = 0.1, hi = 10.0;
    Real best_resolution = 1.0;
    
    for (int iter = 0; iter < 20; ++iter) {
        Real resolution = (lo + hi) / 2.0;
        
        auto communities = memory::aligned_alloc<Index>(graph.rows());
        Array<Index> comm_view = {communities.get(), static_cast<Size>(graph.rows())};
        
        kernel::louvain::louvain(graph, comm_view, resolution);
        
        Index n_communities = count_communities(comm_view);
        
        if (n_communities < target_n_communities) {
            hi = resolution;
        } else {
            lo = resolution;
            best_resolution = resolution;
        }
    }
    
    return best_resolution;
}
```

### Hierarchical Clustering

```cpp
void hierarchical_louvain(
    const CSR& graph,
    std::vector<Array<Index>>& all_levels
) {
    CSR current_graph = graph;
    Array<Index> current_communities = ...;
    
    for (Index level = 0; level < MAX_LEVELS; ++level) {
        // Run Louvain
        kernel::louvain::louvain(current_graph, current_communities);
        
        // Store level
        all_levels.push_back(current_communities);
        
        // Check convergence
        Index n_communities = count_communities(current_communities);
        if (n_communities == current_graph.rows()) {
            break;  // No further aggregation possible
        }
        
        // Aggregate graph
        current_graph = aggregate_graph(current_graph, current_communities);
        current_communities = initialize_communities(current_graph.rows());
    }
}
```

## Related Documentation

- [Leiden Algorithm](./leiden.md) - Improved Louvain variant
- [Components](./components.md) - Connected components
- [Kernels Overview](./overview.md) - General kernel usage
