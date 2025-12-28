# components.hpp

> scl/kernel/components.hpp · High-performance connected components and graph connectivity analysis

## Overview

This file provides efficient graph connectivity algorithms for analyzing sparse graphs. It includes connected component detection, breadth-first search (BFS), graph metrics (diameter, triangle count), and connectivity checks. All operations are optimized for large-scale sparse graphs with parallel execution.

**Header**: `#include "scl/kernel/components.hpp"`

---

## Main APIs

### connected_components

::: source_code file="scl/kernel/components.hpp" symbol="connected_components" collapsed
:::

**Algorithm Description**

Finds all connected components in an undirected graph using parallel union-find:

1. Initialize union-find data structure with each node as its own component
2. For each edge (u, v) in parallel:
   - Find root of u and root of v
   - If roots differ, union the components using lock-free atomic operations
3. Path compression and union-by-rank optimizations for efficiency
4. Final pass assigns component labels to all nodes
5. Returns number of distinct components found

**Edge Cases**

- **Empty graph**: Returns n_components = 0 if no nodes exist
- **Disconnected graph**: Each isolated component gets unique label
- **Self-loops**: Handled correctly (node connected to itself)
- **Duplicate edges**: Union-find naturally handles multiple edges between same nodes

**Data Guarantees (Preconditions)**

- `component_labels.len >= adjacency.primary_dim()`
- Adjacency matrix represents undirected graph (symmetric)
- Graph is valid CSR or CSC format

**Complexity Analysis**

- **Time**: O(nnz * α(n)) where α is inverse Ackermann (effectively O(nnz) in practice)
- **Space**: O(n_nodes) auxiliary - union-find arrays

**Example**

```cpp
#include "scl/kernel/components.hpp"

Sparse<Real, true> adjacency = /* adjacency matrix, CSR */;
Array<Index> component_labels(adjacency.rows());
Index n_components;

scl::kernel::components::connected_components(
    adjacency,
    component_labels,
    n_components
);

// Filter to largest component
Index largest_component = 0;
Array<Index> component_sizes(n_components, 0);
for (Index i = 0; i < adjacency.rows(); ++i) {
    component_sizes[component_labels[i]]++;
    if (component_sizes[component_labels[i]] > 
        component_sizes[largest_component]) {
        largest_component = component_labels[i];
    }
}
```

---

### largest_component

::: source_code file="scl/kernel/components.hpp" symbol="largest_component" collapsed
:::

**Algorithm Description**

Extracts nodes in the largest connected component:

1. Call `connected_components` to find all components
2. Count nodes per component
3. Identify component with maximum size
4. Create binary mask where `node_mask[i] == 1` if node i is in largest component
5. Return size of largest component

**Edge Cases**

- **Empty graph**: Returns component_size = 0
- **Tie in sizes**: Returns first component with maximum size
- **All isolated**: Each node is its own component, returns size 1

**Data Guarantees (Preconditions)**

- `node_mask.len >= adjacency.primary_dim()`
- Adjacency matrix is valid CSR or CSC format

**Complexity Analysis**

- **Time**: O(nnz) - dominated by connected_components call
- **Space**: O(n_nodes) auxiliary - component labels and mask

**Example**

```cpp
Array<Byte> node_mask(adjacency.rows());
Index component_size;

scl::kernel::components::largest_component(
    adjacency,
    node_mask,
    component_size
);

// Filter graph to largest component
// (use node_mask to select nodes)
```

---

### bfs

::: source_code file="scl/kernel/components.hpp" symbol="bfs" collapsed
:::

**Algorithm Description**

Performs breadth-first search from source node to compute shortest path distances:

1. Initialize distance array with -1 (unvisited)
2. Create queue and enqueue source node with distance 0
3. While queue not empty:
   - Dequeue node u
   - For each neighbor v of u:
     - If v not visited (distance == -1), set distance[v] = distance[u] + 1
     - Enqueue v
4. Sequential implementation for correctness

**Edge Cases**

- **Unreachable nodes**: Distance remains -1 for nodes not reachable from source
- **Self-loops**: Source node has distance 0 to itself
- **Empty graph**: All distances remain -1 except source (distance 0)

**Data Guarantees (Preconditions)**

- `distances.len >= adjacency.primary_dim()`
- `visited.len >= adjacency.primary_dim()` (if provided)
- Source is valid node index (0 <= source < n_nodes)

**Complexity Analysis**

- **Time**: O(nnz) for connected component - visits each edge once
- **Space**: O(n_nodes) auxiliary - queue and distance arrays

**Example**

```cpp
Index source = 0;  // Root node
Array<Index> distances(adjacency.rows());
Array<Index> visited(adjacency.rows());

scl::kernel::components::bfs(
    adjacency,
    source,
    distances,
    visited
);

// Find nodes at distance k
for (Index i = 0; i < adjacency.rows(); ++i) {
    if (distances[i] == k) {
        // Node i is k hops from source
    }
}
```

---

### parallel_bfs

::: source_code file="scl/kernel/components.hpp" symbol="parallel_bfs" collapsed
:::

**Algorithm Description**

Performs parallel BFS using direction-optimizing algorithm:

1. Uses bit-vector frontiers for efficient parallel processing
2. Direction optimization: switches between top-down and bottom-up BFS based on frontier density
3. Top-down: expand from current frontier (sparse frontiers)
4. Bottom-up: check all unvisited nodes for neighbors in frontier (dense frontiers)
5. Parallel processing of frontier nodes/edges
6. More efficient than sequential BFS for large graphs

**Edge Cases**

- **Unreachable nodes**: Distance remains -1
- **Very sparse graphs**: Prefers top-down approach
- **Very dense graphs**: Automatically switches to bottom-up

**Data Guarantees (Preconditions)**

- `distances.len >= adjacency.primary_dim()`
- Source is valid node index

**Complexity Analysis**

- **Time**: O(nnz) for connected component - same as sequential but parallelized
- **Space**: O(n_nodes) auxiliary per thread - bit-vector frontiers

**Example**

```cpp
Index source = find_root_cell();
Array<Index> distances(adjacency.rows());

scl::kernel::components::parallel_bfs(
    adjacency,
    source,
    distances
);

// Use distances for downstream analysis (e.g., trajectory inference)
```

---

### is_connected

::: source_code file="scl/kernel/components.hpp" symbol="is_connected" collapsed
:::

**Algorithm Description**

Checks if graph is connected (has single connected component):

1. Call `connected_components` to find all components
2. Check if n_components == 1
3. Returns true if single component, false otherwise

**Edge Cases**

- **Empty graph**: Returns false (no nodes, cannot be connected)
- **Single node**: Returns true (trivially connected)
- **Disconnected**: Returns false if multiple components exist

**Data Guarantees (Preconditions)**

- Graph has at least one node
- Adjacency matrix is valid CSR or CSC format

**Complexity Analysis**

- **Time**: O(nnz) - dominated by connected_components
- **Space**: O(n_nodes) auxiliary

**Example**

```cpp
bool connected = scl::kernel::components::is_connected(adjacency);

if (!connected) {
    // Graph has multiple components, may need filtering
}
```

---

### graph_diameter

::: source_code file="scl/kernel/components.hpp" symbol="graph_diameter" collapsed
:::

**Algorithm Description**

Computes graph diameter (longest shortest path between any two nodes):

1. For each node in parallel:
   - Run BFS from that node
   - Find maximum distance in BFS result
2. Take maximum over all source nodes
3. Returns diameter value

**Edge Cases**

- **Disconnected graph**: Behavior undefined (should check connectivity first)
- **Single node**: Returns diameter 0
- **Linear graph**: Returns n_nodes - 1

**Data Guarantees (Preconditions)**

- Graph is connected (use `is_connected` first)
- Adjacency matrix is valid CSR or CSC format

**Complexity Analysis**

- **Time**: O(n_nodes * nnz) - BFS from each node
- **Space**: O(n_nodes) auxiliary per thread - BFS workspace

**Example**

```cpp
if (scl::kernel::components::is_connected(adjacency)) {
    Index diameter = scl::kernel::components::graph_diameter(adjacency);
    // Use diameter for graph analysis
}
```

---

### triangle_count

::: source_code file="scl/kernel/components.hpp" symbol="triangle_count" collapsed
:::

**Algorithm Description**

Counts triangles in undirected graph using optimized sparse algorithm:

1. For each node u in parallel:
   - For each neighbor v of u where v > u (avoid double counting):
     - For each neighbor w of u where w > v:
       - Check if edge (v, w) exists (binary search in adjacency)
       - If exists, increment triangle count
2. Uses atomic operations for thread-safe counting
3. Optimized for sparse graphs with early termination

**Edge Cases**

- **No triangles**: Returns 0
- **Complete graph**: Returns n_nodes * (n_nodes-1) * (n_nodes-2) / 6
- **Self-loops**: Not counted as triangles

**Data Guarantees (Preconditions)**

- Graph is undirected (symmetric adjacency)
- Adjacency matrix is valid CSR format

**Complexity Analysis**

- **Time**: O(nnz^1.5) for sparse graphs - nested loops with binary search
- **Space**: O(n_nodes) auxiliary - atomic counter

**Example**

```cpp
Size n_triangles = scl::kernel::components::triangle_count(adjacency);

// Compute clustering coefficient
Real clustering = 3.0 * n_triangles / (n_nodes * (n_nodes - 1));
```

---

## Configuration

Default parameters in `scl::kernel::components::config`:

- `INVALID_COMPONENT = -1`: Sentinel value for invalid component
- `UNVISITED = -1`: Sentinel value for unvisited nodes
- `PARALLEL_NODES_THRESHOLD = 1000`: Minimum nodes for parallel processing
- `PARALLEL_EDGES_THRESHOLD = 10000`: Minimum edges for parallel processing
- `DENSE_DEGREE_THRESHOLD = 64`: Threshold for dense vs sparse node handling

---

## See Also

- [Neighbors Module](./neighbors) - KNN graph construction
- [Sparse Matrix](../core/sparse) - Sparse matrix operations
