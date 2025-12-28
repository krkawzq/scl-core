# pseudotime.hpp

> scl/kernel/pseudotime.hpp · Pseudotime inference kernels for trajectory analysis

## Overview

This file provides kernels for computing pseudotime values along cell trajectories, including shortest path methods, diffusion pseudotime, and trajectory analysis utilities.

This file provides:
- Shortest path pseudotime (Dijkstra-based)
- Diffusion pseudotime (DPT)
- Root cell selection
- Branch point detection
- Trajectory segmentation
- Pseudotime smoothing and correlation

**Header**: `#include "scl/kernel/pseudotime.hpp"`

---

## Main APIs

### graph_pseudotime

::: source_code file="scl/kernel/pseudotime.hpp" symbol="graph_pseudotime" collapsed
:::

**Algorithm Description**

Compute pseudotime as normalized shortest path distance from root cell:

1. **Shortest Path Computation**: Run Dijkstra's algorithm from root_cell:
   - Initialize distances: root = 0, others = INF
   - Use 4-ary heap for efficient priority queue
   - Relax edges: update distances to neighbors
   - Continue until all reachable nodes processed

2. **Normalization**: 
   - Find maximum distance from root
   - Normalize: `pseudotime[i] = distance[i] / max_distance`
   - Unreachable cells: set pseudotime = 1.0

3. **Output**: Store normalized pseudotime values:
   - `pseudotime[root_cell] = 0`
   - `pseudotime[i]` in [0, 1] for reachable cells
   - `pseudotime[i] = 1` for unreachable cells

**Edge Cases**

- **Unreachable cells**: Cells not connected to root get pseudotime = 1
- **Isolated root**: If root has no neighbors, all others get pseudotime = 1
- **Disconnected graph**: Each component computed independently
- **Negative edge weights**: Treated as positive (distance = 1)

**Data Guarantees (Preconditions)**

- `pseudotime.len >= adjacency.primary_dim()`
- `root_cell` in valid range [0, adjacency.primary_dim())
- Graph should be connected from root for meaningful results

**Complexity Analysis**

- **Time**: O((V + E) * log_4(V)) for Dijkstra
  - 4-ary heap operations: O(log_4(V)) per node
  - Edge relaxation: O(E)
- **Space**: O(V) auxiliary for heap and distance arrays

**Example**

```cpp
#include "scl/kernel/pseudotime.hpp"

scl::Sparse<Real, true> adjacency = /* ... */;  // Cell neighborhood graph
Index root_cell = /* ... */;  // Starting cell
scl::Array<Real> pseudotime(n_cells);

scl::kernel::pseudotime::graph_pseudotime(
    adjacency,
    root_cell,
    pseudotime
);

// pseudotime[i] contains normalized distance from root
```

---

### diffusion_pseudotime

::: source_code file="scl/kernel/pseudotime.hpp" symbol="diffusion_pseudotime" collapsed
:::

**Algorithm Description**

Compute diffusion pseudotime (DPT) using diffusion map distance from root:

1. **Diffusion Components**: Initialize random components [n x n_dcs]:
   - Random initialization for power iteration
   - Components represent diffusion space

2. **Power Iteration**: For n_iterations:
   - Apply transition: `DC = T * DC`
   - Orthonormalize with modified Gram-Schmidt
   - Extract leading diffusion components

3. **Distance Computation**: 
   - Compute Euclidean distance from root in diffusion component space
   - Distance = sqrt(sum((DC[i, :] - DC[root, :])^2))

4. **Normalization**: 
   - Normalize distances to [0, 1]
   - Store as pseudotime values

**Edge Cases**

- **Small n_dcs**: Fewer components may lose information
- **Large n_dcs**: More components capture more structure but slower
- **Unreachable root**: Handled by transition matrix structure
- **Convergence**: May need more iterations for large graphs

**Data Guarantees (Preconditions)**

- `pseudotime.len >= transition_matrix.primary_dim()`
- `root_cell` in valid range
- Transition matrix should be row-stochastic (rows sum to 1)

**Complexity Analysis**

- **Time**: O(n_iterations * nnz * n_dcs) for power iteration
  - Each iteration: O(nnz * n_dcs) for SpMM
  - Orthonormalization: O(n * n_dcs^2)
  - Distance computation: O(n * n_dcs)
- **Space**: O(n * n_dcs) for diffusion components

**Example**

```cpp
scl::Sparse<Real, true> transition_matrix = /* ... */;  // Markov transition matrix
Index root_cell = /* ... */;
scl::Array<Real> pseudotime(n_cells);

scl::kernel::pseudotime::diffusion_pseudotime(
    transition_matrix,
    root_cell,
    pseudotime,
    config::DEFAULT_N_DCS,        // n_dcs = 10
    config::DEFAULT_N_ITERATIONS  // n_iterations = 100
);

// pseudotime contains DPT values
```

---

### dijkstra_shortest_path

::: source_code file="scl/kernel/pseudotime.hpp" symbol="dijkstra_shortest_path" collapsed
:::

**Algorithm Description**

Compute shortest path distances from a single source node using Dijkstra's algorithm:

1. **Initialization**: 
   - All distances = INF
   - Source distance = 0
   - Initialize 4-ary heap with source

2. **Relaxation Loop**: While heap not empty:
   - Pop minimum distance node from heap
   - For each neighbor:
     - Compute new distance = current + edge_weight
     - If new distance < old distance: update and push to heap

3. **Output**: Store shortest distances:
   - `distances[source] = 0`
   - `distances[i]` = shortest path distance from source to i
   - Unreachable nodes have distance = INF_DISTANCE

**Edge Cases**

- **Unreachable nodes**: Get distance = INF_DISTANCE
- **Negative weights**: Treated as positive (distance = 1)
- **Self-loops**: Handled correctly
- **Multiple paths**: Algorithm finds shortest

**Data Guarantees (Preconditions)**

- `distances.len >= adjacency.primary_dim()`
- `source` in valid range [0, adjacency.primary_dim())
- Edge weights should be positive (negative treated as 1)

**Complexity Analysis**

- **Time**: O((V + E) * log_4(V)) for Dijkstra with 4-ary heap
  - Heap operations: O(log_4(V)) per node
  - Edge relaxation: O(E)
- **Space**: O(V) for heap and distance arrays

**Example**

```cpp
scl::Sparse<Real, true> adjacency = /* ... */;
Index source = 0;
scl::Array<Real> distances(n_nodes);

scl::kernel::pseudotime::dijkstra_shortest_path(
    adjacency,
    source,
    distances
);

// distances[i] contains shortest distance from source to i
```

---

### dijkstra_multi_source

::: source_code file="scl/kernel/pseudotime.hpp" symbol="dijkstra_multi_source" collapsed
:::

**Algorithm Description**

Compute shortest path distances from multiple source nodes in parallel:

1. **Parallel Processing**: Process each source independently:
   - Each thread runs Dijkstra for one source
   - Uses per-thread heap and distance arrays

2. **Per-Source Dijkstra**: For each source:
   - Run standard Dijkstra algorithm
   - Store distances in output matrix row

3. **Output**: Store distance matrix:
   - `distances[s * n + i]` = shortest distance from sources[s] to node i
   - Each source computed independently

**Edge Cases**

- **Same as single-source**: Handles all edge cases per source
- **Parallel efficiency**: Scales with number of threads
- **Memory**: Requires O(n_sources * n) output space

**Data Guarantees (Preconditions)**

- `distances` size >= sources.len * adjacency.primary_dim()
- All source indices in valid range

**Complexity Analysis**

- **Time**: O(n_sources * (V + E) * log_4(V) / n_threads)
  - Parallelized over sources
  - Each source: O((V + E) * log_4(V))
- **Space**: O(V * n_threads) for per-thread heaps

**Example**

```cpp
scl::Sparse<Real, true> adjacency = /* ... */;
scl::Array<const Index> sources = /* ... */;  // Multiple source nodes
Real* distances = /* allocate n_sources * n */;

scl::kernel::pseudotime::dijkstra_multi_source(
    adjacency,
    sources,
    distances
);

// distances[s * n + i] contains distance from sources[s] to i
```

---

### compute_pseudotime

::: source_code file="scl/kernel/pseudotime.hpp" symbol="compute_pseudotime" collapsed
:::

**Algorithm Description**

Generic pseudotime computation with method selection:

1. **Method Dispatch**: Based on PseudotimeMethod:
   - **ShortestPath**: Calls graph_pseudotime
   - **DiffusionPseudotime**: Calls diffusion_pseudotime
   - **GraphDistance**: Alias for ShortestPath
   - **WatershedDescent**: Future implementation

2. **Root Selection**: If root not provided:
   - Uses default selection method
   - Or requires user to provide root

3. **Computation**: Calls appropriate method:
   - Handles method-specific parameters
   - Normalizes output to [0, 1]

4. **Output**: Store computed pseudotime values

**Edge Cases**

- **Method-specific**: Depends on selected method
- **Invalid method**: Returns error or uses default

**Data Guarantees (Preconditions)**

- `pseudotime.len >= adjacency.primary_dim()`
- `root_cell` in valid range
- Method-specific preconditions apply

**Complexity Analysis**

- **Time**: Depends on method
  - ShortestPath: O((V + E) * log_4(V))
  - DiffusionPseudotime: O(n_iterations * nnz * n_dcs)
- **Space**: Method-specific

**Example**

```cpp
scl::Sparse<Real, true> adjacency = /* ... */;
Index root_cell = /* ... */;
scl::Array<Real> pseudotime(n_cells);

scl::kernel::pseudotime::compute_pseudotime(
    adjacency,
    root_cell,
    pseudotime,
    scl::kernel::pseudotime::PseudotimeMethod::DiffusionPseudotime,
    config::DEFAULT_N_DCS  // n_dcs = 10
);
```

---

## Utility Functions

### select_root_cell

Select root cell as the one with minimum marker gene expression.

::: source_code file="scl/kernel/pseudotime.hpp" symbol="select_root_cell" collapsed
:::

**Complexity**

- Time: O(n)
- Space: O(1) auxiliary

---

### select_root_peripheral

Select root cell as the most peripheral node (highest average edge weight).

::: source_code file="scl/kernel/pseudotime.hpp" symbol="select_root_peripheral" collapsed
:::

**Complexity**

- Time: O(nnz)
- Space: O(n) auxiliary

---

### detect_branch_points

Identify branch points in trajectory based on pseudotime topology.

::: source_code file="scl/kernel/pseudotime.hpp" symbol="detect_branch_points" collapsed
:::

**Complexity**

- Time: O(nnz)
- Space: O(n * n_threads) auxiliary

---

### segment_trajectory

Assign cells to trajectory segments based on branch points.

::: source_code file="scl/kernel/pseudotime.hpp" symbol="segment_trajectory" collapsed
:::

**Complexity**

- Time: O(n * n_branch_points)
- Space: O(n_branch_points) auxiliary

---

### smooth_pseudotime

Smooth pseudotime values using neighborhood averaging.

::: source_code file="scl/kernel/pseudotime.hpp" symbol="smooth_pseudotime" collapsed
:::

**Complexity**

- Time: O(n_iterations * nnz)
- Space: O(n) auxiliary

---

### pseudotime_correlation

Compute Pearson correlation between pseudotime and each gene.

::: source_code file="scl/kernel/pseudotime.hpp" symbol="pseudotime_correlation" collapsed
:::

**Complexity**

- Time: O(nnz + n_genes)
- Space: O(n_genes) auxiliary

---

### velocity_weighted_pseudotime

Refine pseudotime using RNA velocity direction information.

::: source_code file="scl/kernel/pseudotime.hpp" symbol="velocity_weighted_pseudotime" collapsed
:::

**Complexity**

- Time: O(n_iterations * nnz)
- Space: O(n) auxiliary

---

### find_terminal_states

Identify terminal (end) states as cells above pseudotime percentile.

::: source_code file="scl/kernel/pseudotime.hpp" symbol="find_terminal_states" collapsed
:::

**Complexity**

- Time: O(n log n)
- Space: O(n) auxiliary

---

### compute_backbone

Select representative backbone cells evenly spaced along pseudotime.

::: source_code file="scl/kernel/pseudotime.hpp" symbol="compute_backbone" collapsed
:::

**Complexity**

- Time: O(n log n)
- Space: O(n) auxiliary

---

## Configuration

Default parameters in `scl::kernel::pseudotime::config`:

- `DEFAULT_N_DCS = 10`: Default number of diffusion components
- `DEFAULT_N_ITERATIONS = 100`: Default power iteration iterations
- `DEFAULT_THRESHOLD = 0.1`: Default neighbor classification threshold
- `DEFAULT_DAMPING = 0.85`: Default damping factor
- `CONVERGENCE_TOL = 1e-6`: Convergence tolerance
- `INF_DISTANCE = 1e30`: Infinity distance value
- `PARALLEL_THRESHOLD = 256`: Minimum size for parallel processing
- `SIMD_THRESHOLD = 16`: Minimum size for SIMD operations
- `HEAP_ARITY = 4`: Heap arity for Dijkstra

---

## Performance Notes

### Method Selection

- **Shortest Path**: Fast, O((V+E)log V), good for simple trajectories
- **Diffusion Pseudotime**: Slower but more robust to noise, captures connectivity structure
- **4-ary Heap**: Faster than binary heap due to better cache locality

### Parallelization

- Multi-source Dijkstra parallelizes over sources
- Diffusion components use parallel SpMM
- Most utility functions parallelize over cells

---

## See Also

- [Graph Algorithms](../components)
- [Neighbors](../neighbors)
- [Velocity](../velocity)
