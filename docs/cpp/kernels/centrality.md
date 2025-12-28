# centrality.hpp

> scl/kernel/centrality.hpp · High-performance graph centrality measures for network analysis

## Overview

This file provides comprehensive graph centrality measures for analyzing node importance in networks. Centrality measures quantify how "central" or important a node is within a graph structure, with different measures capturing different aspects of importance.

Key features:
- Multiple centrality algorithms (degree, PageRank, betweenness, etc.)
- Iterative methods with convergence checking
- Parallelized implementations
- Support for weighted and unweighted graphs

**Header**: `#include "scl/kernel/centrality.hpp"`

---

## Main APIs

### pagerank

::: source_code file="scl/kernel/centrality.hpp" symbol="pagerank" collapsed
:::

**Algorithm Description**

Compute PageRank centrality using power iteration:

1. **Initialization**: Set all scores to 1/n_nodes
2. **Iteration**: For each iteration until convergence:
   - Compute new scores: `scores_new = (1-damping) * teleport + damping * A^T * scores_old`
   - Teleport vector: uniform distribution (1/n_nodes for each node)
   - Matrix-vector multiplication: `A^T * scores` using sparse matrix multiplication
   - Convergence check: `||scores_new - scores_old|| < tolerance`
3. **Normalization**: Scores sum to 1.0

The algorithm models a random surfer who:
- With probability `damping`: follows a random outgoing edge
- With probability `1-damping`: teleports to a random node

**Edge Cases**

- **Dangling nodes** (no outgoing edges): Treated as having edges to all nodes (teleportation)
- **Disconnected graph**: Each component converges independently
- **Convergence failure**: Returns after max_iter iterations (may not be fully converged)
- **Zero adjacency matrix**: All scores equal to 1/n_nodes

**Data Guarantees (Preconditions)**

- Matrix must be valid CSR or CSC format
- `scores.len >= adjacency.primary_dim()`
- `damping` in range (0, 1)
- `max_iter > 0`
- `tol > 0`

**Complexity Analysis**

- **Time**: O(max_iter * nnz) where nnz is number of non-zeros
  - Each iteration: O(nnz) for sparse matrix-vector multiplication
  - Typically converges in O(log(n)) iterations
- **Space**: O(n_nodes) auxiliary for storing old and new scores

**Example**

```cpp
#include "scl/kernel/centrality.hpp"

Sparse<Real, true> adjacency = /* ... */;  // Graph adjacency matrix
Array<Real> scores(n_nodes);

scl::kernel::centrality::pagerank(
    adjacency,
    scores,
    config::DEFAULT_DAMPING,      // damping = 0.85
    config::DEFAULT_MAX_ITER,     // max_iter = 100
    config::DEFAULT_TOLERANCE     // tol = 1e-6
);

// scores[i] now contains PageRank score for node i
// Scores sum to 1.0
```

---

### betweenness_centrality

::: source_code file="scl/kernel/centrality.hpp" symbol="betweenness_centrality" collapsed
:::

**Algorithm Description**

Compute betweenness centrality using Brandes algorithm:

1. **For each source node s** (parallelized):
   - **BFS**: Compute shortest paths from s to all reachable nodes
   - **Dependency accumulation**: For each node v:
     - `betweenness[v] += sum(dependency[s][v])` where dependency is computed from shortest path counts
   - **Dependency calculation**: `dependency[s][v] = sum((sigma[s][v] / sigma[s][w]) * (1 + dependency[s][w]))` for predecessors w of v
2. **Normalization**: If normalize=true, divide by (n-1)*(n-2)/2

Betweenness measures the fraction of shortest paths that pass through a node, identifying "bridge" nodes that connect different parts of the network.

**Edge Cases**

- **Isolated nodes**: Have betweenness = 0
- **Disconnected graph**: Computed per component
- **Unweighted graph**: Uses BFS (O(nnz) per source)
- **Weighted graph**: Uses Dijkstra (O(nnz * log(n)) per source)

**Data Guarantees (Preconditions)**

- Matrix must be valid CSR or CSC format
- `centrality.len >= adjacency.primary_dim()`
- Graph should be connected (or component-wise computation)

**Complexity Analysis**

- **Time**: O(n_nodes * nnz) for unweighted graphs
  - For each source: O(nnz) BFS
  - Parallelized over sources
- **Space**: O(n_nodes) auxiliary per thread for BFS queue and dependencies

**Example**

```cpp
Sparse<Real, true> adjacency = /* ... */;
Array<Real> betweenness(n_nodes);

scl::kernel::centrality::betweenness_centrality(
    adjacency,
    betweenness,
    true  // normalize = true
);

// betweenness[i] contains fraction of shortest paths passing through node i
// Normalized values in [0, 1]
```

---

### degree_centrality

::: source_code file="scl/kernel/centrality.hpp" symbol="degree_centrality" collapsed
:::

**Algorithm Description**

Compute degree centrality (sum of edge weights):

1. **Parallel processing**: For each node i in parallel
2. **Degree computation**: `centrality[i] = sum(adjacency[i, :])` (sum of all edge weights)
3. **Normalization**: If normalize=true, divide by maximum degree

Degree centrality is the simplest centrality measure, counting the number (or sum of weights) of connections a node has.

**Edge Cases**

- **Isolated nodes**: Have degree = 0
- **Empty graph**: All degrees are 0
- **Self-loops**: Counted in degree

**Data Guarantees (Preconditions)**

- Matrix must be valid CSR or CSC format
- `centrality.len >= adjacency.primary_dim()`

**Complexity Analysis**

- **Time**: O(nnz) - single pass through all edges
- **Space**: O(1) auxiliary

**Example**

```cpp
Sparse<Real, true> adjacency = /* ... */;
Array<Real> degree(n_nodes);

scl::kernel::centrality::degree_centrality(
    adjacency,
    degree,
    true  // normalize = true
);

// degree[i] contains normalized degree of node i
```

---

### personalized_pagerank

::: source_code file="scl/kernel/centrality.hpp" symbol="personalized_pagerank" collapsed
:::

**Algorithm Description**

Compute personalized PageRank with custom teleportation vector:

1. **Initialization**: Set scores to personalization vector
2. **Iteration**: Same as standard PageRank, but teleport vector is `personalization` instead of uniform
3. **Convergence**: Check convergence and normalize

Personalized PageRank allows biasing the random walk toward specific nodes, useful for seed-based importance or local importance around specific regions.

**Edge Cases**

- **Personalization doesn't sum to 1.0**: Normalized internally
- **Zero personalization**: Falls back to uniform teleportation
- **Sparse personalization**: Only non-zero nodes receive teleportation probability

**Data Guarantees (Preconditions)**

- `personalization.len == adjacency.primary_dim()`
- Personalization should sum to 1.0 (will be normalized if not)
- `scores.len >= adjacency.primary_dim()`

**Complexity Analysis**

- **Time**: O(max_iter * nnz) - same as standard PageRank
- **Space**: O(n_nodes) auxiliary

**Example**

```cpp
Array<Real> personalization(n_nodes, 0.0);
personalization[seed_node] = 1.0;  // Bias toward seed node

Array<Real> scores(n_nodes);
scl::kernel::centrality::personalized_pagerank(
    adjacency,
    personalization,
    scores
);

// Scores now reflect importance relative to seed node
```

---

### eigenvector_centrality

::: source_code file="scl/kernel/centrality.hpp" symbol="eigenvector_centrality" collapsed
:::

**Algorithm Description**

Compute eigenvector centrality (dominant eigenvector):

1. **Power iteration**: Iteratively compute `centrality = A * centrality`
2. **Normalization**: After each iteration, L2-normalize the vector
3. **Convergence**: Check when `||centrality_new - centrality_old|| < tolerance`

Eigenvector centrality measures long-term influence: a node is important if it is connected to other important nodes.

**Edge Cases**

- **Multiple dominant eigenvalues**: May converge to different eigenvectors
- **Disconnected graph**: Each component has its own dominant eigenvector
- **Zero matrix**: All scores equal

**Data Guarantees (Preconditions)**

- Matrix must be valid CSR or CSC format
- `centrality.len >= adjacency.primary_dim()`
- Graph should be strongly connected for unique solution

**Complexity Analysis**

- **Time**: O(max_iter * nnz)
- **Space**: O(n_nodes) auxiliary

**Example**

```cpp
Array<Real> centrality(n_nodes);
scl::kernel::centrality::eigenvector_centrality(
    adjacency,
    centrality
);

// centrality contains dominant eigenvector (L2-normalized)
```

---

## Utility Functions

### hits

Compute HITS (Hyperlink-Induced Topic Search) hub and authority scores.

::: source_code file="scl/kernel/centrality.hpp" symbol="hits" collapsed
:::

**Complexity**

- Time: O(max_iter * nnz)
- Space: O(n_nodes) auxiliary

---

### katz_centrality

Compute Katz centrality: `centrality = alpha * A * centrality + beta`.

::: source_code file="scl/kernel/centrality.hpp" symbol="katz_centrality" collapsed
:::

**Complexity**

- Time: O(max_iter * nnz)
- Space: O(n_nodes) auxiliary

---

### closeness_centrality

Compute closeness centrality (inverse of average shortest path length).

::: source_code file="scl/kernel/centrality.hpp" symbol="closeness_centrality" collapsed
:::

**Complexity**

- Time: O(n_nodes * nnz)
- Space: O(n_nodes) auxiliary per thread

---

### approximate_betweenness

Compute approximate betweenness using random sampling.

::: source_code file="scl/kernel/centrality.hpp" symbol="approximate_betweenness" collapsed
:::

**Complexity**

- Time: O(n_samples * nnz)
- Space: O(n_nodes) auxiliary per thread

---

### harmonic_centrality

Compute harmonic centrality (sum of inverse distances).

::: source_code file="scl/kernel/centrality.hpp" symbol="harmonic_centrality" collapsed
:::

**Complexity**

- Time: O(n_nodes * nnz)
- Space: O(n_nodes) auxiliary per thread

---

### random_walk_centrality

Compute centrality based on random walk visit frequencies.

::: source_code file="scl/kernel/centrality.hpp" symbol="random_walk_centrality" collapsed
:::

**Complexity**

- Time: O(n_walks * walk_length)
- Space: O(n_nodes) auxiliary per thread

---

## Notes

**Configuration Constants**

Default parameters in `scl::kernel::centrality::config`:
- `DEFAULT_DAMPING = 0.85`
- `DEFAULT_MAX_ITER = 100`
- `DEFAULT_TOLERANCE = 1e-6`

**Performance Considerations**

- Use `approximate_betweenness` for large networks (n_nodes > 10000)
- Iterative methods typically converge in O(log(n)) iterations
- All functions are parallelized for better performance

## See Also

- [Graph Algorithms](/cpp/kernels/graph) - Other graph algorithms
- [Statistics](/cpp/kernels/statistics) - Statistical analysis
