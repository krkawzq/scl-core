# louvain.hpp

> scl/kernel/louvain.hpp · Multi-level Louvain community detection algorithm

## Overview

High-performance implementation of the Louvain algorithm for community detection in graphs. The Louvain method is a greedy optimization algorithm that maximizes modularity to identify communities in large networks.

This file provides:
- Multi-level hierarchical community detection
- Modularity optimization with resolution parameter
- Parallel processing for large graphs
- Utility functions for community analysis

**Header**: `#include "scl/kernel/louvain.hpp"`

---

## Main APIs

### cluster

::: source_code file="scl/kernel/louvain.hpp" symbol="cluster" collapsed
:::

**Algorithm Description**

Multi-level Louvain algorithm for community detection:

1. **Initialization**: Each node is initially assigned to its own community
2. **Local Moving Phase**:
   - For each node, compute modularity gain of moving to each neighbor's community
   - Move node to the community that yields the maximum positive modularity gain
   - Repeat until no improvement can be made
3. **Aggregation Phase**:
   - Build a coarsened graph where nodes represent communities from the previous level
   - Edge weights are the sum of inter-community edges
   - Normalize edge weights appropriately
4. **Iteration**: Repeat steps 2-3 on the coarsened graph until convergence or max_iter is reached

The algorithm uses parallel processing with thread-local workspaces to handle large graphs efficiently.

**Edge Cases**

- **Empty graph**: Returns with all nodes labeled as 0 (single community)
- **Disconnected graph**: Each connected component forms separate communities
- **Single node**: Node is assigned to community 0
- **Max iterations reached**: Algorithm stops and returns current partitioning

**Data Guarantees (Preconditions)**

- Adjacency matrix must be valid sparse matrix (CSR or CSC format)
- Adjacency should represent an undirected graph (symmetric matrix preferred)
- Labels array length must be >= adjacency.primary_dim()
- Resolution parameter must be > 0
- Max iterations must be > 0

**Complexity Analysis**

- **Time**: O(n * log(n) * avg_degree) expected for sparse graphs, where n is the number of nodes. The logarithmic factor comes from the multi-level hierarchy, and avg_degree is the average node degree.
- **Space**: O(n + nnz) for working memory, including thread-local workspaces and intermediate data structures

**Example**

```cpp
#include "scl/kernel/louvain.hpp"

// Create adjacency matrix (CSR format)
Sparse<Real, true> adjacency = /* ... */;  // n_nodes x n_nodes sparse matrix
Array<Index> labels(adjacency.rows());

// Standard clustering with default resolution (1.0)
scl::kernel::louvain::cluster(adjacency, labels);

// Clustering with higher resolution (yields more, smaller communities)
scl::kernel::louvain::cluster(adjacency, labels, resolution = 1.5);

// Clustering with custom iteration limit
scl::kernel::louvain::cluster(adjacency, labels, resolution = 1.0, max_iter = 200);

// Community IDs are now in labels[i], ranging from 0 to n_communities-1
```

---

### compute_modularity

::: source_code file="scl/kernel/louvain.hpp" symbol="compute_modularity" collapsed
:::

**Algorithm Description**

Computes the modularity score Q for a given graph and clustering:

Q = (1/2m) * Σᵢⱼ [Aᵢⱼ - resolution * (kᵢ * kⱼ) / (2m)] * δ(cᵢ, cⱼ)

Where:
- Aᵢⱼ = edge weight between nodes i and j
- m = total edge weight / 2 (half-sum of all edge weights)
- kᵢ = weighted degree of node i
- cᵢ = community assignment of node i
- δ(x, y) = 1 if x == y, 0 otherwise (Kronecker delta)
- resolution = resolution parameter (default 1.0)

The algorithm:
1. Computes total edge weight m and node degrees kᵢ in parallel
2. For each edge, checks if endpoints are in the same community
3. Accumulates the modularity contributions

**Edge Cases**

- **Single community**: Returns Q = 0 (no structure)
- **Each node alone**: Returns negative Q (poor structure)
- **Perfect communities**: Returns Q close to 1.0
- **Empty graph**: Returns Q = 0

**Data Guarantees (Preconditions)**

- Adjacency matrix must be valid sparse matrix
- Labels array length must be >= adjacency.primary_dim()
- All labels[i] must be >= 0
- Resolution parameter should be > 0 (default: 1.0)

**Complexity Analysis**

- **Time**: O(n + nnz) where n is number of nodes and nnz is number of non-zero edges. Parallel processing is used for degree computation.
- **Space**: O(n) auxiliary space for storing node degrees and community totals

**Example**

```cpp
#include "scl/kernel/louvain.hpp"

Sparse<Real, true> adjacency = /* ... */;
Array<Index> labels = /* cluster assignments */;

// Compute modularity with default resolution
Real modularity = scl::kernel::louvain::compute_modularity(adjacency, labels);

// Compute with custom resolution
Real mod_res15 = scl::kernel::louvain::compute_modularity(adjacency, labels, resolution = 1.5);

// Modularity ranges from -0.5 to 1.0
// Q > 0 indicates community structure stronger than random
// Q close to 1.0 indicates very strong community structure
```

---

### community_sizes

::: source_code file="scl/kernel/louvain.hpp" symbol="community_sizes" collapsed
:::

**Algorithm Description**

Counts the number of nodes in each community:

1. Initialize sizes array to zero
2. Iterate through all labels and increment sizes[labels[i]] for each node i
3. Count number of communities as max(labels) + 1

**Edge Cases**

- **Empty labels array**: Returns n_communities = 0, sizes unchanged
- **All nodes in one community**: Returns n_communities = 1, sizes[0] = n_nodes
- **Each node in separate community**: Returns n_communities = n_nodes, each sizes[i] = 1

**Data Guarantees (Preconditions)**

- All labels[i] must be >= 0
- Sizes array length must be >= max(labels) + 1

**Complexity Analysis**

- **Time**: O(n) where n is the length of labels array. Single pass through labels.
- **Space**: O(1) auxiliary space (output sizes array is provided by caller)

**Example**

```cpp
#include "scl/kernel/louvain.hpp"

Array<Index> labels = /* cluster labels from cluster() */;
Index max_label = /* ... */;  // Determine maximum label value
Array<Index> sizes(max_label + 1);
Index n_communities;

scl::kernel::louvain::community_sizes(labels, sizes, n_communities);

// sizes[c] now contains number of nodes in community c
// n_communities contains total number of communities
```

---

### get_community_members

::: source_code file="scl/kernel/louvain.hpp" symbol="get_community_members" collapsed
:::

**Algorithm Description**

Extracts the indices of all nodes belonging to a specific community:

1. Iterate through all labels
2. For each node i where labels[i] == community, add i to members array
3. Count total members and store in n_members

**Edge Cases**

- **Community doesn't exist**: Returns n_members = 0
- **All nodes in target community**: Returns n_members = n_nodes
- **Empty community**: Returns n_members = 0
- **Members array too small**: Only first members.len indices are stored

**Data Guarantees (Preconditions)**

- Community ID must be >= 0
- Members array should be large enough to hold all members (safe to pass large buffer)

**Complexity Analysis**

- **Time**: O(n) where n is the length of labels array. Single pass through labels.
- **Space**: O(1) auxiliary space (output members array is provided by caller)

**Example**

```cpp
#include "scl/kernel/louvain.hpp"

Array<Index> labels = /* cluster labels */;
Index target_community = 5;
Array<Index> members(labels.len);  // Allocate large enough buffer
Index n_members;

scl::kernel::louvain::get_community_members(
    labels, target_community, members, n_members
);

// members[0..n_members-1] now contains indices of nodes in community 5
```

---

## Configuration

The namespace `scl::kernel::louvain::config` provides configuration constants:

- `DEFAULT_RESOLUTION = 1.0`: Default resolution parameter
- `DEFAULT_MAX_ITER = 100`: Default maximum iterations
- `MODULARITY_EPSILON = 1e-8`: Convergence threshold for modularity changes
- `PARALLEL_THRESHOLD = 1000`: Minimum graph size for parallel processing
- `MAX_LEVELS = 100`: Maximum depth of multi-level hierarchy

## Notes

- The Louvain algorithm may produce different results on different runs due to the greedy nature of node ordering. For reproducible results, consider seeding the random number generator if used internally.
- Higher resolution values (e.g., 1.5, 2.0) produce more communities with fewer nodes each.
- Lower resolution values (e.g., 0.5) produce fewer communities with more nodes each.
- The algorithm is optimized for sparse graphs. For dense graphs, consider alternative algorithms.

## See Also

- [Leiden Clustering](./leiden) - Alternative community detection algorithm with guaranteed quality improvement
- [Metrics](./metrics) - Quality metrics for evaluating clustering results
