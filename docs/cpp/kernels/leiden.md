# leiden.hpp

> scl/kernel/leiden.hpp · High-performance Leiden clustering for community detection

## Overview

This file provides the Leiden algorithm for community detection in graphs. The Leiden algorithm is an improvement over the Louvain algorithm, guaranteeing well-connected communities through a refinement step.

**Header**: `#include "scl/kernel/leiden.hpp"`

---

## Main APIs

### cluster

::: source_code file="scl/kernel/leiden.hpp" symbol="cluster" collapsed
:::

**Algorithm Description**

Perform Leiden clustering on adjacency graph using multi-level optimization:

1. **Local Moving Phase**:
   - For each node, compute modularity gain of moving to neighboring communities
   - Move node to community with highest positive gain
   - Repeat until no positive moves possible

2. **Refinement Phase**:
   - Merge nodes within communities to improve connectivity
   - Ensures communities are well-connected (guarantee of Leiden algorithm)

3. **Aggregation Phase**:
   - Create new graph where nodes are communities from previous level
   - Edge weights are sum of edges between communities

4. **Iteration**:
   - Repeat local moving, refinement, and aggregation
   - Continue until convergence or max_iter reached

**Edge Cases**

- **Empty graph**: Returns single community for all nodes
- **Disconnected graph**: Each connected component forms separate communities
- **Zero resolution**: All nodes in single community
- **Very high resolution**: Each node becomes its own community
- **Isolated nodes**: Assigned to their own communities

**Data Guarantees (Preconditions)**

- `adjacency` must be valid CSR or CSC sparse matrix
- `labels` must have capacity >= adjacency.primary_dim()
- Matrix should represent undirected graph (symmetric adjacency)
- Self-loops are allowed but typically removed for community detection

**Complexity Analysis**

- **Time**: O(max_iter * nnz * log(n_nodes)) - each iteration processes all edges, logarithmic factor from refinement
- **Space**: O(n_nodes) auxiliary - stores community assignments, node degrees, and temporary data structures

**Example**

```cpp
#include "scl/kernel/leiden.hpp"

// Create or load adjacency matrix
Sparse<Real, true> adjacency = /* ... */;  // CSR format
Index n_nodes = adjacency.rows();

// Pre-allocate output labels
Array<Index> labels(n_nodes);

// Perform Leiden clustering
scl::kernel::leiden::cluster(
    adjacency, labels,
    resolution = 1.0,    // Higher = more communities
    max_iter = 10,       // Maximum iterations
    seed = 42            // Random seed for reproducibility
);

// labels[i] contains community ID for node i
// Communities are well-connected (guaranteed by Leiden algorithm)

// Analyze results
std::map<Index, Size> community_sizes;
for (Index i = 0; i < n_nodes; ++i) {
    community_sizes[labels[i]]++;
}

std::cout << "Found " << community_sizes.size() << " communities\n";
for (const auto& [comm_id, size] : community_sizes) {
    std::cout << "Community " << comm_id << ": " << size << " nodes\n";
}
```

---

### modularity

::: source_code file="scl/kernel/leiden.hpp" symbol="modularity" collapsed
:::

**Algorithm Description**

Compute modularity Q of a partition, measuring quality of community structure:

1. Compute total edge weight m = sum of all edge weights
2. For each community c:
   - Compute sum of edge weights within community (e_c)
   - Compute sum of node degrees in community (a_c)
3. Modularity Q = sum_c (e_c/m - (a_c/(2*m))^2) - resolution * sum_c (a_c/(2*m))^2

The resolution parameter controls the trade-off between number and size of communities.

**Edge Cases**

- **Empty partition**: Returns 0.0
- **Single community**: Returns negative value (all nodes in one community)
- **Each node separate**: Returns negative value (no within-community edges)
- **Zero resolution**: Standard modularity without resolution parameter

**Data Guarantees (Preconditions)**

- `adjacency` must be valid CSR or CSC sparse matrix
- `labels` must have length >= adjacency.primary_dim()
- Labels should be valid community IDs (non-negative integers)

**Complexity Analysis**

- **Time**: O(nnz) - iterate over all edges to compute within-community weights
- **Space**: O(n_nodes) auxiliary - store community degree sums

**Example**

```cpp
#include "scl/kernel/leiden.hpp"

// Perform clustering
Sparse<Real, true> adjacency = /* ... */;
Array<Index> labels(n_nodes);
scl::kernel::leiden::cluster(adjacency, labels, resolution = 1.0);

// Compute modularity
Real q = scl::kernel::leiden::modularity(
    adjacency, labels,
    resolution = 1.0
);

std::cout << "Modularity: " << q << "\n";
// Higher values indicate better community structure
// Typical range: [-1, 1], with values > 0.3 considered good

// Compare different resolutions
for (Real res = 0.5; res <= 2.0; res += 0.5) {
    scl::kernel::leiden::cluster(adjacency, labels, resolution = res);
    Real q = scl::kernel::leiden::modularity(adjacency, labels, resolution = res);
    std::cout << "Resolution " << res << ": Q = " << q << "\n";
}
```

---

## Configuration

### Default Parameters

```cpp
namespace scl::kernel::leiden::config {
    constexpr Real DEFAULT_RESOLUTION = Real(1.0);
    constexpr Index DEFAULT_MAX_ITER = 10;
    constexpr Index DEFAULT_MAX_MOVES = 100;
    constexpr Real MODULARITY_EPSILON = Real(1e-10);
    constexpr Real THETA = Real(0.05);
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size HASH_LOAD_FACTOR_INV = 2;
    constexpr Size PREFETCH_DISTANCE = 4;
    constexpr Size SIMD_THRESHOLD = 16;
    constexpr Index MIN_COMMUNITY_SIZE = 1;
    constexpr Real AGGREGATION_THRESHOLD = 0.8;
}
```

---

## Notes

**Resolution Parameter**: Controls number of communities. Higher resolution (e.g., 2.0) creates more, smaller communities. Lower resolution (e.g., 0.5) creates fewer, larger communities. Default 1.0 is a good starting point.

**Leiden vs. Louvain**: Leiden algorithm guarantees well-connected communities through refinement step, while Louvain may produce disconnected communities. Leiden is generally preferred for biological networks.

**Convergence**: Algorithm typically converges in 5-10 iterations. If not converging, increase max_iter or check graph connectivity.

**Thread Safety**: Uses atomic operations for parallel updates, safe for concurrent execution.

---

## See Also

- [Louvain](/cpp/kernels/louvain) - Louvain clustering algorithm
- [Neighbors](/cpp/kernels/neighbors) - K-nearest neighbors for graph construction
