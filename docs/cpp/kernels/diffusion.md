# diffusion.hpp

> scl/kernel/diffusion.hpp · High-performance diffusion processes on sparse graphs

## Overview

This file provides efficient diffusion operations for trajectory analysis, pseudotime computation, and signal propagation on sparse graphs. All operations use parallelized sparse matrix-vector multiplication (SpMV) and sparse matrix-matrix multiplication (SpMM) for optimal performance.

**Header**: `#include "scl/kernel/diffusion.hpp"`

Key features:
- Vector and matrix diffusion via transition matrices
- Diffusion distance computation
- Pseudotime from root cells
- Random walk with restart (RWR) scoring
- Diffusion map embeddings

---

## Main APIs

### diffuse_vector

::: source_code file="scl/kernel/diffusion.hpp" symbol="diffuse_vector" collapsed
:::

**Algorithm Description**

Apply diffusion operator to a dense vector for multiple steps:

1. For each step from 1 to n_steps:
   - Compute sparse matrix-vector product: `x_new = transition * x`
   - Update x with x_new
2. Uses parallelized SpMV for each step
3. Each step propagates the signal one hop through the transition graph

**Edge Cases**

- **Empty vector**: Returns immediately without changes
- **Zero transition matrix**: Vector remains unchanged
- **n_steps = 0**: Returns immediately, vector unchanged

**Data Guarantees (Preconditions)**

- `x.len >= transition.primary_dim()`
- Transition matrix must be row-stochastic (each row sums to 1.0)
- Transition matrix must be valid CSR format
- Indices must be sorted within rows

**Complexity Analysis**

- **Time**: O(n_steps * nnz) where nnz is number of non-zeros in transition matrix
- **Space**: O(n_nodes) auxiliary space for temporary vector

**Example**

```cpp
#include "scl/kernel/diffusion.hpp"

// Create transition matrix (row-stochastic)
scl::Sparse<Real, true> transition = /* ... */;

// Initialize signal vector
scl::Array<Real> x(n_nodes);
// ... initialize x ...

// Apply 3 diffusion steps
scl::kernel::diffusion::diffuse_vector(transition, x, 3);

// x now contains diffused signal after 3 steps
```

---

### diffuse_matrix

::: source_code file="scl/kernel/diffusion.hpp" symbol="diffuse_matrix" collapsed
:::

**Algorithm Description**

Apply diffusion operator to a dense matrix (multiple features simultaneously):

1. For each step from 1 to n_steps:
   - For each feature column:
     - Compute sparse matrix-vector product with feature column
     - Update feature column in-place
2. Uses blocked sparse matrix-matrix multiplication (SpMM) for efficiency
3. All features are diffused in parallel

**Edge Cases**

- **Empty matrix**: Returns immediately
- **n_features = 0**: Returns immediately
- **n_steps = 0**: Matrix unchanged

**Data Guarantees (Preconditions)**

- `X.len >= n_nodes * n_features`
- Matrix X is row-major layout: `X[i * n_features + j]` is feature j of node i
- Transition matrix must be valid CSR format

**Complexity Analysis**

- **Time**: O(n_steps * nnz * n_features)
- **Space**: O(n_nodes * n_features) auxiliary space

**Example**

```cpp
// Feature matrix: n_nodes rows, n_features columns
scl::Array<Real> X(n_nodes * n_features);
// ... initialize X ...

// Diffuse all features for 3 steps
scl::kernel::diffusion::diffuse_matrix(
    transition, X, n_nodes, n_features, 3
);

// All features in X are now diffused
```

---

### diffusion_distance

::: source_code file="scl/kernel/diffusion.hpp" symbol="diffusion_distance" collapsed
:::

**Algorithm Description**

Compute diffusion distance matrix between all pairs of nodes:

1. For each source node i:
   - Initialize unit vector at node i
   - Apply n_steps diffusion steps
   - Store resulting distribution in row i of distance matrix
2. Distance between nodes i and j is computed from diffused distributions
3. Uses parallelization over source nodes

**Edge Cases**

- **n_nodes = 0**: Returns empty distance matrix
- **n_steps = 0**: Returns identity-like distances
- **Isolated nodes**: Distance remains high

**Data Guarantees (Preconditions)**

- `distances` has capacity >= n_nodes * n_nodes
- Output matrix is row-major: `distances[i * n_nodes + j]` is distance from i to j
- Transition matrix must be valid CSR format

**Complexity Analysis**

- **Time**: O(n_nodes^2 * n_steps * nnz)
- **Space**: O(n_nodes^2) auxiliary space

**Example**

```cpp
scl::Array<Real> distances(n_nodes * n_nodes);

scl::kernel::diffusion::diffusion_distance(
    transition, distances, 3
);

// distances[i * n_nodes + j] contains diffusion distance
// from node i to node j after 3 steps
```

---

### diffusion_pseudotime

::: source_code file="scl/kernel/diffusion.hpp" symbol="diffusion_pseudotime" collapsed
:::

**Algorithm Description**

Compute diffusion pseudotime from root cells:

1. Initialize pseudotime to infinity for all nodes
2. For each root cell:
   - Initialize unit mass at root
   - Iteratively diffuse until convergence:
     - Apply transition matrix
     - Update pseudotime as minimum distance from any root
   - Continue until convergence or max_iter
3. Pseudotime represents minimum diffusion distance from nearest root

**Edge Cases**

- **No root cells (n_roots = 0)**: All pseudotime values remain infinity
- **Isolated nodes**: Pseudotime remains infinity
- **max_iter reached**: Returns current pseudotime values (may not be converged)

**Data Guarantees (Preconditions)**

- `pseudotime.len >= transition.primary_dim()`
- `root_cells` contains valid node indices in [0, n_nodes)
- All root cell indices are distinct
- Transition matrix must be valid CSR format

**Complexity Analysis**

- **Time**: O(n_roots * max_iter * nnz)
- **Space**: O(n_nodes * n_roots) auxiliary space

**Example**

```cpp
// Define root cells (e.g., early developmental stage)
scl::Array<const Index> root_cells = {0, 5, 10};
Index n_roots = 3;

scl::Array<Real> pseudotime(n_nodes);

scl::kernel::diffusion::diffusion_pseudotime(
    transition, root_cells, n_roots, pseudotime, 100
);

// pseudotime[i] contains pseudotime from nearest root cell
```

---

### random_walk_with_restart

::: source_code file="scl/kernel/diffusion.hpp" symbol="random_walk_with_restart" collapsed
:::

**Algorithm Description**

Compute random walk with restart (RWR) scores:

1. Initialize scores: uniform distribution over seed nodes
2. Iterate until convergence:
   - With probability alpha: restart at seed nodes
   - With probability (1-alpha): take one step via transition matrix
   - Update scores: `scores = alpha * seed_distribution + (1-alpha) * transition * scores`
3. Continue until change < tol or max_iter reached
4. Scores represent steady-state probability of being at each node

**Edge Cases**

- **No seed nodes (n_seeds = 0)**: Scores remain zero
- **alpha = 1.0**: Scores remain at initial seed distribution
- **alpha = 0.0**: Pure random walk (may not converge)
- **Isolated seed nodes**: High scores only at seeds

**Data Guarantees (Preconditions)**

- `scores.len >= transition.primary_dim()`
- `seed_nodes` contains valid node indices
- `alpha` in (0, 1] for proper convergence
- Transition matrix must be row-stochastic

**Complexity Analysis**

- **Time**: O(max_iter * nnz)
- **Space**: O(n_nodes) auxiliary space

**Example**

```cpp
// Seed nodes (e.g., marker cells)
scl::Array<const Index> seed_nodes = {100, 200, 300};
Index n_seeds = 3;

scl::Array<Real> scores(n_nodes);

scl::kernel::diffusion::random_walk_with_restart(
    transition, seed_nodes, scores, 0.85, 100, 1e-6
);

// scores[i] contains RWR score (probability) for node i
// Higher scores indicate closer proximity to seed nodes
```

---

### diffusion_map

::: source_code file="scl/kernel/diffusion.hpp" symbol="diffusion_map" collapsed
:::

**Algorithm Description**

Compute diffusion map embedding using eigendecomposition:

1. For each component from 1 to n_components:
   - Use power method to find dominant eigenvector
   - Orthogonalize against previous components
   - Store as embedding column
2. Embedding captures diffusion geometry of the graph
3. Uses iterative power method with reorthogonalization

**Edge Cases**

- **n_components = 0**: Returns empty embedding
- **Symmetric transition matrix**: Real eigenvalues guaranteed
- **Non-symmetric matrix**: May have complex eigenvalues (handled)

**Data Guarantees (Preconditions)**

- `embedding` has capacity >= n_nodes * n_components
- Embedding is row-major: `embedding[i * n_components + j]` is component j of node i
- Transition matrix must be valid CSR format
- n_components <= n_nodes

**Complexity Analysis**

- **Time**: O(n_components * max_iter * nnz)
- **Space**: O(n_nodes * n_components) auxiliary space

**Example**

```cpp
Index n_components = 10;
scl::Array<Real> embedding(n_nodes * n_components);

scl::kernel::diffusion::diffusion_map(
    transition, embedding, n_nodes, n_components, 100
);

// embedding[i * n_components + j] contains j-th diffusion
// map coordinate for node i
```

---

## Configuration

Default parameters are defined in `scl::kernel::diffusion::config`:

- `DEFAULT_N_STEPS = 3`: Default number of diffusion steps
- `DEFAULT_ALPHA = 0.85`: Default restart probability for RWR
- `CONVERGENCE_TOL = 1e-6`: Convergence tolerance
- `MAX_ITER = 100`: Maximum iterations for iterative methods

---

## Notes

- Transition matrices should be row-stochastic (normalized rows) for proper diffusion behavior
- CSR format is required for optimal SpMV performance
- All operations are thread-safe and automatically parallelized
- Matrix diffusion uses blocked SpMM for cache efficiency

## See Also

- [Sparse Matrix Operations](../core/sparse)
- [Neighbors Module](./neighbors) - For constructing transition matrices
