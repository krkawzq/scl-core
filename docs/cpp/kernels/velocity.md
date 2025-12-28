# velocity.hpp

> scl/kernel/velocity.hpp · RNA velocity analysis for predicting cell state transitions

## Overview

This file provides comprehensive RNA velocity analysis for single-cell transcriptomics data:

- **Kinetic Fitting**: Fit gene degradation rates from spliced/unspliced data
- **Velocity Computation**: Compute velocity vectors for each cell-gene pair
- **Transition Graph**: Build velocity-based transition probabilities from kNN graphs
- **Trajectory Analysis**: Latent time inference, fate probabilities, root/terminal detection
- **Visualization**: Velocity embedding, grid computation, confidence scores

RNA velocity predicts future cell states by modeling the balance between spliced and unspliced mRNA.

**Header**: `#include "scl/kernel/velocity.hpp"`

---

## Main APIs

### fit_gene_kinetics

::: source_code file="scl/kernel/velocity.hpp" symbol="fit_gene_kinetics" collapsed
:::

**Algorithm Description**

Fit kinetic parameters (gamma, degradation rate) for each gene using spliced/unspliced data:

1. **SteadyState Model** (default): Linear regression u = gamma * s
   - For each gene in parallel:
     - Extract spliced (s) and unspliced (u) values across all cells
     - Use SIMD-optimized linear regression with 6 accumulators
     - Compute gamma = sum(u * s) / sum(s^2) using least squares
     - Compute R-squared: R² = 1 - SS_res / SS_tot
   - CSR format: Binary search for gene values across cells
   - CSC format: Direct column access for efficient gene-wise processing

2. **Dynamical Model**: Time-dependent model (not yet implemented)

3. **Stochastic Model**: Stochastic differential equations (not yet implemented)

The SteadyState model assumes genes are near steady-state where unspliced mRNA production equals degradation.

**Edge Cases**

- **Zero spliced values**: Gamma set to 0, R² = 0
- **Constant spliced values**: Regression undefined, gamma = 0, R² = 0
- **Few cells (< 3)**: Regression unreliable, R² may be low
- **Sparse data**: Handled correctly by sparse matrix operations

**Data Guarantees (Preconditions)**

- `gamma.len >= n_genes`
- `r2.len >= n_genes`
- Spliced and unspliced matrices have same dimensions
- Matrices must be valid sparse format (CSR or CSC)

**Complexity Analysis**

- **Time**: O(n_genes * n_cells) parallelized over genes
- **Space**: O(n_cells) per thread for temporary buffers (uses DualWorkspacePool)

**Example**

```cpp
#include "scl/kernel/velocity.hpp"

// Spliced and unspliced expression matrices (cells x genes)
scl::Sparse<Real, true> spliced = /* ... */;
scl::Sparse<Real, true> unspliced = /* ... */;
scl::Index n_cells = spliced.rows();
scl::Index n_genes = spliced.cols();

// Pre-allocate output buffers
scl::Array<Real> gamma(n_genes);
scl::Array<Real> r2(n_genes);

// Fit kinetic parameters using SteadyState model
scl::kernel::velocity::fit_gene_kinetics(
    spliced, unspliced,
    n_cells, n_genes,
    gamma, r2,
    scl::kernel::velocity::VelocityModel::SteadyState
);

// Filter genes by fit quality
for (scl::Index g = 0; g < n_genes; ++g) {
    if (r2[g] > 0.05 && gamma[g] > 0) {
        // Use gene g for velocity analysis
    }
}
```

---

### compute_velocity

::: source_code file="scl/kernel/velocity.hpp" symbol="compute_velocity" collapsed
:::

**Algorithm Description**

Compute RNA velocity for each cell-gene pair using the formula: v = u - gamma * s

1. **CSR format** (cells as rows): Parallel over cells
   - For each cell c in parallel:
     - For each gene g with non-zero expression:
       - velocity[c,g] = unspliced[c,g] - gamma[g] * spliced[c,g]
   - Direct row access for efficient cell-wise processing

2. **CSC format** (genes as rows): Parallel over genes
   - For each gene g in parallel:
     - For each cell c with non-zero expression:
       - velocity[c,g] = unspliced[c,g] - gamma[g] * spliced[c,g]
   - Direct column access for efficient gene-wise processing

Positive velocity indicates gene upregulation (unspliced > gamma * spliced), negative indicates downregulation.

**Edge Cases**

- **Zero spliced/unspliced**: Velocity = 0 (implicit zeros in sparse format)
- **Negative gamma**: Results in positive velocity even for low unspliced values
- **Missing genes**: Velocity = 0 for genes not in gamma array

**Data Guarantees (Preconditions)**

- `gamma.len >= n_genes`
- `velocity_out` has capacity >= `n_cells * n_genes` (row-major layout)
- Spliced and unspliced matrices have same dimensions
- Matrices must be valid sparse format

**Complexity Analysis**

- **Time**: O(nnz) where nnz is total non-zeros in both matrices
- **Space**: O(n_cells * n_genes) for output matrix

**Example**

```cpp
#include "scl/kernel/velocity.hpp"

// Pre-computed degradation rates
scl::Array<Real> gamma = /* ... */;  // [n_genes]

// Pre-allocate velocity matrix (row-major: cells x genes)
scl::Array<Real> velocity(n_cells * n_genes);

scl::kernel::velocity::compute_velocity(
    spliced, unspliced, gamma,
    n_cells, n_genes,
    velocity.ptr
);

// Access velocity: velocity[c * n_genes + g] = velocity for cell c, gene g
// Positive = upregulation, negative = downregulation
```

---

### velocity_graph

::: source_code file="scl/kernel/velocity.hpp" symbol="velocity_graph" collapsed
:::

**Algorithm Description**

Build velocity-based transition graph from kNN graph:

1. For each cell i in parallel:
   - For each neighbor j in kNN graph:
     - Compute direction vector: delta = expression[j] - expression[i]
     - Compute cosine similarity: sim = dot(velocity[i], delta) / (||velocity[i]|| * ||delta||)
     - Store similarity score
   - Apply softmax over neighbors: prob[i,j] = exp(sim[j]) / sum_k(exp(sim[k]))
   - Transition probabilities sum to 1 for each cell

The transition probability reflects how well the velocity vector aligns with the direction to the neighbor, indicating likely state transitions.

**Edge Cases**

- **Zero velocity**: All transition probabilities equal (1/k)
- **Zero expression delta**: Similarity = 0, low transition probability
- **Isolated cells**: No neighbors, transition probabilities undefined
- **Orthogonal velocity and delta**: Similarity = 0, low probability

**Data Guarantees (Preconditions)**

- `velocity` has size `n_cells * n_genes` (row-major)
- `expression` has size `n_cells * n_genes` (row-major)
- `knn` is valid sparse kNN graph
- `transition_probs` has capacity >= `n_cells * k_neighbors`
- kNN graph has exactly k_neighbors per cell

**Complexity Analysis**

- **Time**: O(n_cells * k_neighbors * n_genes) parallelized over cells
- **Space**: O(n_genes) per thread for delta buffers (uses WorkspacePool)

**Example**

```cpp
#include "scl/kernel/velocity.hpp"

// Velocity and expression vectors (row-major: cells x genes)
const Real* velocity = /* ... */;    // [n_cells * n_genes]
const Real* expression = /* ... */;  // [n_cells * n_genes]

// KNN graph (sparse, CSR format)
scl::Sparse<Real, true> knn = /* ... */;
scl::Index k_neighbors = 15;

// Pre-allocate transition probabilities
scl::Array<Real> transition_probs(n_cells * k_neighbors);

scl::kernel::velocity::velocity_graph(
    velocity, expression, knn,
    n_cells, n_genes,
    transition_probs.ptr, k_neighbors
);

// transition_probs[c * k_neighbors + j] = probability of transitioning
// from cell c to its j-th neighbor
```

---

### latent_time

::: source_code file="scl/kernel/velocity.hpp" symbol="latent_time" collapsed
:::

**Algorithm Description**

Infer latent time from transition probabilities using shortest path algorithm:

1. **Graph construction**: Build directed graph from transition probabilities
   - Edge weight from cell i to neighbor j: w[i,j] = -log(transition_probs[i,j])
   - Higher probability = lower edge weight (shorter path)
2. **Shortest path computation**: Bellman-Ford algorithm
   - Initialize: latent_time[root_cell] = 0, others = infinity
   - For each iteration: relax all edges
   - Continue until convergence or maximum iterations
3. **Normalization**: Scale to [0, 1] range
   - min_time = 0 (root cell)
   - max_time = maximum latent time
   - Normalize: latent_time[i] = (latent_time[i] - min_time) / (max_time - min_time)

Latent time represents the inferred developmental time along the trajectory.

**Edge Cases**

- **Disconnected components**: Cells unreachable from root have infinite time (clamped to 1.0)
- **Zero transition probabilities**: Edge weight = infinity, path blocked
- **Cyclic graphs**: Bellman-Ford handles cycles correctly
- **Single cell**: Latent time = 0

**Data Guarantees (Preconditions)**

- `transition_probs` has size `n_cells * k_neighbors`
- `knn_indices` has size `n_cells * k_neighbors`
- `root_cell` is valid index in [0, n_cells)
- `latent_time_out` has capacity >= `n_cells`

**Complexity Analysis**

- **Time**: O(n_cells * k_neighbors * iterations) where iterations <= n_cells
- **Space**: O(n_cells) for distance array

**Example**

```cpp
#include "scl/kernel/velocity.hpp"

const Real* transition_probs = /* ... */;  // [n_cells * k_neighbors]
const Index* knn_indices = /* ... */;      // [n_cells * k_neighbors]
scl::Index root_cell = /* ... */;         // Starting cell index

scl::Array<Real> latent_time(n_cells);

scl::kernel::velocity::latent_time(
    transition_probs, knn_indices,
    n_cells, k_neighbors,
    root_cell, latent_time.ptr
);

// latent_time[i] in [0, 1], with latent_time[root_cell] = 0
```

---

### cell_fate_probability

::: source_code file="scl/kernel/velocity.hpp" symbol="cell_fate_probability" collapsed
:::

**Algorithm Description**

Compute probability of each cell reaching each terminal state using absorbing Markov chain:

1. **Markov chain setup**: Transition probabilities form transition matrix
   - Terminal cells are absorbing states (self-transition probability = 1.0)
2. **Absorption probability computation**: Solve linear system
   - For each non-terminal cell i and terminal state t:
     - fate_probs[i,t] = sum over paths from i to t
     - Uses iterative method or matrix inversion
3. **Normalization**: Ensure probabilities sum to 1 for each cell
   - Sum over all terminal states = 1.0

**Edge Cases**

- **Unreachable terminals**: Probability = 0
- **Multiple paths**: Probabilities sum correctly
- **No terminal cells**: All probabilities = 0
- **Isolated cells**: Cannot reach terminals, probabilities = 0

**Data Guarantees (Preconditions)**

- `transition_probs` has size `n_cells * k_neighbors`
- `knn_indices` has size `n_cells * k_neighbors`
- `terminal_cells` contains valid cell indices
- `fate_probs` has capacity >= `n_cells * n_terminal`

**Complexity Analysis**

- **Time**: O(n_cells^2 * n_terminal) for matrix operations
- **Space**: O(n_cells * n_terminal) for probability matrix

**Example**

```cpp
#include "scl/kernel/velocity.hpp"

// Terminal cell indices (e.g., differentiated cell types)
scl::Array<Index> terminal_cells = /* ... */;
scl::Index n_terminal = terminal_cells.len;

scl::Array<Real> fate_probs(n_cells * n_terminal);

scl::kernel::velocity::cell_fate_probability(
    transition_probs, knn_indices,
    n_cells, k_neighbors,
    terminal_cells,
    fate_probs.ptr
);

// fate_probs[c * n_terminal + t] = probability that cell c reaches terminal t
// Sum over t equals 1.0 for each cell c
```

---

## Utility Functions

### splice_ratio

Compute unspliced/spliced ratio for each cell-gene pair.

::: source_code file="scl/kernel/velocity.hpp" symbol="splice_ratio" collapsed
:::

**Complexity**

- Time: O(nnz)
- Space: O(n_cells * n_genes) for output

---

### velocity_graph_cosine

Build velocity graph using cosine similarity between velocity vectors.

::: source_code file="scl/kernel/velocity.hpp" symbol="velocity_graph_cosine" collapsed
:::

**Complexity**

- Time: O(n_cells * k_neighbors * n_genes)
- Space: O(1) auxiliary

---

### velocity_embedding

Project velocity to low-dimensional embedding space.

::: source_code file="scl/kernel/velocity.hpp" symbol="velocity_embedding" collapsed
:::

**Complexity**

- Time: O(n_cells * k_neighbors * n_dims)
- Space: O(n_cells * n_dims) for output

---

### velocity_grid

Compute velocity on a regular grid for visualization.

::: source_code file="scl/kernel/velocity.hpp" symbol="velocity_grid" collapsed
:::

**Complexity**

- Time: O(n_cells + grid_size^2)
- Space: O(grid_size^2) for output

---

### velocity_confidence

Compute velocity confidence as consistency with neighbors.

::: source_code file="scl/kernel/velocity.hpp" symbol="velocity_confidence" collapsed
:::

**Complexity**

- Time: O(n_cells * k_neighbors * n_genes)
- Space: O(1) auxiliary

---

### select_velocity_genes

Select genes with reliable velocity estimates.

::: source_code file="scl/kernel/velocity.hpp" symbol="select_velocity_genes" collapsed
:::

**Complexity**

- Time: O(n_genes * n_cells)
- Space: O(n_genes) for output

---

### velocity_pseudotime

Compute velocity-informed pseudotime.

::: source_code file="scl/kernel/velocity.hpp" symbol="velocity_pseudotime" collapsed
:::

**Complexity**

- Time: O(n_cells * k_neighbors)
- Space: O(n_cells) for output

---

### velocity_divergence

Compute velocity divergence in embedding space.

::: source_code file="scl/kernel/velocity.hpp" symbol="velocity_divergence" collapsed
:::

**Complexity**

- Time: O(n_cells * k_neighbors * n_dims)
- Space: O(1) auxiliary

---

### select_root_by_velocity

Select root cell as the one with minimum incoming flow.

::: source_code file="scl/kernel/velocity.hpp" symbol="select_root_by_velocity" collapsed
:::

**Complexity**

- Time: O(n_cells * k_neighbors)
- Space: O(1) auxiliary

---

### detect_terminal_states

Detect terminal states based on velocity and outflow.

::: source_code file="scl/kernel/velocity.hpp" symbol="detect_terminal_states" collapsed
:::

**Complexity**

- Time: O(n_cells * k_neighbors)
- Space: O(n_cells) for output

---

## Notes

- The SteadyState model assumes genes are near steady-state dynamics (u = gamma * s)
- Velocity vectors predict future cell states based on current spliced/unspliced balance
- Transition probabilities are normalized (sum to 1) for each cell
- Latent time provides a continuous ordering of cells along trajectories
- Fate probabilities enable prediction of cell differentiation outcomes
- All operations are parallelized and thread-safe

## See Also

- [Neighbor Graph Construction](./neighbors)
- [Trajectory Analysis](./pseudotime)
- [Sparse Matrix Operations](../core/sparse)
