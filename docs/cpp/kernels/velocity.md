# RNA Velocity

RNA velocity analysis for predicting cell state transitions and fate probabilities.

## Overview

RNA velocity kernels provide:

- **Kinetic Fitting** - Fit gene degradation rates from spliced/unspliced data
- **Velocity Computation** - Compute velocity vectors for each cell
- **Transition Graph** - Build velocity-based transition probabilities
- **Trajectory Analysis** - Latent time, fate probabilities, root/terminal detection
- **Visualization** - Velocity embedding, grid, confidence scores

## Kinetic Fitting

### fit_gene_kinetics

Fit kinetic parameters (gamma) for each gene using spliced/unspliced data:

```cpp
#include "scl/kernel/velocity.hpp"

Sparse<Real, true> spliced = /* ... */;      // Spliced expression matrix
Sparse<Real, true> unspliced = /* ... */;      // Unspliced expression matrix
Index n_cells = spliced.rows();
Index n_genes = spliced.cols();

Array<Real> gamma(n_genes);                  // Pre-allocated output
Array<Real> r2(n_genes);                     // Pre-allocated output

scl::kernel::velocity::fit_gene_kinetics(
    spliced, unspliced,
    n_cells, n_genes,
    gamma, r2,
    model = scl::kernel::velocity::VelocityModel::SteadyState
);

// gamma[g] = estimated degradation rate for gene g
// r2[g] = fit quality (0 to 1)
```

**Parameters:**
- `spliced`: Spliced expression matrix (cells × genes or genes × cells)
- `unspliced`: Unspliced expression matrix
- `n_cells`: Number of cells
- `n_genes`: Number of genes
- `gamma`: Output degradation rates, must be pre-allocated, size = n_genes
- `r2`: Output R-squared fit quality, must be pre-allocated, size = n_genes
- `model`: Velocity model - SteadyState, Dynamical, or Stochastic (default: SteadyState)

**Postconditions:**
- `gamma[g] = estimated degradation rate for gene g`
- `r2[g] = fit quality (0 to 1)`

**Algorithm:**
SteadyState model: Linear regression u = gamma * s
- SIMD-optimized linear regression with 6 accumulators
- CSR: Binary search for gene values across cells
- CSC: Direct column access

**Complexity:**
- Time: O(n_genes * n_cells)
- Space: O(n_cells) per thread for buffers

**Thread Safety:**
- Safe - uses DualWorkspacePool for thread-local buffers

## Velocity Computation

### compute_velocity

Compute RNA velocity for each cell-gene pair:

```cpp
Array<Real> gamma = /* ... */;               // Pre-computed degradation rates [n_genes]
Array<Real> velocity(n_cells * n_genes);     // Pre-allocated output [n_cells x n_genes], row-major

scl::kernel::velocity::compute_velocity(
    spliced, unspliced, gamma,
    n_cells, n_genes,
    velocity.ptr
);

// velocity[c * n_genes + g] = unspliced[c,g] - gamma[g] * spliced[c,g]
```

**Parameters:**
- `spliced`: Spliced expression matrix
- `unspliced`: Unspliced expression matrix
- `gamma`: Pre-computed degradation rates, size = n_genes
- `n_cells`: Number of cells
- `n_genes`: Number of genes
- `velocity_out`: Output velocity matrix, must be pre-allocated, size = n_cells × n_genes (row-major)

**Postconditions:**
- `velocity_out[c,g] = unspliced[c,g] - gamma[g] * spliced[c,g]`
- Positive velocity = gene being upregulated
- Negative velocity = gene being downregulated

**Complexity:**
- Time: O(nnz)
- Space: O(n_cells * n_genes) for output

**Thread Safety:**
- Safe - no race conditions for CSR (row-parallel) or CSC (gene-parallel)

## Transition Graph

### velocity_graph

Build velocity-based transition graph from kNN:

```cpp
const Real* velocity = /* ... */;            // Velocity vectors [n_cells x n_genes]
const Real* expression = /* ... */;          // Expression vectors [n_cells x n_genes]
Sparse<Real, true> knn = /* ... */;          // KNN graph
Index k_neighbors = /* ... */;
Array<Real> transition_probs(n_cells * k_neighbors);  // Pre-allocated output

scl::kernel::velocity::velocity_graph(
    velocity, expression, knn,
    n_cells, n_genes,
    transition_probs.ptr, k_neighbors
);
```

**Algorithm:**
For each cell i (parallel):
1. For each neighbor j: delta = expression[j] - expression[i]
2. Compute cosine similarity between velocity[i] and delta
3. Apply softmax to get transition probabilities

**Complexity:**
- Time: O(n_cells * k * n_genes)
- Space: O(n_genes) per thread

**Thread Safety:**
- Safe - uses WorkspacePool for delta buffers

## Trajectory Analysis

### latent_time

Infer latent time from transition probabilities using shortest paths:

```cpp
const Real* transition_probs = /* ... */;    // Transition probabilities [n_cells x k]
const Index* knn_indices = /* ... */;        // KNN indices [n_cells x k]
Index root_cell = /* ... */;
Array<Real> latent_time(n_cells);            // Pre-allocated output

scl::kernel::velocity::latent_time(
    transition_probs, knn_indices,
    n_cells, k_neighbors,
    root_cell, latent_time.ptr
);

// latent_time[root_cell] = 0
// latent_time normalized to [0, 1]
```

**Algorithm:**
Bellman-Ford shortest path with -log(prob) as edge weights.

**Postconditions:**
- `latent_time[root_cell] = 0`
- `latent_time` normalized to [0, 1]

### cell_fate_probability

Compute probability of reaching each terminal state:

```cpp
Array<Index> terminal_cells = /* ... */;     // Terminal cell indices
Index n_terminal = terminal_cells.len;
Array<Real> fate_probs(n_cells * n_terminal);  // Pre-allocated output

scl::kernel::velocity::cell_fate_probability(
    transition_probs, knn_indices,
    n_cells, k_neighbors,
    terminal_cells,
    fate_probs.ptr
);

// fate_probs[c * n_terminal + t] = probability that cell c reaches terminal t
```

**Postconditions:**
- `fate_probs[c, t] = probability that cell c reaches terminal t`
- Sum over t equals 1 for each cell

## Visualization

### velocity_embedding

Project velocity to low-dimensional embedding space:

```cpp
const Real* embedding = /* ... */;           // Low-dim embedding [n_cells x n_dims]
Index n_dims = /* ... */;
Array<Real> velocity_embedded(n_cells * n_dims);  // Pre-allocated output

scl::kernel::velocity::velocity_embedding(
    velocity, embedding, knn_indices,
    n_cells, n_genes, n_dims, k_neighbors,
    velocity_embedded.ptr
);
```

**Algorithm:**
For each cell: weighted average of neighbor direction vectors in embedding space, weighted by velocity magnitude.

### velocity_confidence

Compute velocity confidence as consistency with neighbors:

```cpp
Array<Real> confidence(n_cells);             // Pre-allocated output

scl::kernel::velocity::velocity_confidence(
    velocity, knn_indices,
    n_cells, n_genes, k_neighbors,
    confidence.ptr
);

// confidence[i] = average cosine similarity with neighbors
// Range: [-1, 1], higher = more consistent
```

## Utilities

### select_velocity_genes

Select genes with reliable velocity estimates:

```cpp
Array<Real> gamma = /* ... */;
Array<Real> r2 = /* ... */;
Array<Index> selected_genes(n_genes);        // Pre-allocated output
Index n_selected;

scl::kernel::velocity::select_velocity_genes(
    velocity, gamma, r2,
    n_cells, n_genes,
    min_r2 = 0.05,                           // Minimum R-squared
    min_velocity_var = 0.1,                  // Minimum velocity variance
    selected_genes, n_selected
);
```

### detect_terminal_states

Detect terminal states based on velocity and outflow:

```cpp
const Real* velocity_magnitude = /* ... */;  // Velocity magnitudes [n_cells]
Array<Index> terminal_cells(n_cells);        // Pre-allocated output

Index n_terminal = scl::kernel::velocity::detect_terminal_states(
    transition_probs, knn_indices,
    velocity_magnitude,
    n_cells, k_neighbors,
    magnitude_threshold = 0.5,
    terminal_cells
);

// Returns number of detected terminal cells
```

---

::: tip SteadyState Model
The SteadyState model assumes u = gamma * s, where u is unspliced, s is spliced, and gamma is the degradation rate. This is valid for genes near steady-state dynamics.
:::

