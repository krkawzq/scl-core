# Transition Matrix Analysis

Cell state transition analysis for trajectory inference and fate mapping.

## Overview

Transition matrix kernels provide:

- **Transition Matrix Construction** - Build from velocity vectors and kNN graph
- **Markov Chain Analysis** - Stationary distribution, terminal states, absorption probabilities
- **Trajectory Analysis** - Hitting times, committor probabilities, directional scores
- **Spectral Analysis** - Top eigenvectors, metastable states, coarse-graining
- **Lineage Analysis** - Lineage drivers, fate probabilities

## Transition Matrix Construction

### transition_matrix_from_velocity

Build transition matrix from velocity vectors and kNN graph:

```cpp
#include "scl/kernel/transition.hpp"

const Real* velocity = /* ... */;            // Velocity vectors [n_cells x n_genes]
Sparse<Real, true> knn = /* ... */;          // KNN graph [n_cells x n_cells]
Index n_cells = /* ... */;
Index n_genes = /* ... */;

Sparse<Real, true> transition(n_cells, n_cells);

scl::kernel::transition::transition_matrix_from_velocity(
    velocity, knn, transition,
    n_cells, n_genes,
    trans_type = scl::kernel::transition::TransitionType::Forward
);
```

**Parameters:**
- `velocity`: Velocity vectors, size = n_cells × n_genes
- `knn`: KNN graph (sparse, n_cells × n_cells)
- `transition_out`: Output transition matrix
- `n_cells`: Number of cells
- `n_genes`: Number of genes
- `trans_type`: Type of transition - Forward, Backward, or Symmetric (default: Forward)

**Algorithm:**
For each cell i (parallel):
1. For each neighbor j: compute cosine similarity between velocity[i] and (expression[j] - expression[i])
2. Apply softmax to get transition probabilities
3. Store in transition matrix

**Complexity:**
- Time: O(n_cells * k * n_genes)
- Space: O(n_genes) per thread

**Thread Safety:**
- Safe - parallelized over cells

### row_normalize_to_stochastic

Normalize matrix rows to sum to 1 (make row-stochastic):

```cpp
Sparse<Real, true> matrix = /* ... */;
Index n = matrix.rows();

scl::kernel::transition::row_normalize_to_stochastic(matrix, n);

// Each row now sums to 1.0 (or 0.0 if row was all zeros)
```

**MUTABILITY:**
- INPLACE - modifies matrix values

**Postconditions:**
- Each row sums to 1.0 (or 0.0 if row was all zeros)

## Markov Chain Analysis

### stationary_distribution

Compute stationary distribution using power iteration:

```cpp
Sparse<Real, true> matrix = /* ... */;       // Row-stochastic transition matrix
Array<Real> pi(n);                           // Pre-allocated output

scl::kernel::transition::stationary_distribution(
    matrix, pi.ptr, n,
    tol = 1e-6,                              // Convergence tolerance
    max_iter = 1000                          // Maximum iterations
);

// pi * T = pi (stationary condition)
```

**Parameters:**
- `matrix`: Row-stochastic transition matrix
- `pi`: Output stationary distribution, must be pre-allocated, size = n
- `n`: Dimension
- `tol`: Convergence tolerance (default: 1e-6)
- `max_iter`: Maximum iterations (default: 1000)

**PRECONDITIONS:**
- Matrix is row-stochastic
- Matrix is irreducible (single ergodic class)

**Postconditions:**
- pi * T = pi (stationary condition)
- sum(pi) = 1

**Algorithm:**
Power iteration with Aitken delta-squared acceleration:
1. Initialize pi uniformly
2. Iterate: pi_new = pi * T (SpMV transpose)
3. Apply Aitken acceleration every 10 iterations
4. Normalize and check convergence

**Complexity:**
- Time: O(max_iter * nnz)
- Space: O(n)

### identify_terminal_states

Identify terminal (absorbing) states in the Markov chain:

```cpp
Sparse<Real, true> matrix = /* ... */;
Array<uint8_t> terminal_mask(n);             // Pre-allocated output

scl::kernel::transition::identify_terminal_states(
    matrix, terminal_mask.ptr, n,
    self_loop_thresh = 0.9                   // Self-loop threshold
);

// terminal_mask[i] = 1 if state i is terminal (high self-loop)
```

**Parameters:**
- `matrix`: Transition matrix
- `terminal_mask`: Output mask, must be pre-allocated, size = n (1 = terminal, 0 = transient)
- `n`: Dimension
- `self_loop_thresh`: Self-loop threshold for terminal detection (default: 0.9)

**Postconditions:**
- `terminal_mask[i] = 1` if state i is terminal (high self-loop)

### absorption_probability

Compute absorption probabilities to terminal states:

```cpp
Array<uint8_t> terminal_mask = /* ... */;    // Terminal state mask [n]
Index n_terminal = /* ... */;
Array<Real> absorb_probs(n * n_terminal);    // Pre-allocated output

scl::kernel::transition::absorption_probability(
    matrix, terminal_mask.ptr,
    absorb_probs.ptr, n, n_terminal,
    tol = 1e-6, max_iter = 1000
);

// absorb_probs[i * n_terminal + t] = probability of reaching terminal t from state i
```

**Algorithm:**
Solve (I - Q) * B = R using SOR iteration:
- Q = submatrix of transient-to-transient transitions
- R = submatrix of transient-to-terminal transitions
- omega = 1.5 for over-relaxation (faster convergence)

**Complexity:**
- Time: O(max_iter * nnz * n_terminal)
- Space: O(n * n_terminal)

## Trajectory Analysis

### hitting_time

Compute expected hitting time to a target state:

```cpp
Index target_state = 5;                      // Target state index
Array<Real> hitting_time(n);                 // Pre-allocated output

scl::kernel::transition::hitting_time(
    matrix, target_state, hitting_time.ptr, n,
    tol = 1e-6, max_iter = 1000
);

// hitting_time[i] = expected steps to reach target from i
```

**Postconditions:**
- `hitting_time[target_state] = 0`
- `hitting_time[i] = expected steps to reach target from i`
- Unreachable states have hitting_time = INF

### forward_committor

Compute forward committor probabilities between source and target:

```cpp
Array<uint8_t> source_mask = /* ... */;      // Source states mask [n]
Array<uint8_t> target_mask = /* ... */;      // Target states mask [n]
Array<Real> committor(n);                    // Pre-allocated output

scl::kernel::transition::forward_committor(
    matrix, source_mask.ptr, target_mask.ptr,
    committor.ptr, n,
    tol = 1e-6, max_iter = 1000
);

// committor[i] = probability of hitting target before source from i
```

**Postconditions:**
- `committor[i] = probability of hitting target before source from i`
- committor = 0 for source states, 1 for target states

## Spectral Analysis

### compute_top_eigenvectors

Compute top k eigenvectors using power iteration with deflation:

```cpp
Index k = 10;                                // Number of eigenvectors
Array<Real> eigenvectors(k * n);             // Pre-allocated output [k x n], row-major
Array<Real> eigenvalues(k);                  // Pre-allocated output

scl::kernel::transition::compute_top_eigenvectors(
    matrix, eigenvectors.ptr, eigenvalues.ptr,
    n, k,
    tol = 1e-6, max_iter = 1000
);
```

**Algorithm:**
For each eigenvector:
1. Power iteration with Aitken acceleration
2. Gram-Schmidt orthogonalization against previous
3. Extract eigenvalue and store

**Complexity:**
- Time: O(k * max_iter * nnz)
- Space: O(k * n)

### metastable_states

Identify metastable states using spectral clustering (PCCA+):

```cpp
const Real* eigenvectors = /* ... */;        // Top k eigenvectors [k x n]
Index k = /* ... */;                         // Number of metastable states
Array<Index> assignments(n);                 // Pre-allocated output
Array<Real> membership(n * k);               // Pre-allocated output [n x k]

scl::kernel::transition::metastable_states(
    eigenvectors, n, k,
    assignments.ptr, membership.ptr,
    seed = 42
);
```

**Algorithm:**
1. K-means++ initialization on eigenvector space
2. Parallel k-means assignment and update
3. Compute soft membership via distance-based weights

**Complexity:**
- Time: O(n * k^2 * n_iter)
- Space: O(k * k) for centroids

---

::: tip Transition Type
Forward transitions follow velocity direction (future states). Backward transitions reverse velocity. Symmetric transitions average both directions for undirected analysis.
:::

