# transition.hpp

> scl/kernel/transition.hpp · Cell state transition analysis

## Overview

This file provides functions for analyzing cell state transitions using Markov chain theory. It includes transition matrix construction, steady-state analysis, absorption probabilities, and metastable state identification.

**Header**: `#include "scl/kernel/transition.hpp"`

Key features:
- Transition matrix construction from velocity vectors
- Stationary distribution computation
- Absorption probability to terminal states
- Metastable state identification (PCCA+)
- Lineage driver gene identification
- Forward committor probabilities

---

## Main APIs

### transition_matrix_from_velocity

::: source_code file="scl/kernel/transition.hpp" symbol="transition_matrix_from_velocity" collapsed
:::

**Algorithm Description**

Build transition matrix from velocity vectors and kNN graph:

1. For each cell i (parallel):
   - For each neighbor j in kNN graph:
     - Compute direction vector: `direction = expression[j] - expression[i]`
     - Compute cosine similarity: `cos_sim = dot(velocity[i], direction) / (|velocity[i]| * |direction|)`
     - If `trans_type == Forward`: use cos_sim directly
     - If `trans_type == Backward`: use -cos_sim
     - If `trans_type == Symmetric`: use |cos_sim|
   - Apply softmax to neighbor similarities to get transition probabilities
   - Store probabilities in transition matrix row i
2. Matrix is row-stochastic (each row sums to 1.0)

**Edge Cases**

- **No neighbors**: Row sums to 0 (isolated cell)
- **Zero velocity**: All transitions have equal probability
- **All neighbors identical**: Uniform transition probabilities
- **Empty kNN graph**: All transitions are zero

**Data Guarantees (Preconditions)**

- Velocity vectors are row-major: `velocity[i * n_genes + j]` is gene j of cell i
- kNN graph must be valid CSR format
- `transition_out` must be pre-allocated with correct dimensions
- kNN graph should be symmetric or directed as needed

**Complexity Analysis**

- **Time**: O(n_cells * k * n_genes) where k is average neighbors
- **Space**: O(n_genes) per thread for temporary vectors

**Example**

```cpp
#include "scl/kernel/transition.hpp"

const Real* velocity = /* velocity vectors [n_cells * n_genes] */;
scl::Sparse<Real, true> knn = /* kNN graph */;
scl::Sparse<Real, true> transition(n_cells, n_cells);

scl::kernel::transition::transition_matrix_from_velocity(
    velocity, knn, transition, n_cells, n_genes,
    scl::kernel::transition::TransitionType::Forward
);

// transition is now row-stochastic transition matrix
```

---

### stationary_distribution

::: source_code file="scl/kernel/transition.hpp" symbol="stationary_distribution" collapsed
:::

**Algorithm Description**

Compute stationary distribution using power iteration:

1. Initialize: `pi = [1/n, 1/n, ..., 1/n]` (uniform distribution)
2. Iterate until convergence:
   - Compute `pi_new = pi * T` (sparse matrix-vector product with transpose)
   - Apply Aitken delta-squared acceleration every 10 iterations
   - Normalize: `pi = pi_new / sum(pi_new)`
   - Check convergence: `||pi_new - pi|| < tol`
3. Returns stationary distribution where `pi * T = pi`

**Edge Cases**

- **Disconnected graph**: May have multiple stationary distributions
- **Absorbing states**: Stationary distribution concentrates on absorbing states
- **Periodic chain**: May not converge (handled by max_iter)
- **max_iter reached**: Returns current distribution (may not be converged)

**Data Guarantees (Preconditions)**

- Matrix must be row-stochastic (rows sum to 1.0)
- Matrix should be irreducible (single ergodic class) for unique solution
- `pi` must be pre-allocated with size n

**Complexity Analysis**

- **Time**: O(max_iter * nnz) - SpMV per iteration
- **Space**: O(n) auxiliary space

**Example**

```cpp
scl::Array<Real> pi(n_cells);

scl::kernel::transition::stationary_distribution(
    transition, pi.data(), n_cells,
    1e-6,   // tol
    1000    // max_iter
);

// pi[i] contains stationary probability of state i
// pi * T = pi (stationary condition)
```

---

### absorption_probability

::: source_code file="scl/kernel/transition.hpp" symbol="absorption_probability" collapsed
:::

**Algorithm Description**

Compute absorption probabilities to terminal states:

1. Partition transition matrix:
   - Q = transient-to-transient submatrix
   - R = transient-to-terminal submatrix
2. Solve linear system: `(I - Q) * B = R` using SOR iteration
   - B[i, j] = probability of being absorbed by terminal j from transient state i
   - omega = 1.5 for over-relaxation (faster convergence)
3. Returns absorption probability matrix

**Edge Cases**

- **No terminal states**: All probabilities are 0
- **All states terminal**: Identity matrix (immediate absorption)
- **Unreachable terminals**: Probabilities are 0
- **Multiple paths**: Probabilities sum over all paths

**Data Guarantees (Preconditions)**

- `terminal_mask` identifies terminal states (1 = terminal, 0 = transient)
- Matrix must be valid transition matrix
- `absorb_probs` is row-major: `absorb_probs[i * n_terminal + j]` is probability from state i to terminal j

**Complexity Analysis**

- **Time**: O(max_iter * nnz * n_terminal) - SOR iteration
- **Space**: O(n * n_terminal) for output matrix

**Example**

```cpp
uint8_t* terminal_mask = /* terminal state mask */;
scl::Array<Real> absorb_probs(n_cells * n_terminal);

scl::kernel::transition::absorption_probability(
    transition, terminal_mask, absorb_probs.data(),
    n_cells, n_terminal,
    1e-6,   // tol
    1000    // max_iter
);

// absorb_probs[i * n_terminal + j] = probability of being
// absorbed by terminal j from state i
```

---

### metastable_states

::: source_code file="scl/kernel/transition.hpp" symbol="metastable_states" collapsed
:::

**Algorithm Description**

Identify metastable states using spectral clustering (PCCA+):

1. K-means++ initialization on eigenvector space:
   - Select initial centroids based on eigenvector distances
2. Parallel k-means assignment and update:
   - Assign each state to nearest centroid in eigenvector space
   - Update centroids as mean of assigned states
   - Iterate until convergence
3. Compute soft membership:
   - Membership[i, j] = distance-based weight to metastable state j
   - Normalized to sum to 1.0 per state

**Edge Cases**

- **k = 1**: All states assigned to single metastable state
- **k >= n**: Each state is its own metastable state
- **Identical eigenvectors**: May have degenerate clustering

**Data Guarantees (Preconditions)**

- `eigenvectors` is row-major: `eigenvectors[i * n + j]` is component j of eigenvector i
- `assignments` and `membership` must be pre-allocated
- k should be <= n (number of states)

**Complexity Analysis**

- **Time**: O(n * k^2 * n_iter) - k-means iterations
- **Space**: O(k * k) for centroids

**Example**

```cpp
const Real* eigenvectors = /* top k eigenvectors [k * n] */;
scl::Array<Index> assignments(n_cells);
scl::Array<Real> membership(n_cells * k);

scl::kernel::transition::metastable_states(
    eigenvectors, n_cells, k,
    assignments.data(), membership.data(),
    42  // seed
);

// assignments[i] = hard assignment to metastable state
// membership[i * k + j] = soft membership probability
```

---

### lineage_drivers

::: source_code file="scl/kernel/transition.hpp" symbol="lineage_drivers" collapsed
:::

**Algorithm Description**

Identify genes driving lineage transitions:

1. For each gene g and terminal state t (parallel):
   - Extract gene expression: `expr = expression[:, g]`
   - Extract absorption probabilities: `probs = absorb_probs[:, t]`
   - Compute Pearson correlation: `corr = corr(expr, probs)`
   - Store as driver score: `driver_scores[g * n_terminal + t] = corr`
2. Higher scores indicate genes correlated with commitment to terminal state
3. Uses SIMD-optimized dot products for correlation computation

**Edge Cases**

- **Constant expression**: Correlation is 0 (undefined)
- **Constant probabilities**: Correlation is 0 (undefined)
- **Perfect correlation**: Score approaches 1.0

**Data Guarantees (Preconditions)**

- Expression matrix must be valid CSR format
- `absorb_probs` is row-major: `absorb_probs[i * n_terminal + t]`
- `driver_scores` is row-major: `driver_scores[g * n_terminal + t]`

**Complexity Analysis**

- **Time**: O(n_genes * n_terminal * n_cells) - correlation per gene-terminal pair
- **Space**: O(n_cells) per thread

**Example**

```cpp
const Real* absorb_probs = /* absorption probabilities */;
scl::Array<Real> driver_scores(n_genes * n_terminal);

scl::kernel::transition::lineage_drivers(
    expression, absorb_probs, driver_scores.data(),
    n_cells, n_genes, n_terminal
);

// driver_scores[g * n_terminal + t] = correlation between
// gene g expression and commitment to terminal state t
```

---

### forward_committor

::: source_code file="scl/kernel/transition.hpp" symbol="forward_committor" collapsed
:::

**Algorithm Description**

Compute forward committor probabilities between source and target:

1. Solve linear system: `(I - Q) * q = r` where:
   - Q = transition probabilities between intermediate states
   - r = transition probabilities from intermediate to target
   - q = committor probabilities
2. Boundary conditions:
   - `committor[i] = 0` if i is source state
   - `committor[i] = 1` if i is target state
3. Uses iterative solver (SOR) with over-relaxation

**Edge Cases**

- **Source = target**: All committors are 0 (trivial)
- **No path from source to target**: Committors are 0
- **All states are targets**: All committors are 1

**Data Guarantees (Preconditions)**

- `source_mask` and `target_mask` identify source and target states
- Matrix must be valid transition matrix
- `committor` must be pre-allocated with size n

**Complexity Analysis**

- **Time**: O(max_iter * nnz) - iterative solver
- **Space**: O(n) auxiliary space

**Example**

```cpp
uint8_t* source_mask = /* source states */;
uint8_t* target_mask = /* target states */;
scl::Array<Real> committor(n_cells);

scl::kernel::transition::forward_committor(
    transition, source_mask, target_mask, committor.data(),
    n_cells, 1e-6, 1000
);

// committor[i] = probability of hitting target before
// source from state i
```

---

## Utility Functions

### sparse_matvec

Sparse matrix-vector product: y = A * x

::: source_code file="scl/kernel/transition.hpp" symbol="sparse_matvec" collapsed
:::

**Complexity**

- Time: O(nnz)
- Space: O(1)

---

### sparse_matvec_transpose

Sparse matrix-vector product with transpose: y = A^T * x

::: source_code file="scl/kernel/transition.hpp" symbol="sparse_matvec_transpose" collapsed
:::

**Complexity**

- Time: O(nnz)
- Space: O(1)

---

### is_stochastic

Check if matrix is row-stochastic (rows sum to 1).

::: source_code file="scl/kernel/transition.hpp" symbol="is_stochastic" collapsed
:::

**Complexity**

- Time: O(nnz)
- Space: O(1)

---

### row_normalize_to_stochastic

Normalize matrix rows to sum to 1 (make row-stochastic).

::: source_code file="scl/kernel/transition.hpp" symbol="row_normalize_to_stochastic" collapsed
:::

**Complexity**

- Time: O(nnz)
- Space: O(1)

---

### symmetrize_transition

Symmetrize transition matrix: T_sym = 0.5 * (T + T^T)

::: source_code file="scl/kernel/transition.hpp" symbol="symmetrize_transition" collapsed
:::

**Complexity**

- Time: O(nnz)
- Space: O(nnz) for output

---

### identify_terminal_states

Identify terminal (absorbing) states in the Markov chain.

::: source_code file="scl/kernel/transition.hpp" symbol="identify_terminal_states" collapsed
:::

**Complexity**

- Time: O(nnz)
- Space: O(1)

---

### hitting_time

Compute expected hitting time to a target state.

::: source_code file="scl/kernel/transition.hpp" symbol="hitting_time" collapsed
:::

**Complexity**

- Time: O(max_iter * nnz)
- Space: O(n)

---

### time_to_absorption

Compute expected time to absorption for transient states.

::: source_code file="scl/kernel/transition.hpp" symbol="time_to_absorption" collapsed
:::

**Complexity**

- Time: O(max_iter * nnz)
- Space: O(n)

---

### compute_top_eigenvectors

Compute top k eigenvectors using power iteration with deflation.

::: source_code file="scl/kernel/transition.hpp" symbol="compute_top_eigenvectors" collapsed
:::

**Complexity**

- Time: O(k * max_iter * nnz)
- Space: O(k * n)

---

### coarse_grain_transition

Compute coarse-grained transition matrix between metastable states.

::: source_code file="scl/kernel/transition.hpp" symbol="coarse_grain_transition" collapsed
:::

**Complexity**

- Time: O(nnz * k^2)
- Space: O(k^2)

---

### directional_score

Compute directional bias score for each cell.

::: source_code file="scl/kernel/transition.hpp" symbol="directional_score" collapsed
:::

**Complexity**

- Time: O(nnz)
- Space: O(1)

---

## Notes

- Transition matrices should be row-stochastic for proper Markov chain analysis
- Stationary distribution requires irreducible chain (single ergodic class)
- Absorption probabilities are useful for lineage commitment analysis
- Metastable states identify long-lived intermediate states
- Lineage drivers help identify key regulatory genes

## See Also

- [Velocity Module](./velocity) - For velocity vector computation
- [Diffusion Module](./diffusion) - For diffusion-based analysis
- [Pseudotime Module](./pseudotime) - For pseudotime computation
