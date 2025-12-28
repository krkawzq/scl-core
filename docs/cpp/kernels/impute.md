# impute.hpp

> scl/kernel/impute.hpp · High-performance imputation kernels for single-cell expression data

## Overview

This file provides efficient imputation methods for sparse single-cell RNA-seq data. Imputation fills in missing values (dropouts) to recover true expression signals and improve downstream analysis quality.

This file provides:
- K-nearest neighbor (KNN) imputation
- Diffusion-based imputation (MAGIC-style)
- ALRA (Adaptively-thresholded Low-Rank Approximation)
- Distance-weighted KNN imputation
- Dropout detection and quality assessment

**Header**: `#include "scl/kernel/impute.hpp"`

---

## Main APIs

### knn_impute_dense

::: source_code file="scl/kernel/impute.hpp" symbol="knn_impute_dense" collapsed
:::

**Algorithm Description**

Impute missing values using K-nearest neighbor averaging on dense output:

1. **Affinity-weighted averaging**: For each cell i in parallel:
   - Compute weighted sum: X_imputed[i, j] = sum_k(affinity[i, k] * X_sparse[k, j])
   - Normalize by row sum: X_imputed[i, j] /= sum_k(affinity[i, k])
   - Uses sparse matrix operations for efficiency

2. **Edge case handling**:
   - If row sum < epsilon: copy original row (no neighbors)
   - Handles missing values in sparse input gracefully

3. **Output format**: Dense matrix suitable for downstream dense operations

**Edge Cases**

- **No neighbors**: Cells with zero affinity row sum copy original values
- **Empty affinity matrix**: Returns original sparse matrix converted to dense
- **All zeros**: Cells with no expression remain zero after imputation

**Data Guarantees (Preconditions)**

- `X_sparse` must be CSR format (cells x genes)
- `affinity` must be row-normalized (rows sum to 1)
- `X_imputed` must be pre-allocated with n_cells * n_genes elements
- Affinity matrix should be pre-computed (e.g., from KNN graph)

**Complexity Analysis**

- **Time**: O(n_cells * avg_neighbors * n_genes)
  - O(n_cells * avg_neighbors) for sparse matrix access
  - O(n_cells * avg_neighbors * n_genes) for weighted averaging
- **Space**: O(n_cells * n_genes) for dense output

**Example**

```cpp
#include "scl/kernel/impute.hpp"

// Sparse expression matrix: cells x genes
Sparse<Real, true> X_sparse = /* ... */;
Index n_cells = X_sparse.rows();
Index n_genes = X_sparse.cols();

// Pre-computed affinity matrix (e.g., from KNN)
Sparse<Real, true> affinity = /* ... */;
// ... ensure affinity is row-normalized ...

// Pre-allocate dense output
Array<Real> X_imputed(n_cells * n_genes);

// Impute missing values
scl::kernel::impute::knn_impute_dense(
    X_sparse,
    affinity,
    n_cells,
    n_genes,
    X_imputed.data()
);

// X_imputed now contains dense imputed expression
```

---

### magic_impute

::: source_code file="scl/kernel/impute.hpp" symbol="magic_impute" collapsed
:::

**Algorithm Description**

MAGIC (Markov Affinity-based Graph Imputation of Cells) algorithm for diffusion-based imputation:

1. **Diffusion process**: Apply t steps of diffusion:
   - X_new = T * X_old (sparse matrix multiplication)
   - T is the transition matrix (symmetric normalized)
   - Each step smooths expression across the cell graph

2. **Double buffering**: Use two buffers to avoid data races:
   - Buffer A: current state
   - Buffer B: next state
   - Swap after each step

3. **Convergence**: Higher t = more smoothing/imputation
   - Typical t in [1, 5] for single-cell data
   - t=1: minimal smoothing
   - t=5: strong smoothing, may over-impute

**Edge Cases**

- **t=0**: Returns original matrix (no diffusion)
- **Isolated cells**: Cells with no neighbors remain unchanged
- **Zero transition matrix**: Returns original matrix

**Data Guarantees (Preconditions)**

- `transition_matrix` from MAGIC preprocessing (symmetric normalized)
- `t >= 1`, typically t in [1, 5]
- `X_imputed` must be pre-allocated

**Complexity Analysis**

- **Time**: O(t * n_cells * avg_nnz * n_genes)
  - O(n_cells * avg_nnz * n_genes) per diffusion step
  - t steps total
- **Space**: O(2 * n_cells * n_genes) for double buffering

**Example**

```cpp
// Pre-compute MAGIC transition matrix
Sparse<Real, true> transition_matrix = /* ... */;

Array<Real> X_imputed(n_cells * n_genes);

// Apply MAGIC imputation with t=3
scl::kernel::impute::magic_impute(
    X_sparse,
    transition_matrix,
    n_cells,
    n_genes,
    3,  // diffusion time
    X_imputed.data()
);
```

---

### alra_impute

::: source_code file="scl/kernel/impute.hpp" symbol="alra_impute" collapsed
:::

**Algorithm Description**

ALRA (Adaptively-thresholded Low-Rank Approximation) imputation using randomized SVD:

1. **Randomized SVD**: Compute rank-k approximation:
   - Random Gaussian projection for efficiency
   - Power iteration for numerical stability
   - QR orthogonalization
   - X_approx = U * S * V^T

2. **Thresholding**: Set negative values to zero:
   - Biological constraint: expression cannot be negative
   - X_imputed = max(0, X_approx)

3. **Preserve originals**: Where imputed < original non-zero, keep original:
   - Prevents over-imputation of true zeros

**Edge Cases**

- **k > min(n_cells, n_genes)**: Clamped to minimum dimension
- **Zero variance genes**: Genes with zero variance excluded from SVD
- **All zeros**: Returns zero matrix if input is all zeros

**Data Guarantees (Preconditions)**

- `X_dense` already log-normalized
- `n_components <= min(n_cells, n_genes)`
- `X_imputed` must be pre-allocated

**Complexity Analysis**

- **Time**: O(n_iter * n_cells * n_genes * n_components)
  - O(n_cells * n_genes * n_components) per power iteration
  - n_iter iterations (default: 5)
- **Space**: O(n_cells * n_components + n_genes * n_components) for SVD factors

**Example**

```cpp
// Dense normalized expression (log-normalized)
Array<Real> X_dense(n_cells * n_genes);
// ... fill X_dense ...

Array<Real> X_imputed(n_cells * n_genes);

// ALRA imputation with 50 components
scl::kernel::impute::alra_impute(
    X_dense.data(),
    n_cells,
    n_genes,
    50,   // n_components
    X_imputed.data(),
    5,    // n_iter
    42    // seed
);
```

---

### diffusion_impute_sparse_transition

::: source_code file="scl/kernel/impute.hpp" symbol="diffusion_impute_sparse_transition" collapsed
:::

**Algorithm Description**

Diffusion-based imputation using sparse transition matrix:

1. **Convert to dense**: Initialize dense buffer from sparse input
2. **Iterative diffusion**: For t = 1 to n_steps:
   - SpMM: buffer_out = T * buffer_in (parallel)
   - Swap buffers (double buffering)
3. **Output**: Copy final result to X_imputed

**Edge Cases**

- **n_steps=0**: Returns original matrix
- **Zero transition**: Returns original matrix

**Data Guarantees (Preconditions)**

- `transition_matrix` must be row-stochastic (rows sum to 1)
- `n_steps >= 1`
- `X_imputed` must be pre-allocated

**Complexity Analysis**

- **Time**: O(n_steps * n_cells * avg_nnz_per_row * n_genes)
- **Space**: O(2 * n_cells * n_genes) for double buffering

---

### knn_impute_weighted_dense

::: source_code file="scl/kernel/impute.hpp" symbol="knn_impute_weighted_dense" collapsed
:::

**Algorithm Description**

Impute with distance-weighted KNN contributions:

1. **Weight computation**: For each cell i:
   - weights[k] = 1 / (dist[k] + epsilon) for k neighbors
   - Normalize weights to sum to 1

2. **Weighted averaging**: X_imputed[i, j] = sum_k(weight[k] * X[neighbor_k, j])

**Edge Cases**

- **Zero distances**: Handled with epsilon to avoid division by zero
- **k=0**: Returns original values

**Data Guarantees (Preconditions)**

- `knn_indices` and `knn_distances` pre-computed for all cells
- `k <= n_cells - 1`
- `X_imputed` must be pre-allocated

**Complexity Analysis**

- **Time**: O(n_cells * k * n_genes)
- **Space**: O(n_cells * n_genes)

---

## Utility Functions

### impute_selected_genes

Impute only a subset of genes for efficiency.

::: source_code file="scl/kernel/impute.hpp" symbol="impute_selected_genes" collapsed
:::

**Complexity**

- Time: O(n_cells * avg_neighbors * n_selected)
- Space: O(n_cells * n_selected)

---

### detect_dropouts

Detect likely dropout events (technical zeros vs biological zeros).

::: source_code file="scl/kernel/impute.hpp" symbol="detect_dropouts" collapsed
:::

**Complexity**

- Time: O(n_cells * n_genes)
- Space: O(n_genes)

---

### imputation_quality

Compute imputation quality metrics (correlation with held-out data).

::: source_code file="scl/kernel/impute.hpp" symbol="imputation_quality" collapsed
:::

**Complexity**

- Time: O(n_cells * n_genes)
- Space: O(n_threads)

---

### smooth_expression

Smooth expression profiles using local averaging.

::: source_code file="scl/kernel/impute.hpp" symbol="smooth_expression" collapsed
:::

**Complexity**

- Time: O(n_cells * avg_neighbors * n_genes)
- Space: O(n_cells * n_genes)

---

## Notes

**Method Selection**:
- **KNN**: Fast, simple, good for most cases
- **MAGIC**: Strong smoothing, preserves structure, may over-impute
- **ALRA**: Low-rank approximation, good for denoising, preserves global structure
- **Weighted KNN**: Accounts for distance, more accurate than uniform KNN

**Performance**:
- All methods parallelized over cells
- Sparse input preserved where possible
- Dense output for downstream analysis

**Typical Workflow**:
1. Pre-compute affinity/transition matrix from KNN graph
2. Choose imputation method based on data characteristics
3. Impute missing values
4. Assess quality with imputation_quality if ground truth available

## See Also

- [Neighbors](/cpp/kernels/neighbors) - KNN computation for affinity matrices
- [Normalization](/cpp/kernels/normalize) - Expression normalization before imputation
