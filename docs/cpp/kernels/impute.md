# Imputation

High-performance imputation kernels for single-cell expression data.

## Overview

The `impute` module provides efficient imputation methods for sparse single-cell data:

- **KNN imputation**: K-nearest neighbor averaging
- **Diffusion imputation**: MAGIC-style diffusion-based imputation
- **ALRA**: Adaptively-thresholded Low-Rank Approximation
- **Weighted KNN**: Distance-weighted neighbor averaging

All operations are:
- Parallelized over cells
- Memory-efficient with sparse inputs
- Support both sparse and dense outputs

## Core Functions

### knn_impute_dense

Impute missing values using K-nearest neighbor averaging on dense output.

```cpp
#include "scl/kernel/impute.hpp"

Sparse<Real, true> X_sparse = /* sparse expression matrix */;
Sparse<Real, true> affinity = /* cell-cell affinity matrix */;
Real* X_imputed = /* pre-allocated [n_cells * n_genes] */;

scl::kernel::impute::knn_impute_dense(
    X_sparse, affinity, n_cells, n_genes, X_imputed
);
```

**Parameters:**
- `X_sparse` [in] - Input sparse expression matrix (n_cells x n_genes)
- `affinity` [in] - Cell-cell affinity matrix (n_cells x n_cells)
- `n_cells` [in] - Number of cells
- `n_genes` [in] - Number of genes
- `X_imputed` [out] - Dense imputed matrix (n_cells x n_genes, row-major)

**Preconditions:**
- `X_sparse` must be CSR format (cells x genes)
- `affinity` must be row-normalized (rows sum to 1)
- `X_imputed` must be pre-allocated with n_cells * n_genes elements

**Postconditions:**
- `X_imputed[i, j]` = weighted average of gene j across neighbors of cell i
- Weights from affinity matrix define neighbor contributions
- Dense output suitable for downstream dense operations

**Complexity:**
- Time: O(n_cells * avg_neighbors * n_genes)
- Space: O(n_cells * n_genes) for output

**Thread Safety:** Safe - parallelized over cells, each writes to independent memory

### knn_impute_weighted_dense

Impute with distance-weighted KNN contributions.

```cpp
const Index* knn_indices = /* KNN indices [n_cells * k] */;
const Real* knn_distances = /* KNN distances [n_cells * k] */;

scl::kernel::impute::knn_impute_weighted_dense(
    X_sparse, knn_indices, knn_distances, n_cells, n_genes, k, X_imputed
);
```

**Parameters:**
- `X_sparse` [in] - Input sparse expression matrix
- `knn_indices` [in] - K-nearest neighbor indices [n_cells x k]
- `knn_distances` [in] - K-nearest neighbor distances [n_cells x k]
- `n_cells` [in] - Number of cells
- `n_genes` [in] - Number of genes
- `k` [in] - Number of neighbors
- `X_imputed` [out] - Dense imputed matrix

**Preconditions:**
- `knn_indices` and `knn_distances` pre-computed for all cells
- `k <= n_cells - 1`
- `X_imputed` must be pre-allocated

**Postconditions:**
- `X_imputed[i,j]` = sum_k(weight[k] * X[neighbor_k, j]) / sum(weights)
- Weights inversely proportional to distance

**Complexity:**
- Time: O(n_cells * k * n_genes)
- Space: O(n_cells * n_genes)

**Thread Safety:** Safe - parallelized over cells

### magic_impute

MAGIC (Markov Affinity-based Graph Imputation of Cells) algorithm.

```cpp
Sparse<Real, true> transition_matrix = /* MAGIC diffusion operator */;
Index t = 3;  // Diffusion time

scl::kernel::impute::magic_impute(
    X_sparse, transition_matrix, n_cells, n_genes, t, X_imputed
);
```

**Parameters:**
- `X_sparse` [in] - Input sparse expression matrix
- `transition_matrix` [in] - Diffusion operator (from MAGIC)
- `n_cells` [in] - Number of cells
- `n_genes` [in] - Number of genes
- `t` [in] - Diffusion time parameter
- `X_imputed` [out] - Dense imputed matrix

**Preconditions:**
- `transition_matrix` from MAGIC preprocessing (symmetric normalized)
- `t >= 1`, typically t in [1, 5]
- `X_imputed` must be pre-allocated

**Postconditions:**
- `X_imputed = (T^t) * X`
- Denoised and imputed expression values
- Preserves overall expression structure

**Complexity:**
- Time: O(t * n_cells * avg_nnz * n_genes)
- Space: O(2 * n_cells * n_genes)

**Thread Safety:** Safe - all operations parallelized

**Reference:**
- van Dijk et al., MAGIC, Cell 2018

### alra_impute

ALRA (Adaptively-thresholded Low-Rank Approximation) imputation.

```cpp
const Real* X_dense = /* dense normalized expression */;
Index n_components = 50;

scl::kernel::impute::alra_impute(
    X_dense, n_cells, n_genes, n_components, X_imputed, 5, 42
);
```

**Parameters:**
- `X_dense` [in] - Dense normalized expression [n_cells x n_genes]
- `n_cells` [in] - Number of cells
- `n_genes` [in] - Number of genes
- `n_components` [in] - Number of SVD components (rank)
- `X_imputed` [out] - Dense imputed matrix
- `n_iter` [in] - Number of power iterations for SVD (default: 5)
- `seed` [in] - Random seed (default: 42)

**Preconditions:**
- `X_dense` already log-normalized
- `n_components <= min(n_cells, n_genes)`
- `X_imputed` must be pre-allocated

**Postconditions:**
- `X_imputed = U * S * V^T` (rank-k approximation)
- Negative values set to zero (biological constraint)
- Original non-zero values preserved where imputed < original

**Complexity:**
- Time: O(n_iter * n_cells * n_genes * n_components)
- Space: O(n_cells * n_components + n_genes * n_components)

**Thread Safety:** Safe - parallel matrix operations

**Reference:**
- Linderman et al., ALRA, bioRxiv 2018

### diffusion_impute_sparse_transition

Diffusion-based imputation using sparse transition matrix.

```cpp
Sparse<Real, true> transition_matrix = /* row-stochastic transition matrix */;
Index n_steps = 3;

scl::kernel::impute::diffusion_impute_sparse_transition(
    X_sparse, transition_matrix, n_cells, n_genes, n_steps, X_imputed
);
```

**Parameters:**
- `X_sparse` [in] - Input sparse expression matrix
- `transition_matrix` [in] - Row-stochastic transition matrix
- `n_cells` [in] - Number of cells
- `n_genes` [in] - Number of genes
- `n_steps` [in] - Number of diffusion steps
- `X_imputed` [out] - Dense imputed matrix

**Preconditions:**
- `transition_matrix` must be row-stochastic (rows sum to 1)
- `n_steps >= 1`
- `X_imputed` must be pre-allocated

**Postconditions:**
- `X_imputed = T^n_steps * X` where T is transition matrix
- Higher n_steps = more smoothing/imputation

**Complexity:**
- Time: O(n_steps * n_cells * avg_nnz_per_row * n_genes)
- Space: O(2 * n_cells * n_genes) for double buffering

**Thread Safety:** Safe - uses double buffering with parallel SpMM

## Auxiliary Functions

### impute_selected_genes

Impute only a subset of genes for efficiency.

```cpp
const Index* gene_indices = /* genes to impute [n_selected] */;
Real* X_imputed = /* output [n_cells * n_selected] */;

scl::kernel::impute::impute_selected_genes(
    X_sparse, affinity, gene_indices, n_selected, n_cells, X_imputed
);
```

**Parameters:**
- `X_sparse` [in] - Input sparse expression matrix
- `affinity` [in] - Cell-cell affinity matrix
- `gene_indices` [in] - Indices of genes to impute
- `n_selected` [in] - Number of genes to impute
- `n_cells` [in] - Number of cells
- `X_imputed` [out] - Imputed values for selected genes [n_cells x n_selected]

**Preconditions:**
- All gene indices in [0, n_genes)
- `X_imputed` pre-allocated with n_cells * n_selected elements

**Postconditions:**
- `X_imputed[i, j]` contains imputed value for cell i, selected gene j
- Only computes imputation for specified genes (memory efficient)

**Complexity:**
- Time: O(n_cells * avg_neighbors * n_selected)
- Space: O(n_cells * n_selected)

**Thread Safety:** Safe - parallelized over cells

### smooth_expression

Smooth expression profiles using local averaging.

```cpp
Real alpha = 0.5;  // Smoothing factor
scl::kernel::impute::smooth_expression(
    X_sparse, affinity, n_cells, n_genes, alpha, X_smooth
);
```

**Parameters:**
- `X_sparse` [in] - Input sparse expression matrix
- `affinity` [in] - Cell-cell affinity matrix
- `n_cells` [in] - Number of cells
- `n_genes` [in] - Number of genes
- `alpha` [in] - Smoothing factor (0 = original, 1 = full neighbor average)
- `X_smooth` [out] - Smoothed dense matrix

**Preconditions:**
- `alpha` in [0, 1]
- `affinity` row-normalized
- `X_smooth` must be pre-allocated

**Postconditions:**
- `X_smooth[i] = (1 - alpha) * X[i] + alpha * neighbor_average[i]`
- Interpolates between original and fully smoothed

**Complexity:**
- Time: O(n_cells * avg_neighbors * n_genes)
- Space: O(n_cells * n_genes)

**Thread Safety:** Safe - parallelized over cells

### detect_dropouts

Detect likely dropout events (technical zeros vs biological zeros).

```cpp
const Real* gene_means = /* pre-computed gene means [n_genes] */;
Index* n_dropouts = /* output [n_genes] */;

scl::kernel::impute::detect_dropouts(
    X_sparse, gene_means, n_cells, n_genes, n_dropouts, 0.5
);
```

**Parameters:**
- `X_sparse` [in] - Input sparse expression matrix
- `gene_means` [in] - Pre-computed gene means [n_genes]
- `n_cells` [in] - Number of cells
- `n_genes` [in] - Number of genes
- `n_dropouts` [out] - Count of detected dropouts [n_genes]
- `threshold` [in] - Detection threshold (default: 0.5)

**Preconditions:**
- `gene_means` pre-computed from normalized data
- `n_dropouts` must be pre-allocated with n_genes elements

**Postconditions:**
- `n_dropouts[g]` = number of cells where gene g is likely dropout
- Uses gene mean and detection rate to infer dropouts

**Complexity:**
- Time: O(n_cells * n_genes)
- Space: O(n_genes) for output

**Thread Safety:** Safe - uses atomic accumulation

## Configuration

```cpp
namespace scl::kernel::impute::config {
    constexpr Real DISTANCE_EPSILON = Real(1e-10);
    constexpr Real DEFAULT_ALPHA = Real(1.0);
    constexpr Index DEFAULT_K_NEIGHBORS = 15;
    constexpr Index DEFAULT_N_STEPS = 3;
    constexpr Index DEFAULT_N_COMPONENTS = 50;
    constexpr Size PARALLEL_THRESHOLD = 32;
    constexpr Size GENE_BLOCK_SIZE = 64;
    constexpr Size CELL_BLOCK_SIZE = 32;
}
```

## Use Cases

### KNN Imputation

```cpp
// 1. Compute cell-cell affinity (e.g., from neighbors)
Sparse<Real, true> affinity = /* compute from KNN graph */;

// 2. Normalize affinity matrix (rows sum to 1)
scl::kernel::normalize::normalize_rows_inplace(affinity, NormMode::L1);

// 3. Impute expression
Real* X_imputed = new Real[n_cells * n_genes];
scl::kernel::impute::knn_impute_dense(
    X_sparse, affinity, n_cells, n_genes, X_imputed
);
```

### MAGIC Imputation

```cpp
// 1. Build MAGIC transition matrix (from diffusion kernel)
Sparse<Real, true> transition = /* MAGIC transition matrix */;

// 2. Apply MAGIC imputation
Index t = 3;  // Diffusion time
Real* X_imputed = new Real[n_cells * n_genes];
scl::kernel::impute::magic_impute(
    X_sparse, transition, n_cells, n_genes, t, X_imputed
);
```

### ALRA Imputation

```cpp
// 1. Normalize and log-transform
Sparse<Real, true> X_normalized = /* normalized expression */;
scl::kernel::log1p::log1p_inplace(X_normalized);

// 2. Convert to dense
Real* X_dense = /* convert sparse to dense */;

// 3. Apply ALRA
Index n_components = 50;
Real* X_imputed = new Real[n_cells * n_genes];
scl::kernel::impute::alra_impute(
    X_dense, n_cells, n_genes, n_components, X_imputed
);
```

### Selective Gene Imputation

```cpp
// Impute only highly variable genes
Array<Index> hvg_indices = /* highly variable gene indices */;
Real* X_hvg_imputed = new Real[n_cells * n_hvgs];

scl::kernel::impute::impute_selected_genes(
    X_sparse, affinity, hvg_indices.ptr, n_hvgs, n_cells, X_hvg_imputed
);
```

## Performance

- **Parallelization**: Scales linearly with number of cells
- **Memory efficient**: Sparse input, optional dense output
- **Block processing**: Optimized for cache locality
- **SIMD acceleration**: Vectorized averaging operations

---

::: tip Method Selection
- **KNN**: Fast, good for small datasets
- **MAGIC**: Best for preserving biological structure
- **ALRA**: Good for large datasets, rank-based denoising
- **Weighted KNN**: Better when distance information is available
:::

