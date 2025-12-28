# grn.hpp

> scl/kernel/grn.hpp · Gene regulatory network inference from expression data

## Overview

This file provides efficient inference of gene regulatory networks (GRNs) from single-cell expression data:

- **Correlation-based**: Simple Pearson correlation networks
- **Partial Correlation**: Networks controlling for other genes (removes indirect effects)
- **Mutual Information**: Information-theoretic networks
- **GENIE3**: Tree-based ensemble method for directed networks
- **Combined**: Weighted combination of multiple methods

All operations are parallelized over gene pairs and use memory-efficient sparse storage.

**Header**: `#include "scl/kernel/grn.hpp"`

---

## Main APIs

### infer_grn

::: source_code file="scl/kernel/grn.hpp" symbol="infer_grn" collapsed
:::

**Algorithm Description**

Infer gene regulatory network from expression data using various methods:

1. **Correlation Method** (default):
   - For each gene pair (i, j) in parallel:
     - Compute Pearson correlation: `corr = cov(X_i, X_j) / (std(X_i) * std(X_j))`
     - Apply threshold: if `|corr| < threshold`, set edge weight to 0
     - Store edge weight: `network[i * n_genes + j] = corr`
   - Uses SIMD-accelerated correlation computation

2. **Partial Correlation Method**:
   - Compute full correlation matrix
   - Invert correlation matrix to get precision matrix
   - Partial correlation: `partial_corr[i,j] = -precision[i,j] / sqrt(precision[i,i] * precision[j,j])`
   - Removes indirect effects mediated by other genes

3. **Mutual Information Method**:
   - For each gene pair, compute MI using binning:
     - Discretize expression values into bins
     - Compute joint and marginal distributions
     - MI = sum(p(x,y) * log(p(x,y) / (p(x) * p(y))))
   - Threshold by MI value

4. **GENIE3 Method**:
   - For each target gene, train random forest:
     - Use other genes as features to predict target
     - Edge weight = feature importance from forest
   - More computationally expensive but captures directed relationships

5. **Combined Method**:
   - Weighted combination of multiple methods
   - Allows integration of different network inference approaches

**Edge Cases**

- **Constant genes (zero variance)**: Correlation undefined, set to 0
- **Few cells (< 3)**: Correlation may be unreliable
- **Sparse expression**: Handled correctly by sparse matrix operations
- **Self-edges**: Can be included or excluded based on use case

**Data Guarantees (Preconditions)**

- `network` has capacity >= `n_genes * n_genes`
- Expression matrix should be CSR format (cells x genes)
- `n_cells >= 3` for reliable correlation computation
- `threshold` should be in range [0, 1] for correlation methods

**Complexity Analysis**

- **Time**: 
  - Correlation: O(n_genes^2 * n_cells) parallelized over gene pairs
  - Partial Correlation: O(n_genes^3 * n_cells) due to matrix inversion
  - Mutual Information: O(n_genes^2 * n_cells * n_bins)
  - GENIE3: O(n_genes * n_trees * n_cells * log(n_cells))
- **Space**: O(n_genes^2) auxiliary space for correlation/precision matrices

**Example**

```cpp
#include "scl/kernel/grn.hpp"

// Expression matrix (cells x genes)
scl::Sparse<Real, true> expression = /* ... */;
scl::Index n_cells = expression.rows();
scl::Index n_genes = expression.cols();

// Pre-allocate network adjacency matrix
Real* network = new Real[n_genes * n_genes];

// Infer correlation-based GRN with threshold 0.3
scl::kernel::grn::infer_grn(
    expression, n_cells, n_genes, network,
    scl::kernel::grn::GRNMethod::Correlation, 0.3
);

// Filter edges by threshold (already applied, but can filter further)
for (scl::Index i = 0; i < n_genes; ++i) {
    for (scl::Index j = 0; j < n_genes; ++j) {
        if (std::abs(network[i * n_genes + j]) < 0.3) {
            network[i * n_genes + j] = 0.0;
        }
    }
}
```

---

### partial_correlation

::: source_code file="scl/kernel/grn.hpp" symbol="partial_correlation" collapsed
:::

**Algorithm Description**

Compute partial correlation matrix controlling for other genes:

1. **Compute full correlation matrix**: C[i,j] = corr(X_i, X_j) for all gene pairs
2. **Invert correlation matrix**: P = C^-1 (precision matrix)
3. **Compute partial correlations**: 
   - `partial_corr[i,j] = -P[i,j] / sqrt(P[i,i] * P[j,j])`
   - This removes indirect effects mediated by other genes
4. **Threshold**: Apply threshold to filter weak edges

Partial correlation measures direct relationships by controlling for all other genes, providing more accurate inference of direct regulatory relationships.

**Edge Cases**

- **Singular correlation matrix**: Cannot invert, returns error or uses regularization
- **Few cells**: Matrix inversion may be unstable
- **Highly correlated genes**: Precision matrix may have numerical issues

**Data Guarantees (Preconditions)**

- `partial_corr` has capacity >= `n_genes * n_genes`
- Expression matrix should be CSR format
- `n_cells >= n_genes` for stable matrix inversion (recommended)
- Correlation matrix must be invertible (positive definite)

**Complexity Analysis**

- **Time**: O(n_genes^3 * n_cells)
  - O(n_genes^2 * n_cells) for correlation computation
  - O(n_genes^3) for matrix inversion
- **Space**: O(n_genes^2) auxiliary space for correlation and precision matrices

**Example**

```cpp
#include "scl/kernel/grn.hpp"

scl::Sparse<Real, true> expression = /* ... */;
scl::Index n_cells = expression.rows();
scl::Index n_genes = expression.cols();

// Pre-allocate partial correlation matrix
Real* partial_corr = new Real[n_genes * n_genes];

// Compute partial correlations
scl::kernel::grn::partial_correlation(
    expression, n_cells, n_genes, partial_corr
);

// Partial correlation removes indirect effects
// More accurate for direct regulatory relationships
```

---

## Configuration

The module provides configuration constants in `scl::kernel::grn::config`:

- `DEFAULT_CORRELATION_THRESHOLD`: Default threshold for correlation filtering (0.3)
- `EPSILON`: Small constant for numerical stability (1e-15)
- `DEFAULT_N_BINS`: Default number of bins for MI computation (10)
- `DEFAULT_N_TREES`: Default number of trees for GENIE3 (100)
- `DEFAULT_SUBSAMPLE`: Default subsample size for GENIE3 (500)
- `PARALLEL_THRESHOLD`: Minimum size for parallelization (500)
- `SIMD_THRESHOLD`: Minimum size for SIMD operations (32)
- `PREFETCH_DISTANCE`: Prefetch distance for cache optimization (16)

---

## GRN Method Types

The `GRNMethod` enum provides different inference methods:

- `GRNMethod::Correlation`: Pearson correlation (fastest, default)
- `GRNMethod::PartialCorrelation`: Partial correlation (removes indirect effects)
- `GRNMethod::MutualInformation`: Mutual information (non-linear relationships)
- `GRNMethod::GENIE3`: Tree ensemble (directed networks, slower)
- `GRNMethod::Combined`: Weighted combination of methods

---

## Notes

- Correlation method is fastest and suitable for most use cases
- Partial correlation is more accurate but requires n_cells >= n_genes for stability
- GENIE3 captures directed relationships but is computationally expensive
- For large gene sets (>1000 genes), consider filtering by variance first
- Network can be stored as sparse matrix to save memory
- All operations are parallelized and thread-safe

## See Also

- [Sparse Matrix Operations](../core/sparse)
- [Correlation Analysis](./correlation)
- [Statistical Tests](./stat)
