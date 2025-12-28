# scoring.hpp

> scl/kernel/scoring.hpp · Gene set scoring operations

## Overview

This file provides high-performance kernels for computing gene set scores across cells. It supports multiple scoring methods including mean expression, rank-based AUC scores, weighted sums, Seurat-style module scores, and z-score normalized scores. Also includes specialized functions for cell cycle scoring and multi-signature batch processing.

**Header**: `#include "scl/kernel/scoring.hpp"`

---

## Main APIs

### mean_score

::: source_code file="scl/kernel/scoring.hpp" symbol="mean_score" collapsed
:::

**Algorithm Description**

Computes mean expression score for each cell over a gene set:

1. Build bitset lookup table for O(1) gene membership check
2. For CSR format: Parallel over cells, scan each row for gene set members
3. For CSC format: Parallel over genes, atomic accumulation to cells
4. Compute mean: `scores[c] = sum(X[c, g] for g in gene_set) / |gene_set|`
5. Uses efficient sparse matrix iteration with bitset membership test

**Edge Cases**

- **Empty gene set**: Returns zero scores for all cells
- **Gene not in matrix**: Invalid gene indices cause undefined behavior
- **Zero expression**: Cells with zero expression for all genes in set get score 0
- **Sparse genes**: Efficiently handles genes with few non-zero values

**Data Guarantees (Preconditions)**

- `scores.len >= n_cells`
- All gene indices in `gene_set` are in range [0, n_genes)
- Expression matrix is valid CSR or CSC format
- Matrix dimensions match n_cells and n_genes

**Complexity Analysis**

- **Time**: O(nnz) for CSR or O(|gene_set| * avg_col_nnz) for CSC
- **Space**: O(n_genes / 64) for bitset lookup table

**Example**

```cpp
#include "scl/kernel/scoring.hpp"

Sparse<Real, true> expression = /* cells x genes, CSR */;
Array<const Index> gene_set = /* gene indices in the set */;
Array<Real> scores(n_cells);

scl::kernel::scoring::mean_score(
    expression,
    gene_set,
    scores,
    n_cells,
    n_genes
);

// scores[c] contains mean expression of gene set in cell c
```

---

### weighted_score

::: source_code file="scl/kernel/scoring.hpp" symbol="weighted_score" collapsed
:::

**Algorithm Description**

Computes weighted sum score for each cell over a gene set:

1. Build weight map from gene indices to weights
2. For each cell in parallel:
   - Iterate over non-zero elements in cell's row
   - If gene is in set, accumulate `weight[gene] * expression[cell, gene]`
3. Normalize by sum of weights: `scores[c] = sum(weight[i] * X[c, gene_set[i]]) / sum(weight)`
4. Uses atomic operations for CSC format to accumulate across genes

**Edge Cases**

- **Zero weights**: Genes with zero weight contribute nothing
- **Negative weights**: Allowed, can produce negative scores
- **Empty gene set**: Returns zero scores
- **Weight sum zero**: Returns zero scores (division by zero avoided)

**Data Guarantees (Preconditions)**

- `scores.len >= n_cells`
- `gene_weights.len >= gene_set.len`
- All gene indices in [0, n_genes)
- Expression matrix is valid sparse format

**Complexity Analysis**

- **Time**: O(nnz) - linear in number of non-zeros
- **Space**: O(n_genes) for weight map

**Example**

```cpp
Array<const Index> gene_set = /* gene indices */;
Array<const Real> gene_weights = /* weights for each gene */;
Array<Real> scores(n_cells);

scl::kernel::scoring::weighted_score(
    expression,
    gene_set,
    gene_weights,
    scores,
    n_cells,
    n_genes
);

// scores[c] = weighted average of gene set expression in cell c
```

---

### auc_score

::: source_code file="scl/kernel/scoring.hpp" symbol="auc_score" collapsed
:::

**Algorithm Description**

Computes AUC-based score using expression ranks per cell:

1. For each cell in parallel:
   - Extract expression values for all genes
   - Compute ranks using shell sort + insertion sort hybrid
   - Count how many gene set genes are in top quantile (e.g., top 5%)
   - Score = fraction of gene set genes in top quantile
2. Uses WorkspacePool for thread-local buffers to avoid allocations
3. Efficient ranking algorithm optimized for typical gene counts

**Edge Cases**

- **Empty gene set**: Returns zero scores
- **Quantile = 0**: Returns zero scores (no genes in top 0%)
- **Quantile = 1**: Returns 1.0 if all genes in set are expressed
- **Tied ranks**: Standard tie-breaking applied

**Data Guarantees (Preconditions)**

- `scores.len >= n_cells`
- `0 < quantile <= 1`
- Expression matrix is valid sparse format

**Complexity Analysis**

- **Time**: O(n_cells * n_genes * log(n_genes)) - ranking dominates
- **Space**: O(n_genes) per thread for workspace

**Example**

```cpp
Array<Real> scores(n_cells);

scl::kernel::scoring::auc_score(
    expression,
    gene_set,
    scores,
    n_cells,
    n_genes,
    0.05  // Top 5% quantile
);

// scores[c] = fraction of gene set in top 5% by expression in cell c
```

---

### module_score

::: source_code file="scl/kernel/scoring.hpp" symbol="module_score" collapsed
:::

**Algorithm Description**

Computes Seurat-style module score with expression-matched control genes:

1. Compute gene means across all cells
2. Bin genes by expression level (default 25 bins)
3. For each gene in set:
   - Sample control genes from same expression bin
   - Default: 1 control per target gene
4. For each cell:
   - Compute mean expression of gene set
   - Compute mean expression of control genes
   - Score = gene_set_mean - control_mean
5. Uses random seed for reproducible control selection

**Edge Cases**

- **Empty gene set**: Returns zero scores
- **Insufficient controls**: If bin has fewer genes than requested, uses all available
- **Zero expression genes**: Handled correctly in binning
- **Random seed**: Same seed produces same control selection

**Data Guarantees (Preconditions)**

- `scores.len >= n_cells`
- Expression matrix is valid sparse format
- Sufficient genes available for control matching

**Complexity Analysis**

- **Time**: O(nnz + n_genes * n_bins) - mean computation plus binning
- **Space**: O(n_genes) for bins and control gene arrays

**Example**

```cpp
Array<Real> scores(n_cells);

scl::kernel::scoring::module_score(
    expression,
    gene_set,
    scores,
    n_cells,
    n_genes,
    1,      // 1 control per gene
    25,     // 25 expression bins
    42      // Random seed
);

// scores[c] = mean(gene_set) - mean(control_genes) for cell c
```

---

### zscore_score

::: source_code file="scl/kernel/scoring.hpp" symbol="zscore_score" collapsed
:::

**Algorithm Description**

Computes z-score normalized gene set score:

1. Compute gene-wise mean and standard deviation across all cells
2. Precompute z-score for zero expression: `z_zero = (0 - mean) / std`
3. For each cell in parallel:
   - Extract expression values for genes in set
   - Convert to z-scores: `z = (expr - mean) / std` or use precomputed z_zero
   - Average z-scores: `scores[c] = mean(z-scores for genes in set)`
4. Uses WorkspacePool for thread-local buffers

**Edge Cases**

- **Zero-variance genes**: Genes with zero variance get z-score 0
- **Empty gene set**: Returns zero scores
- **Zero expression**: Uses precomputed z_zero for efficiency
- **Negative z-scores**: Allowed, indicates below-average expression

**Data Guarantees (Preconditions)**

- `scores.len >= n_cells`
- Expression matrix is valid sparse format
- At least 2 cells required for variance computation

**Complexity Analysis**

- **Time**: O(nnz + n_cells * |gene_set|) - stats computation plus per-cell scoring
- **Space**: O(n_genes + |gene_set|) per thread for workspace

**Example**

```cpp
Array<Real> scores(n_cells);

scl::kernel::scoring::zscore_score(
    expression,
    gene_set,
    scores,
    n_cells,
    n_genes
);

// scores[c] = mean z-score of gene set in cell c
```

---

### cell_cycle_score

::: source_code file="scl/kernel/scoring.hpp" symbol="cell_cycle_score" collapsed
:::

**Algorithm Description**

Computes cell cycle phase scores and assignments:

1. Compute S-phase score using S-phase gene set
2. Compute G2/M-phase score using G2/M-phase gene set
3. For each cell:
   - If both scores <= 0: assign G1 phase (label 0)
   - Otherwise: assign phase with highest positive score
     - S score highest: S phase (label 1)
     - G2M score highest: G2/M phase (label 2)
4. Uses mean_score internally for phase scoring

**Edge Cases**

- **Both scores negative**: Cell assigned to G1 phase
- **Equal scores**: G2/M takes precedence if both positive
- **Empty gene sets**: Returns zero scores and G1 assignment
- **Tie at zero**: Assigned to G1

**Data Guarantees (Preconditions)**

- `s_scores.len >= n_cells`
- `g2m_scores.len >= n_cells`
- `phase_labels.len >= n_cells`
- Expression matrix is valid sparse format

**Complexity Analysis**

- **Time**: O(nnz) - dominated by mean_score calls
- **Space**: O(n_genes) for bitset lookups

**Example**

```cpp
Array<const Index> s_genes = /* S-phase gene indices */;
Array<const Index> g2m_genes = /* G2/M-phase gene indices */;
Array<Real> s_scores(n_cells);
Array<Real> g2m_scores(n_cells);
Array<Index> phase_labels(n_cells);

scl::kernel::scoring::cell_cycle_score(
    expression,
    s_genes,
    g2m_genes,
    s_scores,
    g2m_scores,
    phase_labels,
    n_cells,
    n_genes
);

// phase_labels[c] = 0 (G1), 1 (S), or 2 (G2/M)
```

---

## Utility Functions

### compute_gene_means

Computes mean expression for each gene across all cells.

::: source_code file="scl/kernel/scoring.hpp" symbol="compute_gene_means" collapsed
:::

**Complexity**

- Time: O(nnz)
- Space: O(n_genes) for atomic counters (CSR only)

---

### gene_set_score

Generic gene set scoring dispatcher that routes to appropriate method.

::: source_code file="scl/kernel/scoring.hpp" symbol="gene_set_score" collapsed
:::

**Complexity**

- Time: Depends on selected method
- Space: Depends on selected method

---

### differential_score

Computes differential score between positive and negative gene sets.

::: source_code file="scl/kernel/scoring.hpp" symbol="differential_score" collapsed
:::

**Complexity**

- Time: O(nnz)
- Space: O(n_genes) for bitset lookups

---

### quantile_score

Computes quantile of gene set expression per cell.

::: source_code file="scl/kernel/scoring.hpp" symbol="quantile_score" collapsed
:::

**Complexity**

- Time: O(n_cells * |gene_set| * log(|gene_set|))
- Space: O(|gene_set|) per thread

---

### multi_signature_score

Scores multiple gene signatures in parallel.

::: source_code file="scl/kernel/scoring.hpp" symbol="multi_signature_score" collapsed
:::

**Complexity**

- Time: O(n_sets * nnz) - linear in number of signatures
- Space: O(n_genes) for bitset per signature

---

## Scoring Methods

The `ScoringMethod` enum provides different scoring approaches:

- `Mean`: Simple average of gene expression
- `RankBased`: AUC-based score using expression ranks
- `Weighted`: Weighted sum with user-provided weights
- `SeuratModule`: Seurat-style module score with control genes
- `ZScore`: Z-score normalized average

---

## Cell Cycle Phases

The `CellCyclePhase` enum represents cell cycle phases:

- `G1 = 0`: Gap 1 phase
- `S = 1`: Synthesis phase
- `G2M = 2`: G2/Mitosis phase

---

## See Also

- [Normalize Module](./normalize) - Expression normalization
- [Comparison Module](./comparison) - Statistical comparisons
