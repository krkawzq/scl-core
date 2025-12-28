# Gene Set Scoring

Gene set scoring operations for computing signature scores, module scores, and cell cycle scores.

## Overview

Gene set scoring kernels provide:

- **Multiple Scoring Methods** - Mean, rank-based (AUC), weighted, Seurat module score, z-score
- **Cell Cycle Scoring** - S-phase and G2/M-phase scores
- **Gene Set Operations** - Generic dispatcher, differential scores, quantile scores
- **High Performance** - SIMD-optimized with efficient sparse matrix operations

## Scoring Methods

### mean_score

Compute mean expression score for each cell over a gene set:

```cpp
#include "scl/kernel/scoring.hpp"

Sparse<Real, true> X = /* ... */;            // Expression matrix [n_cells x n_genes]
Array<Index> gene_set = /* ... */;            // Gene indices in the set
Index n_cells = X.rows();
Index n_genes = X.cols();

Array<Real> scores(n_cells);                  // Pre-allocated output

scl::kernel::scoring::mean_score(
    X, gene_set, scores,
    n_cells, n_genes
);

// scores[c] = mean of X[c, g] for g in gene_set
```

**Parameters:**
- `X`: Sparse matrix (cells × genes)
- `gene_set`: Array of gene indices in the set
- `scores`: Output scores, must be pre-allocated, size = n_cells
- `n_cells`: Number of cells
- `n_genes`: Number of genes

**Postconditions:**
- `scores[c] = mean of X[c, g] for g in gene_set`

**Algorithm:**
- Uses bitset lookup for O(1) gene membership check
- CSR: Parallel over cells, scan row for gene set members
- CSC: Parallel over genes, atomic accumulation to cells

**Complexity:**
- Time: O(nnz) or O(|gene_set| * avg_col_nnz)
- Space: O(n_genes / 64) for bitset

**Thread Safety:**
- Safe - uses atomic operations for CSC format

### auc_score

Compute AUC-based score using expression ranks per cell:

```cpp
Array<Real> scores(n_cells);
scl::kernel::scoring::auc_score(
    X, gene_set, scores,
    n_cells, n_genes,
    quantile = 0.05                          // Top quantile to consider
);

// scores[c] = fraction of gene_set genes in top quantile by expression
```

**Parameters:**
- `X`: Sparse matrix (cells × genes)
- `gene_set`: Array of gene indices
- `scores`: Output AUC scores, must be pre-allocated, size = n_cells
- `n_cells`: Number of cells
- `n_genes`: Number of genes
- `quantile`: Top quantile to consider (default: 0.05)

**Postconditions:**
- `scores[c] = fraction of gene_set genes in top quantile by expression`

**Algorithm:**
Per cell (parallel):
1. Extract expression values
2. Compute ranks (shell sort + insertion sort)
3. Count gene set genes in top quantile

**Complexity:**
- Time: O(n_cells * n_genes * log(n_genes))
- Space: O(n_genes) per thread for workspace

**Thread Safety:**
- Safe - uses WorkspacePool for thread-local buffers

### module_score

Compute Seurat-style module score with expression-matched control genes:

```cpp
Array<Real> scores(n_cells);
scl::kernel::scoring::module_score(
    X, gene_set, scores,
    n_cells, n_genes,
    n_control_per_gene = 1,                  // Control genes per target gene
    n_bins = 25,                             // Expression bins for matching
    seed = 42                                // Random seed
);

// scores[c] = mean(gene_set expression) - mean(control expression)
```

**Parameters:**
- `X`: Sparse matrix (cells × genes)
- `gene_set`: Array of gene indices
- `scores`: Output module scores, must be pre-allocated, size = n_cells
- `n_cells`: Number of cells
- `n_genes`: Number of genes
- `n_control_per_gene`: Control genes per target gene (default: 1)
- `n_bins`: Expression bins for matching (default: 25)
- `seed`: Random seed for control selection (default: 42)

**Postconditions:**
- `scores[c] = mean(gene_set expression) - mean(control expression)`

**Algorithm:**
1. Compute gene means and bin by expression
2. For each gene in set, sample control genes from same bin
3. Compute score as (gene_set mean) - (control mean)

**Complexity:**
- Time: O(nnz + n_genes * n_bins)
- Space: O(n_genes) for bins and control genes

**Thread Safety:**
- Safe - parallel cell scoring

## Cell Cycle Scoring

### cell_cycle_score

Compute cell cycle phase scores and assignments:

```cpp
Array<Index> s_genes = /* ... */;            // S-phase gene indices
Array<Index> g2m_genes = /* ... */;          // G2/M-phase gene indices
Array<Real> s_scores(n_cells);               // Pre-allocated
Array<Real> g2m_scores(n_cells);             // Pre-allocated
Array<Index> phase_labels(n_cells);          // Pre-allocated

scl::kernel::scoring::cell_cycle_score(
    X, s_genes, g2m_genes,
    s_scores, g2m_scores, phase_labels,
    n_cells, n_genes
);

// phase_labels[c] = phase with highest positive score, or G1 if both <= 0
// Phase: 0=G1, 1=S, 2=G2M
```

**Parameters:**
- `X`: Sparse matrix
- `s_genes`: S-phase gene indices
- `g2m_genes`: G2/M-phase gene indices
- `s_scores`: Output S-phase scores, must be pre-allocated, size = n_cells
- `g2m_scores`: Output G2/M-phase scores, must be pre-allocated, size = n_cells
- `phase_labels`: Output phase assignments, must be pre-allocated, size = n_cells
- `n_cells`: Number of cells
- `n_genes`: Number of genes

**Postconditions:**
- `phase_labels[c] = phase with highest positive score, or G1 if both <= 0`
- Phase values: 0 = G1, 1 = S, 2 = G2M

## Generic Dispatcher

### gene_set_score

Generic gene set scoring dispatcher:

```cpp
scl::kernel::scoring::gene_set_score(
    X, gene_set,
    scl::kernel::scoring::ScoringMethod::Mean,  // or RankBased, Weighted, SeuratModule, ZScore
    scores,
    n_cells, n_genes,
    quantile = 0.05                           // For AUC method
);
```

**Scoring Methods:**
- `Mean`: Simple average of gene expression
- `RankBased`: AUC-based score using expression ranks
- `Weighted`: Weighted sum with user-provided weights
- `SeuratModule`: Seurat-style module score with control genes
- `ZScore`: Z-score normalized average

---

::: tip Module Score vs. Mean Score
Module score (Seurat-style) subtracts control gene expression to reduce technical bias. Use mean score for simplicity, module score for robustness to technical variation.
:::

