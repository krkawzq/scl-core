# markers.hpp

> scl/kernel/markers.hpp · Marker gene selection and specificity scoring

## Overview

Marker gene identification and scoring functions for single-cell RNA-seq analysis. These functions identify genes that are differentially expressed in specific clusters or cell types, enabling cell type annotation and biological interpretation.

This file provides:
- Marker gene discovery through differential expression analysis
- Multiple ranking methods (fold change, effect size, p-value, combined)
- Cluster-specific gene scoring
- Parallel processing for large datasets

**Header**: `#include "scl/kernel/markers.hpp"`

---

## Main APIs

### find_markers

::: source_code file="scl/kernel/markers.hpp" symbol="find_markers" collapsed
:::

**Algorithm Description**

Identifies marker genes for each cluster using differential expression analysis:

1. **For each cluster c** (processed in parallel):
   - Extract cells belonging to cluster c (in-group)
   - Extract cells not in cluster c (out-group)
   - For each gene g:
     - Compute mean expression in in-group and out-group
     - Compute fold change: FC = (mean_in + pseudo_count) / (mean_out + pseudo_count)
     - Compute statistical test (t-test or Mann-Whitney U test) p-value
     - Compute effect size (Cohen's d or similar)
2. **Rank genes** according to selected method:
   - FoldChange: Sort by fold change (descending)
   - EffectSize: Sort by effect size (descending)
   - PValue: Sort by p-value (ascending)
   - Combined: Weighted combination of fold change, p-value, and effect size
3. **Filter markers**:
   - Keep genes with FC >= min_fc
   - Keep genes with p-value <= max_pval
   - Select top max_markers genes per cluster
4. **Store results** in marker_genes and marker_scores arrays

**Edge Cases**

- **Empty cluster**: No markers found, all marker_genes set to invalid index
- **Single-cell cluster**: Statistical tests may be unreliable, handled gracefully
- **All genes filtered**: Returns fewer than max_markers per cluster
- **Tied scores**: Ordering is deterministic but arbitrary for ties
- **Zero expression**: Uses pseudo_count (1.0) to avoid division by zero

**Data Guarantees (Preconditions)**

- Expression matrix must be valid sparse matrix (CSR format)
- Cluster labels array length must equal number of cells (n_cells)
- Marker arrays must have capacity >= n_clusters * max_markers
- All cluster labels must be in range [0, n_clusters-1]
- Number of clusters must be > 0

**Complexity Analysis**

- **Time**: O(n_clusters * n_genes * n_cells) in worst case. Parallel processing over clusters reduces effective time. Statistical tests dominate the complexity.
- **Space**: O(n_cells) auxiliary space per cluster for storing in-group/out-group indices (with parallelization, scales with number of threads)

**Example**

```cpp
#include "scl/kernel/markers.hpp"

Sparse<Real, true> expression = /* expression matrix [n_cells x n_genes] */;
Array<Index> cluster_labels = /* cluster assignments [n_cells] */;

Index n_cells = expression.rows();
Index n_genes = expression.cols();
Index n_clusters = /* ... */;
Index max_markers = 50;

Index* marker_genes = new Index[n_clusters * max_markers];
Real* marker_scores = new Real[n_clusters * max_markers];

// Find markers with default parameters
scl::kernel::markers::find_markers(
    expression, cluster_labels, n_cells, n_genes, n_clusters,
    marker_genes, marker_scores, max_markers
);

// Find markers with custom thresholds
scl::kernel::markers::find_markers(
    expression, cluster_labels, n_cells, n_genes, n_clusters,
    marker_genes, marker_scores, max_markers,
    min_fc = 2.0,        // Minimum 2-fold change
    max_pval = 0.01,     // Maximum p-value 0.01
    method = scl::kernel::markers::RankingMethod::Combined
);

// Access markers for cluster c:
// marker_genes[c * max_markers + i] = gene index
// marker_scores[c * max_markers + i] = score
```

---

### specificity_score

::: source_code file="scl/kernel/markers.hpp" symbol="specificity_score" collapsed
:::

**Algorithm Description**

Computes a cluster-specific expression score for a single gene:

1. Compute mean expression of the gene in the target cluster
2. Compute mean expression of the gene in all other clusters
3. Compute specificity as a ratio or difference metric:
   - Often uses log fold change or z-score normalization
   - Higher values indicate greater specificity to target cluster

The exact formula may vary, but typically measures how much more expressed the gene is in the target cluster compared to others.

**Edge Cases**

- **Gene not expressed**: Returns low specificity score (near zero or negative)
- **Uniform expression**: Returns specificity near zero
- **Empty target cluster**: Returns undefined/invalid score
- **Single-cell cluster**: May have high variance in specificity

**Data Guarantees (Preconditions)**

- Expression matrix must be valid sparse matrix
- Cluster labels array length must equal n_cells
- Target cluster ID must be valid (in range [0, n_clusters-1])
- Gene index must be valid (in range [0, n_genes-1])

**Complexity Analysis**

- **Time**: O(n_cells) - must examine expression in all cells for the given gene
- **Space**: O(1) auxiliary space - only accumulates statistics

**Example**

```cpp
#include "scl/kernel/markers.hpp"

Sparse<Real, true> expression = /* expression matrix */;
Array<Index> cluster_labels = /* cluster labels */;
Index target_cluster = 3;
Index gene_index = 125;  // Gene of interest
Index n_cells = expression.rows();

Real specificity;
scl::kernel::markers::specificity_score(
    expression, cluster_labels, gene_index, target_cluster, n_cells, specificity
);

// specificity now contains the cluster-specific expression score
// Higher values indicate the gene is more specific to cluster 3
```

---

## Configuration

The namespace `scl::kernel::markers::config` provides configuration constants:

- `DEFAULT_MIN_FC = 1.5`: Default minimum fold change threshold
- `DEFAULT_MIN_PCT = 0.1`: Default minimum percentage of cells expressing gene
- `DEFAULT_MAX_PVAL = 0.05`: Default maximum p-value threshold
- `MIN_EXPR = 1e-9`: Minimum expression value (numerical stability)
- `PSEUDO_COUNT = 1.0`: Pseudo-count added to avoid log(0) and division by zero
- `PARALLEL_THRESHOLD = 500`: Minimum cluster size for parallel processing
- `SIMD_THRESHOLD = 32`: Minimum vector length for SIMD optimization

## Ranking Methods

The `RankingMethod` enum provides different strategies for ranking marker genes:

- `FoldChange`: Rank by fold change (descending)
- `EffectSize`: Rank by effect size (descending)
- `PValue`: Rank by p-value (ascending)
- `Combined`: Weighted combination of multiple metrics

## Notes

- Marker identification is sensitive to the expression matrix preprocessing (normalization, log transformation). Ensure consistent preprocessing before marker analysis.
- For small clusters (< 5 cells), statistical tests may be unreliable. Consider using effect size or fold change alone.
- The pseudo_count is added to prevent numerical issues with zero expression values, but may affect results for very lowly expressed genes.
- Marker lists should be interpreted in context of the biological system. Highly expressed genes may dominate rankings even if not biologically meaningful.

## See Also

- [Multiple Testing](./multiple_testing) - Statistical correction for multiple hypothesis testing
- [Statistics](./statistics) - Statistical tests used in marker identification
- [Scoring](./scoring) - Alternative scoring methods for gene selection
