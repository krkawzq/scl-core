# Marker Gene Selection

Marker gene identification and specificity scoring for cell type annotation.

## Overview

Marker selection kernels provide:

- **Marker Identification** - Find marker genes for clusters using differential expression
- **Specificity Scoring** - Compute cluster-specific expression scores
- **Multiple Ranking Methods** - Fold change, effect size, p-value, combined
- **Parallel Processing** - Efficient for large datasets

## Marker Selection

### find_markers

Find marker genes for each cluster using differential expression:

```cpp
#include "scl/kernel/markers.hpp"

Sparse<Real, true> expression = /* ... */;    // Expression matrix [n_cells x n_genes]
Array<Index> cluster_labels = /* ... */;      // Cluster labels [n_cells]
Index n_cells = expression.rows();
Index n_genes = expression.cols();
Index n_clusters = /* ... */;

Index max_markers = 100;
Array<Index> marker_genes(n_clusters * max_markers);  // Pre-allocated
Array<Real> marker_scores(n_clusters * max_markers);  // Pre-allocated

scl::kernel::markers::find_markers(
    expression, cluster_labels,
    n_cells, n_genes, n_clusters,
    marker_genes.ptr, marker_scores.ptr,
    max_markers,
    min_fc = 1.5,                             // Minimum fold change
    max_pval = 0.05,                          // Maximum p-value
    method = scl::kernel::markers::RankingMethod::Combined
);

// marker_genes[c * max_markers + i] contains marker gene index
// marker_scores[c * max_markers + i] contains corresponding score
```

**Parameters:**
- `expression`: Expression matrix (cells × genes, CSR format)
- `cluster_labels`: Cluster labels, size = n_cells
- `n_cells`: Number of cells
- `n_genes`: Number of genes
- `n_clusters`: Number of clusters
- `marker_genes`: Output marker gene indices, must be pre-allocated, size = n_clusters × max_markers
- `marker_scores`: Output marker scores, must be pre-allocated, size = n_clusters × max_markers
- `max_markers`: Maximum markers per cluster
- `min_fc`: Minimum fold change (default: 1.5)
- `max_pval`: Maximum p-value (default: 0.05)
- `method`: Ranking method (default: Combined)

**Postconditions:**
- `marker_genes[c * max_markers + i]` contains marker gene index
- Returns number of markers found per cluster
- Markers are ranked by selected method

**Ranking Methods:**
- `FoldChange`: Rank by fold change (highest first)
- `EffectSize`: Rank by effect size (Cohen's d)
- `PValue`: Rank by p-value (lowest first)
- `Combined`: Combined score (fold change × -log10(p-value))

**Complexity:**
- Time: O(n_clusters * n_genes * n_cells)
- Space: O(n_cells) auxiliary per cluster

**Thread Safety:**
- Safe - parallelized over clusters

**Use cases:**
- Cell type annotation
- Cluster characterization
- Feature selection
- Differential expression analysis

## Specificity Scoring

### specificity_score

Compute gene specificity score for a cluster:

```cpp
Index gene_index = 10;                        // Gene to score
Index target_cluster = 5;                     // Target cluster ID
Real specificity;

scl::kernel::markers::specificity_score(
    expression, cluster_labels,
    gene_index, target_cluster,
    n_cells,
    specificity
);

// specificity contains cluster-specific expression score
```

**Parameters:**
- `expression`: Expression matrix (cells × genes, CSR format)
- `cluster_labels`: Cluster labels, size = n_cells
- `gene_index`: Gene index to score
- `target_cluster`: Target cluster ID
- `n_cells`: Number of cells
- `specificity`: Output specificity score

**Postconditions:**
- `specificity` contains cluster-specific expression score
- Higher scores indicate more specific expression

**Complexity:**
- Time: O(n_cells)
- Space: O(1) auxiliary

**Thread Safety:**
- Safe - no shared state

**Use cases:**
- Evaluate marker quality
- Filter candidate markers
- Quantify gene specificity

## Configuration

### Default Parameters

```cpp
namespace config {
    constexpr Real DEFAULT_MIN_FC = Real(1.5);
    constexpr Real DEFAULT_MIN_PCT = Real(0.1);
    constexpr Real DEFAULT_MAX_PVAL = Real(0.05);
    constexpr Real MIN_EXPR = Real(1e-9);
    constexpr Real PSEUDO_COUNT = Real(1.0);
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size SIMD_THRESHOLD = 32;
}
```

---

::: tip Combined Ranking
The Combined method balances statistical significance (p-value) and biological effect (fold change), providing a robust ranking that captures both aspects of marker quality.
:::

