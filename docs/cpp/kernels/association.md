# association.hpp

> scl/kernel/association.hpp · Feature association analysis across modalities (RNA + ATAC)

## Overview

This file provides kernels for analyzing associations between features across different modalities, particularly for multi-omics integration of RNA expression and ATAC accessibility data.

This file provides:
- Gene-peak correlation computation
- Cis-regulatory element identification
- Multi-modal feature linking

**Header**: `#include "scl/kernel/association.hpp"`

---

## Main APIs

### gene_peak_correlation

::: source_code file="scl/kernel/association.hpp" symbol="gene_peak_correlation" collapsed
:::

**Algorithm Description**

Compute correlation between RNA expression and ATAC peak accessibility across cells:

1. **Data Extraction**: For each gene-peak pair:
   - Extract expression values for gene across all cells
   - Extract accessibility values for peak across all cells
   - Handle sparse matrices efficiently

2. **Correlation Computation**: For each gene-peak pair:
   - Compute Pearson correlation coefficient:
     - Mean and standard deviation for gene expression
     - Mean and standard deviation for peak accessibility
     - Covariance between gene and peak
     - Correlation = covariance / (std_gene * std_peak)
   - Handle zero variance cases (constant values)

3. **Filtering**: Apply minimum correlation threshold:
   - Skip pairs with correlation < config::MIN_CORRELATION
   - Skip pairs with insufficient cells (< config::MIN_CELLS_FOR_CORRELATION)
   - Set low correlations to zero

4. **Output**: Store correlations in dense matrix (genes x peaks)

**Edge Cases**

- **Constant expression**: Genes or peaks with zero variance get correlation = 0
- **Insufficient cells**: Pairs with < MIN_CELLS_FOR_CORRELATION cells get correlation = 0
- **Sparse data**: Efficiently handles sparse matrices with many zeros
- **Negative correlation**: Negative correlations indicate inverse relationship
- **Perfect correlation**: Correlation = 1.0 or -1.0 indicates perfect linear relationship

**Data Guarantees (Preconditions)**

- `correlations` has capacity >= n_genes * n_peaks
- Both matrices must have same number of cells (n_cells)
- Matrices must be valid CSR format
- Cell indices must align (same cell order in both matrices)

**Complexity Analysis**

- **Time**: O(n_genes * n_peaks * n_cells) for correlation computation
  - For each gene-peak pair: O(n_cells) for correlation
  - Total: O(n_genes * n_peaks * n_cells)
  - Parallelized over gene-peak pairs
- **Space**: O(n_cells) auxiliary per gene-peak pair for storing expression/accessibility vectors

**Example**

```cpp
#include "scl/kernel/association.hpp"

scl::Sparse<Real, true> rna_expression = /* ... */;  // [n_cells x n_genes]
scl::Sparse<Real, true> atac_peaks = /* ... */;      // [n_cells x n_peaks]

Real* correlations = /* allocate n_genes * n_peaks */;

scl::kernel::association::gene_peak_correlation(
    rna_expression,
    atac_peaks,
    n_cells,
    n_genes,
    n_peaks,
    correlations
);

// Access correlation: correlations[g * n_peaks + p]
for (Index g = 0; g < n_genes; ++g) {
    for (Index p = 0; p < n_peaks; ++p) {
        Real corr = correlations[g * n_peaks + p];
        if (std::abs(corr) > config::MIN_CORRELATION) {
            // Significant association
        }
    }
}
```

---

### cis_regulatory_elements

::: source_code file="scl/kernel/association.hpp" symbol="cis_regulatory_elements" collapsed
:::

**Algorithm Description**

Identify cis-regulatory elements (peaks) linked to genes based on correlation and genomic distance:

1. **Distance Computation**: For each gene:
   - Get gene position (start, end) from gene_positions
   - For each peak, compute genomic distance:
     - Distance = min(|peak_start - gene_start|, |peak_end - gene_end|)
     - Or use overlap distance if peak overlaps gene
   - Filter peaks within max_distance

2. **Correlation Filtering**: For each gene-peak pair within distance:
   - Check correlation from correlation matrix
   - Filter by correlation threshold (config::MIN_CORRELATION)
   - Store pairs with significant correlation

3. **Score Computation**: Compute link score:
   - Score = correlation * distance_weight
   - Distance weight: higher for closer peaks (inverse distance)
   - Or use correlation directly as score

4. **Output**: Store linked pairs and scores:
   - linked_pairs[i * 2] = gene_index
   - linked_pairs[i * 2 + 1] = peak_index
   - link_scores[i] = score

**Edge Cases**

- **No nearby peaks**: Genes with no peaks within max_distance return 0 links
- **Low correlation**: Peaks with correlation < threshold are filtered out
- **Overlapping peaks**: Peaks overlapping genes get distance = 0
- **Multiple peaks per gene**: Limited by max_results, selects top by score
- **Boundary cases**: Genes/peaks at chromosome boundaries handled correctly

**Data Guarantees (Preconditions)**

- `linked_pairs` has capacity >= max_results * 2
- `link_scores` has capacity >= max_results
- Peak and gene positions must be valid (start <= end)
- Positions must be in same coordinate system (same chromosome/assembly)
- Correlation matrix must be valid (from gene_peak_correlation)

**Complexity Analysis**

- **Time**: O(n_genes * n_peaks) for distance and correlation checking
  - For each gene: O(n_peaks) to check distances and correlations
  - Filtering and sorting: O(n_genes * log(n_peaks)) worst case
  - Parallelized over genes
- **Space**: O(1) auxiliary per gene for distance computation

**Example**

```cpp
const Real* correlations = /* from gene_peak_correlation */;
const Index* peak_positions = /* [n_peaks * 2] (start, end) */;
const Index* gene_positions = /* [n_genes * 2] (start, end) */;

Index* linked_pairs = /* allocate max_results * 2 */;
Real* link_scores = /* allocate max_results */;

Index max_distance = 1000000;  // 1 Mb
Index max_results = 10000;

Index n_linked = scl::kernel::association::cis_regulatory_elements(
    correlations,
    peak_positions,
    gene_positions,
    n_genes,
    n_peaks,
    max_distance,
    linked_pairs,
    link_scores,
    max_results
);

// Process linked pairs
for (Index i = 0; i < n_linked; ++i) {
    Index gene_idx = linked_pairs[i * 2];
    Index peak_idx = linked_pairs[i * 2 + 1];
    Real score = link_scores[i];
    
    // Use linked gene-peak pair for downstream analysis
    // e.g., enhancer-promoter links, regulatory network construction
}
```

---

## Configuration

Default parameters in `scl::kernel::association::config`:

- `EPSILON = 1e-10`: Small constant for numerical stability
- `MIN_CORRELATION = 0.1`: Minimum correlation threshold for significance
- `MIN_CELLS_FOR_CORRELATION = 10`: Minimum cells required for reliable correlation
- `MAX_LINKS_PER_GENE = 1000`: Maximum links per gene to prevent explosion
- `PARALLEL_THRESHOLD = 32`: Minimum size for parallel processing

---

## Performance Notes

### Parallelization

- `gene_peak_correlation`: Parallelized over gene-peak pairs
- `cis_regulatory_elements`: Parallelized over genes
- Efficient sparse matrix access patterns

### Memory Efficiency

- Pre-allocated output buffers
- Efficient correlation computation with minimal allocations
- Sparse matrix handling for large datasets

### Optimization Tips

1. **Filter Early**: Apply distance and correlation thresholds early to reduce computation
2. **Batch Processing**: Process genes in batches to control memory usage
3. **Sparse Storage**: Consider storing only significant correlations to save memory

---

## Use Cases

### Multi-Omics Integration

```cpp
// 1. Compute gene-peak correlations
Real* correlations = /* allocate */;
scl::kernel::association::gene_peak_correlation(
    rna_expression, atac_peaks,
    n_cells, n_genes, n_peaks,
    correlations
);

// 2. Identify cis-regulatory elements
Index* linked_pairs = /* allocate */;
Real* link_scores = /* allocate */;
Index n_linked = scl::kernel::association::cis_regulatory_elements(
    correlations, peak_positions, gene_positions,
    n_genes, n_peaks, 1000000,  // 1 Mb
    linked_pairs, link_scores, max_results
);

// 3. Build regulatory network
// Use linked pairs to construct gene regulatory network
```

### Enhancer-Promoter Links

```cpp
// Find peaks near gene promoters (within 100 kb)
Index promoter_distance = 100000;  // 100 kb

Index n_links = scl::kernel::association::cis_regulatory_elements(
    correlations, peak_positions, gene_positions,
    n_genes, n_peaks, promoter_distance,
    linked_pairs, link_scores, max_results
);

// Filter by correlation strength
for (Index i = 0; i < n_links; ++i) {
    if (link_scores[i] > 0.5) {
        // Strong enhancer-promoter link
    }
}
```

---

## See Also

- [Correlation Analysis](../correlation)
- [Sparse Matrices](../core/sparse)
- [Multi-Omics Integration](../alignment)
