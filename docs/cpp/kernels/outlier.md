# outlier.hpp

> scl/kernel/outlier.hpp · Outlier and anomaly detection kernels

## Overview

This file provides kernels for detecting outliers and anomalies in single-cell data, including statistical deviation, local density, ambient RNA contamination, and quality control filtering.

This file provides:
- Isolation score computation
- Local Outlier Factor (LOF) detection
- Ambient RNA detection
- Empty droplet identification
- Outlier gene detection
- Doublet scoring
- Mitochondrial outlier detection
- QC filtering

**Header**: `#include "scl/kernel/outlier.hpp"`

---

## Main APIs

### isolation_score

::: source_code file="scl/kernel/outlier.hpp" symbol="isolation_score" collapsed
:::

**Algorithm Description**

Compute isolation scores based on statistical deviation from global cell population characteristics:

1. **Per-cell Statistics**: For each cell:
   - Compute mean expression over all genes
   - Compute variance of expression over all genes
   - Handle sparse expression efficiently

2. **Global Statistics**: Compute population-level statistics:
   - Global mean = mean of all cell means
   - Global variance = variance of all cell means
   - Global variance of variances

3. **Deviation Scoring**: For each cell:
   - Mean deviation = |cell_mean - global_mean| / global_std
   - Variance deviation = |cell_var - global_var| / global_var_std
   - Isolation score = (mean_deviation + variance_deviation) / 2

4. **Output**: Store scores in output array:
   - Higher scores indicate more isolated/outlier cells

**Edge Cases**

- **Zero expression**: Cells with all-zero expression get high isolation score
- **Constant expression**: Cells with zero variance get high variance deviation
- **Extreme values**: Cells with very high or very low expression get high scores
- **Empty cells**: Cells with no expression get maximum isolation score

**Data Guarantees (Preconditions)**

- `data.rows() == scores.len`
- Scores array must be pre-allocated
- Expression matrix must be valid

**Complexity Analysis**

- **Time**: O(nnz + n_cells * n_genes) for statistics computation
  - Per-cell statistics: O(nnz)
  - Global statistics: O(n_cells)
  - Scoring: O(n_cells)
- **Space**: O(n_cells) auxiliary for storing per-cell statistics

**Example**

```cpp
#include "scl/kernel/outlier.hpp"

scl::Sparse<Real, true> expression = /* ... */;  // [n_cells x n_genes]
scl::Array<Real> isolation_scores(n_cells);

scl::kernel::outlier::isolation_score(expression, isolation_scores);

// Filter outliers
for (Index i = 0; i < n_cells; ++i) {
    if (isolation_scores[i] > 3.0) {
        // Cell i is an outlier
    }
}
```

---

### local_outlier_factor

::: source_code file="scl/kernel/outlier.hpp" symbol="local_outlier_factor" collapsed
:::

**Algorithm Description**

Compute Local Outlier Factor (LOF) for each cell based on local density compared to neighbors:

1. **K-distance Computation**: For each cell:
   - K-distance = distance to k-th nearest neighbor
   - Used to define local neighborhood

2. **Reachability Distance**: For each cell-neighbor pair:
   - Reachability distance = max(k-distance(neighbor), distance(cell, neighbor))
   - Ensures symmetric reachability

3. **Local Reachability Density (LRD)**: For each cell:
   - LRD = 1 / (mean reachability distance to k neighbors)
   - Higher LRD indicates denser local neighborhood

4. **LOF Computation**: For each cell:
   - LOF = mean(LRD of neighbors) / LRD of cell
   - LOF ~ 1 indicates normal density
   - LOF > 1.5 typically indicates outlier

**Edge Cases**

- **Isolated cells**: Cells with no neighbors get LOF = infinity or very high value
- **Dense clusters**: Cells in dense clusters get LOF < 1
- **Border cells**: Cells at cluster borders get LOF > 1
- **Insufficient neighbors**: Cells with < k neighbors use available neighbors

**Data Guarantees (Preconditions)**

- All matrices have same number of rows
- Neighbors and distances must be aligned (same structure)
- `lof_scores.len == data.rows()`
- KNN graph must be valid

**Complexity Analysis**

- **Time**: O(n_cells * k^2) where k = neighbors per cell
  - K-distance: O(n_cells * k)
  - Reachability: O(n_cells * k)
  - LRD and LOF: O(n_cells * k)
- **Space**: O(n_cells + k) auxiliary for storing distances and LRD

**Example**

```cpp
scl::Sparse<Real, true> expression = /* ... */;
scl::Sparse<Index, true> neighbors = /* ... */;  // KNN graph
scl::Sparse<Real, true> distances = /* ... */;   // KNN distances

scl::Array<Real> lof_scores(n_cells);

scl::kernel::outlier::local_outlier_factor(
    expression,
    neighbors,
    distances,
    lof_scores
);

// Filter outliers (LOF > 1.5)
for (Index i = 0; i < n_cells; ++i) {
    if (lof_scores[i] > config::LOF_THRESHOLD) {
        // Cell i is an outlier
    }
}
```

---

### ambient_detection

::: source_code file="scl/kernel/outlier.hpp" symbol="ambient_detection" collapsed
:::

**Algorithm Description**

Compute ambient RNA contamination score for each cell based on correlation with estimated ambient profile:

1. **UMI Computation**: For each cell:
   - Compute total UMI count (sum of expression)
   - Sort cells by UMI count

2. **Ambient Profile Estimation**: 
   - Identify bottom 10% UMI cells as ambient reference
   - Build ambient profile = mean expression of reference cells
   - Normalize ambient profile

3. **Correlation Computation**: For each cell:
   - Compute cosine similarity between cell expression and ambient profile
   - Higher correlation indicates more ambient contamination

4. **Output**: Store ambient scores:
   - Score in [0, 1]
   - 1 indicates high correlation with ambient profile
   - 0 indicates no correlation

**Edge Cases**

- **Low UMI cells**: Cells with very low UMI get high ambient scores
- **Empty cells**: Cells with zero expression get maximum ambient score
- **High quality cells**: Cells with low ambient correlation get low scores
- **Insufficient reference**: If < 10 cells, uses all cells as reference

**Data Guarantees (Preconditions)**

- `expression.rows() == ambient_scores.len`
- Expression matrix must be valid
- Must have at least some cells for reference estimation

**Complexity Analysis**

- **Time**: O(n_cells * n_genes + nnz) for profile and correlation computation
  - UMI computation: O(nnz)
  - Sorting: O(n_cells log n_cells)
  - Profile building: O(nnz_ref) where nnz_ref is non-zeros in reference
  - Correlation: O(n_cells * n_genes)
- **Space**: O(n_cells + n_genes) auxiliary for UMI counts and ambient profile

**Example**

```cpp
scl::Sparse<Real, true> expression = /* ... */;
scl::Array<Real> ambient_scores(n_cells);

scl::kernel::outlier::ambient_detection(expression, ambient_scores);

// Filter high-ambient cells
for (Index i = 0; i < n_cells; ++i) {
    if (ambient_scores[i] > config::AMBIENT_THRESHOLD) {
        // Cell i has high ambient RNA contamination
    }
}
```

---

### empty_drops

::: source_code file="scl/kernel/outlier.hpp" symbol="empty_drops" collapsed
:::

**Algorithm Description**

Identify empty droplets using deviance test against ambient profile (EmptyDrops-style algorithm):

1. **UMI Sorting**: Sort cells by total UMI count (ascending)

2. **Ambient Profile**: Estimate from lowest-UMI barcodes:
   - Use bottom barcodes as ambient reference
   - Build ambient profile = mean expression of reference
   - Normalize to probabilities

3. **Deviance Test**: For each cell above minimum UMI:
   - Compute expected counts from ambient profile
   - Compute deviance = sum(observed * log(observed/expected))
   - P-value from chi-squared distribution (Wilson-Hilferty approximation)

4. **FDR Correction**: Apply Benjamini-Hochberg correction:
   - Control false discovery rate
   - Mark cells failing test as empty

5. **Output**: Store boolean mask:
   - `is_empty[i] = true` if cell i is likely empty
   - Cells with UMI < EMPTY_DROPS_MIN_UMI automatically marked empty

**Edge Cases**

- **Very low UMI**: Cells with UMI < MIN_UMI automatically marked empty
- **High UMI cells**: Cells with very high UMI rarely marked empty
- **Ambient-like cells**: Cells matching ambient profile get marked empty
- **Insufficient data**: If too few cells, uses all as reference

**Data Guarantees (Preconditions)**

- `raw_counts.rows() == is_empty.len`
- `fdr_threshold` in (0, 1)
- Raw counts matrix must be valid

**Complexity Analysis**

- **Time**: O(n_cells * n_genes + n_cells log n_cells) for sorting and testing
  - UMI computation: O(nnz)
  - Sorting: O(n_cells log n_cells)
  - Deviance test: O(n_cells * n_genes)
  - FDR correction: O(n_cells log n_cells)
- **Space**: O(n_cells + n_genes) auxiliary for UMI counts and ambient profile

**Example**

```cpp
scl::Sparse<Real, true> raw_counts = /* ... */;  // Raw UMI counts
scl::Array<bool> is_empty(n_cells);

scl::kernel::outlier::empty_drops(
    raw_counts,
    is_empty,
    Real(0.01)  // FDR threshold
);

// Filter empty droplets
Index n_empty = 0;
for (Index i = 0; i < n_cells; ++i) {
    if (is_empty[i]) n_empty++;
}
```

---

### outlier_genes

::: source_code file="scl/kernel/outlier.hpp" symbol="outlier_genes" collapsed
:::

**Algorithm Description**

Identify genes with outlier dispersion characteristics based on median absolute deviation of log CV^2:

1. **Gene Statistics**: For each gene:
   - Compute mean expression across cells
   - Compute variance across cells
   - Skip genes with mean < epsilon

2. **Coefficient of Variation**: For each gene:
   - CV = std / mean
   - Log CV^2 = log(variance / mean^2)
   - Only for genes with mean > epsilon

3. **MAD Computation**: 
   - Compute median of log CV^2
   - Compute MAD (median absolute deviation) scaled by 1.4826
   - Used for robust outlier detection

4. **Z-score and Filtering**: For each gene:
   - Z-score = (log_CV^2 - median) / MAD
   - Flag genes with |z-score| > threshold as outliers

**Edge Cases**

- **Low expression genes**: Genes with mean < epsilon are skipped
- **Constant genes**: Genes with zero variance get undefined CV (skipped)
- **Extreme dispersion**: Genes with very high or very low CV get flagged
- **Few genes**: If too few genes, MAD may be unreliable

**Data Guarantees (Preconditions)**

- `outlier_gene_indices` has capacity >= expression.cols()
- `threshold > 0`
- Expression matrix must be valid

**Complexity Analysis**

- **Time**: O(nnz + n_genes log n_genes) for statistics and sorting
  - Gene statistics: O(nnz)
  - CV computation: O(n_genes)
  - MAD computation: O(n_genes log n_genes) for sorting
  - Filtering: O(n_genes)
- **Space**: O(n_genes) auxiliary for storing statistics

**Example**

```cpp
scl::Sparse<Real, true> expression = /* ... */;
Index* outlier_indices = /* allocate n_genes */;
Size n_outliers = 0;

scl::kernel::outlier::outlier_genes(
    expression,
    outlier_indices,
    n_outliers,
    Real(3.0)  // Z-score threshold
);

// outlier_indices[0..n_outliers-1] contains outlier gene indices
```

---

### doublet_score

::: source_code file="scl/kernel/outlier.hpp" symbol="doublet_score" collapsed
:::

**Algorithm Description**

Compute doublet scores based on expression dissimilarity from local neighborhood:

1. **Neighbor Analysis**: For each cell:
   - Get k nearest neighbors from KNN graph
   - Compute neighbor statistics per feature

2. **Z-score Computation**: For each cell and feature:
   - Compute mean and std of neighbor expression
   - Z-score = (cell_value - neighbor_mean) / neighbor_std
   - Measures deviation from local neighborhood

3. **Score Aggregation**: For each cell:
   - Score = mean absolute z-score across features
   - Higher scores indicate potential doublets

**Edge Cases**

- **No neighbors**: Cells with no neighbors get undefined score (handled)
- **Constant neighbors**: If neighbors have zero variance, z-score undefined
- **Isolated cells**: Cells far from others get high scores
- **Mixed neighborhoods**: Cells in mixed neighborhoods get moderate scores

**Data Guarantees (Preconditions)**

- `expression.rows() == neighbors.rows() == scores.len`
- KNN graph must be valid
- Expression matrix must be valid

**Complexity Analysis**

- **Time**: O(n_cells * k * avg_nnz_per_row) where k = neighbors per cell
  - Neighbor statistics: O(n_cells * k * n_genes)
  - Z-score computation: O(n_cells * n_genes)
- **Space**: O(k) auxiliary per cell for neighbor storage

**Example**

```cpp
scl::Sparse<Real, true> expression = /* ... */;
scl::Sparse<Index, true> neighbors = /* ... */;  // KNN graph
scl::Array<Real> doublet_scores(n_cells);

scl::kernel::outlier::doublet_score(
    expression,
    neighbors,
    doublet_scores
);

// Filter potential doublets
for (Index i = 0; i < n_cells; ++i) {
    if (doublet_scores[i] > 2.0) {
        // Cell i is a potential doublet
    }
}
```

---

### mitochondrial_outliers

::: source_code file="scl/kernel/outlier.hpp" symbol="mitochondrial_outliers" collapsed
:::

**Algorithm Description**

Identify cells with high mitochondrial gene content, typically indicating damaged or dying cells:

1. **Mito Fraction Computation**: For each cell:
   - Sum expression of mitochondrial genes
   - Sum total expression (all genes)
   - Mito fraction = mito_UMI / total_UMI

2. **Outlier Detection**: For each cell:
   - Compare mito fraction to threshold
   - Mark as outlier if fraction > threshold

3. **Output**: Store fractions and outlier mask:
   - `mito_fraction[i]` = mitochondrial fraction for cell i
   - `is_outlier[i]` = true if fraction > threshold

**Edge Cases**

- **No mito genes**: If no mito genes provided, all fractions = 0
- **Zero total UMI**: Cells with zero expression get fraction = 0
- **High mito cells**: Cells with > threshold get marked as outliers
- **Invalid gene indices**: Mito gene indices outside range are ignored

**Data Guarantees (Preconditions)**

- All arrays sized to expression.rows()
- Mito gene indices must be within [0, expression.cols())
- `threshold` in (0, 1)

**Complexity Analysis**

- **Time**: O(nnz + n_cells) for fraction computation
  - Mito gene lookup: O(nnz)
  - Fraction computation: O(n_cells)
- **Space**: O(max_gene_idx) for gene lookup table

**Example**

```cpp
scl::Sparse<Real, true> expression = /* ... */;
scl::Array<const Index> mito_genes = /* ... */;  // Mitochondrial gene indices
scl::Array<Real> mito_fraction(n_cells);
scl::Array<bool> is_outlier(n_cells);

scl::kernel::outlier::mitochondrial_outliers(
    expression,
    mito_genes,
    mito_fraction,
    is_outlier,
    Real(0.2)  // 20% threshold
);

// Filter high-mito cells
for (Index i = 0; i < n_cells; ++i) {
    if (is_outlier[i]) {
        // Cell i has high mitochondrial content
    }
}
```

---

### qc_filter

::: source_code file="scl/kernel/outlier.hpp" symbol="qc_filter" collapsed
:::

**Algorithm Description**

Apply combined QC filtering based on gene count, UMI count, and mitochondrial fraction thresholds:

1. **Per-cell Metrics**: For each cell:
   - Count detected genes (non-zero entries)
   - Sum total counts (UMI)
   - Compute mitochondrial fraction

2. **Threshold Checking**: For each cell:
   - Check gene count in [min_genes, max_genes]
   - Check UMI count in [min_counts, max_counts]
   - Check mito fraction <= max_mito_fraction

3. **QC Pass**: Cell passes if all criteria satisfied:
   - `pass_qc[i] = true` if all thresholds passed
   - `pass_qc[i] = false` otherwise

**Edge Cases**

- **Too few genes**: Cells with < min_genes fail QC
- **Too many genes**: Cells with > max_genes fail QC (potential doublets)
- **Low UMI**: Cells with < min_counts fail QC
- **High UMI**: Cells with > max_counts fail QC (potential doublets)
- **High mito**: Cells with > max_mito_fraction fail QC

**Data Guarantees (Preconditions)**

- `pass_qc.len == expression.rows()`
- `min_genes <= max_genes`
- `min_counts <= max_counts`
- `max_mito_fraction` in [0, 1]

**Complexity Analysis**

- **Time**: O(nnz + n_cells) for metric computation and filtering
  - Gene counting: O(nnz)
  - UMI summing: O(nnz)
  - Mito fraction: O(nnz)
  - Filtering: O(n_cells)
- **Space**: O(max_gene_idx) for mito gene lookup

**Example**

```cpp
scl::Sparse<Real, true> expression = /* ... */;
scl::Array<const Index> mito_genes = /* ... */;
scl::Array<bool> pass_qc(n_cells);

scl::kernel::outlier::qc_filter(
    expression,
    Real(200),   // min_genes
    Real(5000),   // max_genes
    Real(1000),   // min_counts
    Real(50000),  // max_counts
    Real(0.2),    // max_mito_fraction
    mito_genes,
    pass_qc
);

// Filter to passing cells
Index n_pass = 0;
for (Index i = 0; i < n_cells; ++i) {
    if (pass_qc[i]) n_pass++;
}
```

---

## Configuration

Default parameters in `scl::kernel::outlier::config`:

- `EPSILON = 1e-10`: Small constant for numerical stability
- `MIN_K_NEIGHBORS = 5`: Minimum neighbors for LOF
- `DEFAULT_K = 20`: Default number of neighbors
- `LOF_THRESHOLD = 1.5`: LOF threshold for outlier detection
- `AMBIENT_THRESHOLD = 0.1`: Ambient RNA threshold
- `EMPTY_DROPS_MIN_UMI = 100`: Minimum UMI for empty drops test
- `EMPTY_DROPS_MAX_AMBIENT = 10`: Maximum ambient for empty drops
- `MONTE_CARLO_ITERATIONS = 10000`: Iterations for Monte Carlo tests
- `PARALLEL_THRESHOLD = 256`: Minimum size for parallel processing

---

## Performance Notes

### Sequential Implementation

Most functions are sequential (not parallelized) due to:
- Complex dependencies between cells
- Need for global statistics
- Atomic operations in some cases

### Memory Efficiency

- Pre-allocated output buffers
- Efficient sparse matrix access
- Minimal temporary allocations

---

## See Also

- [Quality Control](../qc)
- [Neighbors](../neighbors)
- [Statistics](../stat)
