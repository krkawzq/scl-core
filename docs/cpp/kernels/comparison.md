# comparison.hpp

> scl/kernel/comparison.hpp · Group comparison and differential abundance analysis

## Overview

This file provides statistical kernels for comparing groups, testing differential abundance, and computing effect sizes. It supports composition analysis, abundance testing, multi-sample differential abundance (DAseq/Milo-style), and condition response analysis.

**Header**: `#include "scl/kernel/comparison.hpp"`

---

## Main APIs

### composition_analysis

::: source_code file="scl/kernel/comparison.hpp" symbol="composition_analysis" collapsed
:::

**Algorithm Description**

Analyzes cell type composition across conditions using chi-squared test:

1. Count cells per type per condition in a single pass
2. Compute proportions: `proportions[t * n_conditions + c] = count[t,c] / total_cells[c]`
3. For each cell type:
   - Compute expected counts under null hypothesis (equal proportions)
   - Compute chi-squared statistic: sum((observed - expected)^2 / expected)
   - Convert to p-value using Wilson-Hilferty approximation for better accuracy
4. Parallelized over cell types for efficiency

**Edge Cases**

- **Empty conditions**: Conditions with zero cells produce NaN proportions
- **Single cell type**: If only one type exists, p-value is 1.0 (no variation)
- **Zero counts**: Types with zero cells in all conditions produce NaN p-values

**Data Guarantees (Preconditions)**

- `cell_types.len == conditions.len == n_cells`
- `proportions` has capacity >= `n_types * n_conditions`
- `p_values` has capacity >= n_types
- All cell type indices < n_types
- All condition indices < n_conditions

**Complexity Analysis**

- **Time**: O(n_cells + n_types * n_conditions) - linear scan plus pairwise computation
- **Space**: O(n_types * n_conditions) auxiliary - count matrix

**Example**

```cpp
#include "scl/kernel/comparison.hpp"

Array<const Index> cell_types = /* cell type labels [n_cells] */;
Array<const Index> conditions = /* condition labels [n_cells] */;
Real* proportions = new Real[n_types * n_conditions];
Real* p_values = new Real[n_types];

scl::kernel::comparison::composition_analysis(
    cell_types,
    conditions,
    proportions,
    p_values,
    n_types,
    n_conditions
);

// Check significant composition changes
for (Size t = 0; t < n_types; ++t) {
    if (p_values[t] < 0.05) {
        // Type t shows significant composition change across conditions
    }
}
```

---

### abundance_test

::: source_code file="scl/kernel/comparison.hpp" symbol="abundance_test" collapsed
:::

**Algorithm Description**

Tests differential abundance of clusters between two conditions using Fisher's exact test:

1. Count cells per cluster per condition (condition must be binary: 0 or 1)
2. Compute proportions: `prop[c, cond] = count[c, cond] / total_cells[cond]`
3. For each cluster:
   - Compute log2 fold change: `log2(prop[c, 1] / prop[c, 0])`
   - Build 2x2 contingency table: [count[c,0], count[other,0]; count[c,1], count[other,1]]
   - Compute Fisher's exact test p-value using chi-squared approximation
4. Parallelized over clusters

**Edge Cases**

- **Zero proportion in condition 0**: Fold change is +infinity
- **Zero proportion in condition 1**: Fold change is -infinity
- **Both zero**: Fold change is NaN, p-value is 1.0
- **Small counts**: Chi-squared approximation valid for counts >= 5

**Data Guarantees (Preconditions)**

- `cluster_labels.len == condition.len`
- `fold_changes` has capacity >= n_clusters
- `p_values` has capacity >= n_clusters
- Condition labels are strictly 0 or 1 (binary)

**Complexity Analysis**

- **Time**: O(n_cells + n_clusters) - single pass plus cluster-wise computation
- **Space**: O(n_clusters) auxiliary - count arrays

**Example**

```cpp
Array<const Index> cluster_labels = /* cluster assignment [n_cells] */;
Array<const Index> condition = /* condition labels (0 or 1) [n_cells] */;
Array<Real> fold_changes(n_clusters);
Array<Real> p_values(n_clusters);

scl::kernel::comparison::abundance_test(
    cluster_labels,
    condition,
    fold_changes,
    p_values
);

// Find significantly enriched clusters
for (Index c = 0; c < n_clusters; ++c) {
    if (p_values[c] < 0.05 && fold_changes[c] > 1.0) {
        // Cluster c is significantly enriched in condition 1
    }
}
```

---

### differential_abundance

::: source_code file="scl/kernel/comparison.hpp" symbol="differential_abundance" collapsed
:::

**Algorithm Description**

Tests differential abundance across samples using Wilcoxon rank-sum test (DAseq/Milo-style):

1. Map each sample to its condition
2. Count cells per cluster per sample
3. Compute proportions per sample: `prop[s, c] = count[s, c] / total_cells[s]`
4. For each cluster:
   - Collect proportions for condition 0 and condition 1 separately
   - Compute log2 fold change: `log2(mean(prop[cond1]) / mean(prop[cond0]))`
   - Perform Wilcoxon rank-sum test on proportions between conditions
5. Uses workspace pools for thread-local buffers

**Edge Cases**

- **Single sample per condition**: Cannot compute Wilcoxon test, returns NaN
- **Zero cells in sample**: Sample excluded from analysis
- **All samples same condition**: Returns NaN p-value
- **Tied ranks**: Uses standard tie correction in Wilcoxon test

**Data Guarantees (Preconditions)**

- `cluster_labels.len == sample_ids.len == conditions.len`
- `da_scores` has capacity >= n_clusters
- `p_values` has capacity >= n_clusters
- At least 2 conditions and 2 samples required

**Complexity Analysis**

- **Time**: O(n_cells + n_clusters * n_samples) - counting plus per-cluster statistics
- **Space**: O(n_clusters * n_samples) auxiliary - proportion matrix

**Example**

```cpp
Array<const Index> cluster_labels = /* cluster labels [n_cells] */;
Array<const Index> sample_ids = /* sample IDs [n_cells] */;
Array<const Index> conditions = /* condition labels [n_cells] */;
Array<Real> da_scores(n_clusters);
Array<Real> p_values(n_clusters);

scl::kernel::comparison::differential_abundance(
    cluster_labels,
    sample_ids,
    conditions,
    da_scores,
    p_values
);

// Filter significant DA clusters
for (Index c = 0; c < n_clusters; ++c) {
    if (p_values[c] < 0.05 && std::abs(da_scores[c]) > 1.0) {
        // Cluster c shows significant differential abundance
    }
}
```

---

### condition_response

::: source_code file="scl/kernel/comparison.hpp" symbol="condition_response" collapsed
:::

**Algorithm Description**

Tests gene expression response between conditions using Wilcoxon rank-sum test:

1. For each gene in parallel:
   - Extract expression values for condition 0 using binary search in sparse matrix
   - Extract expression values for condition 1 using binary search
   - Compute mean expression per condition
   - Compute log2 fold change: `log2(mean[cond1] / mean[cond0])`
   - Perform Wilcoxon rank-sum test on expression values between conditions
2. Uses workspace pools for efficient sparse matrix access
3. Binary search optimization for sparse matrix row access

**Edge Cases**

- **Zero expression in both conditions**: Fold change is NaN, p-value is 1.0
- **Zero expression in one condition**: Fold change is +/-infinity
- **Sparse genes**: Genes with very few non-zero values may have unreliable statistics
- **Tied values**: Standard tie correction applied in Wilcoxon test

**Data Guarantees (Preconditions)**

- `expression.rows() == conditions.len`
- `response_scores` has capacity >= n_genes
- `p_values` has capacity >= n_genes
- At least 2 conditions required
- Expression matrix is valid CSR format

**Complexity Analysis**

- **Time**: O(n_genes * n_cells * log(nnz_per_cell)) - binary search per gene per cell
- **Space**: O(n_cells) auxiliary per thread - workspace for expression extraction

**Example**

```cpp
Sparse<Real, true> expression = /* cells x genes, CSR */;
Array<const Index> conditions = /* condition labels [n_cells] */;
Real* response_scores = new Real[n_genes];
Real* p_values = new Real[n_genes];

scl::kernel::comparison::condition_response(
    expression,
    conditions,
    response_scores,
    p_values,
    n_genes
);

// Find significantly responsive genes
for (Index g = 0; g < n_genes; ++g) {
    if (p_values[g] < 0.05 && std::abs(response_scores[g]) > 1.0) {
        // Gene g shows significant response to condition change
    }
}
```

---

## Utility Functions

### effect_size

Computes Cohen's d effect size between two groups.

::: source_code file="scl/kernel/comparison.hpp" symbol="effect_size" collapsed
:::

**Complexity**

- Time: O(n1 + n2)
- Space: O(1) auxiliary

---

### glass_delta

Computes Glass's delta effect size using control group standard deviation.

::: source_code file="scl/kernel/comparison.hpp" symbol="glass_delta" collapsed
:::

**Complexity**

- Time: O(n_control + n_treatment)
- Space: O(1) auxiliary

---

### hedges_g

Computes Hedges' g bias-corrected effect size.

::: source_code file="scl/kernel/comparison.hpp" symbol="hedges_g" collapsed
:::

**Complexity**

- Time: O(n1 + n2)
- Space: O(1) auxiliary

---

## Configuration

Default parameters in `scl::kernel::comparison::config`:

- `EPSILON = 1e-10`: Numerical stability constant
- `MIN_CELLS_PER_GROUP = 3`: Minimum cells required per group for reliable statistics
- `PERMUTATION_COUNT = 1000`: Default permutation count (if used)
- `PARALLEL_THRESHOLD = 32`: Minimum size for parallel processing

---

## See Also

- [Communication Module](./communication) - L-R interaction analysis
- [Multiple Testing Module](./multiple_testing) - P-value correction
