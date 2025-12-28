# qc.hpp

> scl/kernel/qc.hpp · Quality control metrics with SIMD optimization

## Overview

This file provides high-performance quality control metric computation for single-cell expression data, including gene counts, total counts, and subset percentages.

This file provides:
- Basic QC metrics (gene counts, total counts)
- Subset percentage computation (e.g., mitochondrial genes)
- Fused QC computation (all metrics in single pass)
- SIMD-optimized operations

**Header**: `#include "scl/kernel/qc.hpp"`

---

## Main APIs

### compute_basic_qc

::: source_code file="scl/kernel/qc.hpp" symbol="compute_basic_qc" collapsed
:::

**Algorithm Description**

Compute basic quality control metrics: number of genes and total counts per cell:

1. **Parallel Processing**: Process each cell in parallel:
   - Each thread handles independent cells
   - No synchronization needed

2. **Per-Cell Metrics**: For each cell:
   - **Gene Count**: Count non-zero elements in row
     - Iterate over non-zero values
     - Count distinct genes (non-zero entries)
   - **Total Counts**: Sum all values using SIMD-optimized sum
     - Use vectorized accumulation
     - Handle sparse zeros efficiently

3. **Output**: Store metrics in output arrays:
   - `out_n_genes[i]` = number of expressed genes in cell i
   - `out_total_counts[i]` = sum of all counts in cell i

**Edge Cases**

- **Empty cells**: Cells with no expression get n_genes = 0, total_counts = 0
- **Dense cells**: Cells with many expressed genes handled efficiently
- **Zero values**: Sparse format may not store zeros (handled correctly)
- **Very sparse cells**: Minimal overhead for cells with few genes

**Data Guarantees (Preconditions)**

- `out_n_genes.len == matrix.rows()`
- `out_total_counts.len == matrix.rows()`
- Matrix must be valid CSR format
- Output arrays must be pre-allocated

**Complexity Analysis**

- **Time**: O(nnz) for iterating over all non-zeros
  - Each non-zero accessed once
  - Parallelized over cells
  - SIMD reduces constant factor
- **Space**: O(1) auxiliary space per thread

**Example**

```cpp
#include "scl/kernel/qc.hpp"

scl::Sparse<Real, true> expression = /* ... */;  // [n_cells x n_genes]
scl::Array<Index> n_genes(n_cells);
scl::Array<Real> total_counts(n_cells);

scl::kernel::qc::compute_basic_qc(
    expression,
    n_genes,
    total_counts
);

// n_genes[i] contains number of expressed genes in cell i
// total_counts[i] contains total UMI count in cell i
```

---

### compute_subset_pct

::: source_code file="scl/kernel/qc.hpp" symbol="compute_subset_pct" collapsed
:::

**Algorithm Description**

Compute percentage of total counts that come from a subset of genes (e.g., mitochondrial genes):

1. **Parallel Processing**: Process each cell in parallel

2. **Fused Computation**: For each cell:
   - **Total Counts**: Sum all values (SIMD-optimized)
   - **Subset Counts**: Sum values where mask[gene] != 0
     - Check mask for each non-zero gene
     - Accumulate subset counts separately
   - **Percentage**: Compute (subset / total) * 100
     - Handle zero total counts (return 0.0)

3. **Output**: Store percentages in output array:
   - `out_pcts[i]` = percentage (0-100) of counts from subset in cell i

**Edge Cases**

- **Zero total counts**: Returns 0.0 (avoid division by zero)
- **All subset**: If all genes in subset, percentage = 100.0
- **No subset**: If no genes in subset, percentage = 0.0
- **Empty cells**: Cells with no expression get percentage = 0.0

**Data Guarantees (Preconditions)**

- `out_pcts.len == matrix.rows()`
- `subset_mask.len >= matrix.cols()`
- Matrix must be valid CSR format
- Mask values: 0 = not in subset, non-zero = in subset

**Complexity Analysis**

- **Time**: O(nnz) for checking mask and summing
  - Each non-zero checked against mask
  - Parallelized over cells
  - SIMD for efficient accumulation
- **Space**: O(1) auxiliary space

**Example**

```cpp
scl::Sparse<Real, true> expression = /* ... */;
scl::Array<const uint8_t> mito_mask(n_genes);  // Mitochondrial gene mask

// Set mask: 1 for mitochondrial genes, 0 otherwise
for (Index g = 0; g < n_genes; ++g) {
    if (is_mitochondrial(g)) {
        mito_mask[g] = 1;
    }
}

scl::Array<Real> mito_pct(n_cells);

scl::kernel::qc::compute_subset_pct(
    expression,
    mito_mask,
    mito_pct
);

// mito_pct[i] contains percentage of mitochondrial counts in cell i
```

---

### compute_fused_qc

::: source_code file="scl/kernel/qc.hpp" symbol="compute_fused_qc" collapsed
:::

**Algorithm Description**

Compute all QC metrics in a single pass: gene counts, total counts, and subset percentages:

1. **Parallel Processing**: Process each cell in parallel

2. **Fused Computation**: For each cell in single loop:
   - **Gene Count**: Count non-zero elements
   - **Total Counts**: Sum all values (SIMD)
   - **Subset Counts**: Sum values where mask != 0 (SIMD)
   - **Percentage**: Compute (subset / total) * 100

3. **Output**: Store all metrics in output arrays:
   - `out_n_genes[i]` = number of expressed genes
   - `out_total_counts[i]` = total UMI counts
   - `out_pcts[i]` = subset percentage

**Edge Cases**

- **Same as individual functions**: Handles all edge cases
- **Efficiency**: Single pass more efficient than multiple passes
- **Memory**: Better cache locality than separate calls

**Data Guarantees (Preconditions)**

- All output arrays have length == matrix.rows()
- `subset_mask.len >= matrix.cols()`
- Matrix must be valid CSR format

**Complexity Analysis**

- **Time**: O(nnz) for single pass over all non-zeros
  - More efficient than calling functions separately
  - Parallelized over cells
- **Space**: O(1) auxiliary space

**Example**

```cpp
scl::Sparse<Real, true> expression = /* ... */;
scl::Array<const uint8_t> mito_mask(n_genes);
scl::Array<Index> n_genes(n_cells);
scl::Array<Real> total_counts(n_cells);
scl::Array<Real> mito_pct(n_cells);

// Compute all QC metrics in single pass
scl::kernel::qc::compute_fused_qc(
    expression,
    mito_mask,
    n_genes,
    total_counts,
    mito_pct
);

// All metrics computed efficiently
```

---

## Configuration

Default parameters in `scl::kernel::qc::config`:

- `PREFETCH_DISTANCE = 16`: Cache line prefetch distance
- `PCT_SCALE = 100`: Scale factor for percentage (0-100)

---

## Performance Notes

### SIMD Optimization

- All operations use SIMD for vectorized accumulation
- Fused operations reduce memory access
- Efficient sparse matrix traversal

### Parallelization

- All functions parallelize over cells
- No synchronization needed (distinct output elements)
- Scales with hardware concurrency

---

## Use Cases

### Basic QC Filtering

```cpp
// Compute basic metrics
scl::Array<Index> n_genes(n_cells);
scl::Array<Real> total_counts(n_cells);
scl::kernel::qc::compute_basic_qc(expression, n_genes, total_counts);

// Filter cells
for (Index i = 0; i < n_cells; ++i) {
    if (n_genes[i] < 200 || total_counts[i] < 1000) {
        // Low quality cell
    }
}
```

### Mitochondrial Filtering

```cpp
// Compute mitochondrial percentage
scl::Array<Real> mito_pct(n_cells);
scl::kernel::qc::compute_subset_pct(expression, mito_mask, mito_pct);

// Filter high-mito cells
for (Index i = 0; i < n_cells; ++i) {
    if (mito_pct[i] > 20.0) {
        // High mitochondrial content (damaged cell)
    }
}
```

### Comprehensive QC

```cpp
// Compute all metrics in single pass
scl::kernel::qc::compute_fused_qc(
    expression, mito_mask,
    n_genes, total_counts, mito_pct
);

// Apply comprehensive filtering
for (Index i = 0; i < n_cells; ++i) {
    bool pass = (n_genes[i] >= 200) &&
                (n_genes[i] <= 5000) &&
                (total_counts[i] >= 1000) &&
                (total_counts[i] <= 50000) &&
                (mito_pct[i] <= 20.0);
    // Use pass flag
}
```

---

## See Also

- [Outlier Detection](../outlier)
- [Normalization](../normalization)
- [Sparse Matrices](../core/sparse)
