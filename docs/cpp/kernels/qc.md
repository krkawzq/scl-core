# Quality Control

Quality control metrics computation with SIMD optimization for single-cell data.

## Overview

The `qc` module provides efficient computation of quality control metrics commonly used in single-cell analysis:

- **Gene counts**: Number of expressed genes per cell
- **Total counts**: Total UMI counts per cell
- **Subset percentages**: Percentage of counts from specific gene subsets (e.g., mitochondrial genes)

All operations are:
- SIMD-accelerated with fused operations
- Parallelized over cells
- Zero-allocation (output arrays pre-allocated)

## Functions

### compute_basic_qc

Compute basic quality control metrics: number of genes and total counts per cell.

```cpp
#include "scl/kernel/qc.hpp"

Sparse<Real, true> matrix = /* expression matrix [n_cells x n_genes] */;
Array<Index> n_genes(matrix.rows());
Array<Real> total_counts(matrix.rows());

scl::kernel::qc::compute_basic_qc(matrix, n_genes, total_counts);
```

**Parameters:**
- `matrix` [in] - Expression matrix (cells x genes, CSR)
- `out_n_genes` [out] - Number of expressed genes per cell [n_cells]
- `out_total_counts` [out] - Total UMI counts per cell [n_cells]

**Preconditions:**
- `out_n_genes.len == matrix.rows()`
- `out_total_counts.len == matrix.rows()`
- Matrix must be valid CSR format

**Postconditions:**
- `out_n_genes[i]` contains number of non-zero genes in cell i
- `out_total_counts[i]` contains sum of all counts in cell i
- Matrix is unchanged

**Complexity:**
- Time: O(nnz)
- Space: O(1) auxiliary

**Thread Safety:** Safe - parallelized over cells

**Algorithm:**
For each cell in parallel:
1. Count non-zero elements (number of genes)
2. Sum all values using SIMD-optimized sum
3. Write results to output arrays

### compute_subset_pct

Compute percentage of total counts that come from a subset of genes (e.g., mitochondrial genes) for each cell.

```cpp
Array<const uint8_t> mito_mask(n_genes);  // 1 for mitochondrial genes
Array<Real> mito_pct(n_cells);

scl::kernel::qc::compute_subset_pct(matrix, mito_mask, mito_pct);
```

**Parameters:**
- `matrix` [in] - Expression matrix (cells x genes, CSR)
- `subset_mask` [in] - Mask array, non-zero indicates subset gene [n_genes]
- `out_pcts` [out] - Percentage values [n_cells]

**Preconditions:**
- `out_pcts.len == matrix.rows()`
- `subset_mask.len >= matrix.cols()`
- Matrix must be valid CSR format

**Postconditions:**
- `out_pcts[i]` contains percentage (0-100) of counts from subset in cell i
- Returns 0.0 if total counts are zero
- Matrix is unchanged

**Complexity:**
- Time: O(nnz)
- Space: O(1) auxiliary

**Thread Safety:** Safe - parallelized over cells

**Algorithm:**
For each cell in parallel:
1. Compute total counts and subset counts using fused SIMD operation
2. Compute percentage = (subset / total) * 100
3. Write result to output

### compute_fused_qc

Compute all QC metrics in a single pass: gene counts, total counts, and subset percentages.

```cpp
Array<const uint8_t> mito_mask(n_genes);
Array<Index> n_genes(n_cells);
Array<Real> total_counts(n_cells);
Array<Real> mito_pct(n_cells);

scl::kernel::qc::compute_fused_qc(
    matrix, mito_mask, n_genes, total_counts, mito_pct
);
```

**Parameters:**
- `matrix` [in] - Expression matrix (cells x genes, CSR)
- `subset_mask` [in] - Mask array for subset genes [n_genes]
- `out_n_genes` [out] - Number of expressed genes per cell [n_cells]
- `out_total_counts` [out] - Total UMI counts per cell [n_cells]
- `out_pcts` [out] - Subset percentages per cell [n_cells]

**Preconditions:**
- All output arrays have length == matrix.rows()
- `subset_mask.len >= matrix.cols()`
- Matrix must be valid CSR format

**Postconditions:**
- All metrics computed for each cell
- Matrix is unchanged

**Complexity:**
- Time: O(nnz)
- Space: O(1) auxiliary

**Thread Safety:** Safe - parallelized over cells

**Algorithm:**
For each cell in parallel:
1. Count non-zero elements
2. Compute total and subset counts using fused SIMD operation
3. Compute percentage
4. Write all results to output arrays

## Configuration

```cpp
namespace scl::kernel::qc::config {
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Real PCT_SCALE = Real(100);
}
```

## Use Cases

### Standard QC Pipeline

```cpp
// Load expression matrix
Sparse<Real, true> expression = /* ... */;

// Create mitochondrial gene mask
Array<uint8_t> mito_mask(n_genes, 0);
for (Index g = 0; g < n_genes; ++g) {
    if (gene_names[g].starts_with("MT-")) {
        mito_mask.ptr[g] = 1;
    }
}

// Compute all QC metrics in one pass
Array<Index> n_genes(n_cells);
Array<Real> total_counts(n_cells);
Array<Real> mito_pct(n_cells);

scl::kernel::qc::compute_fused_qc(
    expression, mito_mask, n_genes, total_counts, mito_pct
);

// Filter cells based on QC metrics
for (Index i = 0; i < n_cells; ++i) {
    if (n_genes.ptr[i] < 200 || total_counts.ptr[i] < 1000 || 
        mito_pct.ptr[i] > 20.0) {
        // Mark cell for removal
    }
}
```

### Individual Metric Computation

```cpp
// Compute only basic metrics
Array<Index> n_genes(n_cells);
Array<Real> total_counts(n_cells);
scl::kernel::qc::compute_basic_qc(expression, n_genes, total_counts);

// Compute only subset percentage
Array<Real> mito_pct(n_cells);
scl::kernel::qc::compute_subset_pct(expression, mito_mask, mito_pct);
```

## Performance

- **Fused operations**: Single pass for multiple metrics reduces memory traffic
- **SIMD acceleration**: 4-way unrolled accumulation for maximum throughput
- **Parallelization**: Scales linearly with CPU cores
- **Zero allocations**: All output arrays must be pre-allocated

---

::: tip Performance Tip
Use `compute_fused_qc` when you need multiple metrics - it's faster than calling individual functions separately.
:::

