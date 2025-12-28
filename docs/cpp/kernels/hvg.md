# hvg.hpp

> scl/kernel/hvg.hpp · Highly variable gene selection kernels

## Overview

This file provides efficient methods for selecting highly variable genes (HVGs) in single-cell RNA-seq analysis. HVG selection is a critical preprocessing step that identifies genes with high biological variability for downstream analysis.

This file provides:
- Dispersion-based gene selection (variance/mean ratio)
- Variance-stabilizing transformation (VST) method
- SIMD-accelerated computation
- Partial sorting for efficient top-k selection

**Header**: `#include "scl/kernel/hvg.hpp"`

---

## Main APIs

### select_by_dispersion

::: source_code file="scl/kernel/hvg.hpp" symbol="select_by_dispersion" collapsed
:::

**Algorithm Description**

Select highly variable genes by dispersion (variance/mean ratio):

1. **Compute moments**: For each gene in parallel:
   - Compute mean: mean[g] = sum(expression[g, :]) / n_cells
   - Compute variance: var[g] = sum((expression[g, :] - mean[g])^2) / (n_cells - ddof)
   - Uses SIMD-optimized vectorized operations

2. **Compute dispersion**: For each gene:
   - dispersion[g] = var[g] / mean[g] if mean[g] > epsilon
   - dispersion[g] = 0 if mean[g] <= epsilon (avoid division by zero)
   - Uses 4-way SIMD unroll with prefetch

3. **Select top k**: Use partial sort to select n_top genes with highest dispersion:
   - O(n_genes + n_top * log(n_top)) complexity
   - Outputs indices and binary mask

**Edge Cases**

- **Zero mean genes**: Genes with mean <= epsilon have dispersion = 0 (excluded)
- **Constant genes**: Genes with zero variance have dispersion = 0
- **Empty matrix**: Returns empty selection if matrix has no non-zeros
- **n_top > n_genes**: Clamped to n_genes, all genes selected

**Data Guarantees (Preconditions)**

- `out_indices.len >= n_top`
- `out_mask.len >= n_genes`
- `out_dispersions.len >= n_genes`
- Matrix must be valid CSR/CSC format

**Complexity Analysis**

- **Time**: O(nnz + n_genes * log(n_top))
  - O(nnz) for computing moments
  - O(n_genes) for dispersion computation
  - O(n_genes + n_top * log(n_top)) for partial sort
- **Space**: O(n_genes) for intermediate buffers (means, variances, dispersions)

**Example**

```cpp
#include "scl/kernel/hvg.hpp"

// Expression matrix: genes x cells
Sparse<Real, true> expression = /* ... */;
Index n_genes = expression.rows();
Size n_top = 2000;

// Pre-allocate output
Array<Index> selected_indices(n_top);
Array<uint8_t> mask(n_genes, 0);
Array<Real> dispersions(n_genes);

// Select top 2000 highly variable genes
scl::kernel::hvg::select_by_dispersion(
    expression,
    n_top,
    selected_indices,
    mask,
    dispersions
);

// Use selected genes
for (Size i = 0; i < n_top; ++i) {
    Index gene_idx = selected_indices[i];
    Real dispersion = dispersions[gene_idx];
    // Process highly variable gene
}
```

---

### select_by_vst

::: source_code file="scl/kernel/hvg.hpp" symbol="select_by_vst" collapsed
:::

**Algorithm Description**

Select highly variable genes using variance-stabilizing transformation (VST) method:

1. **Clip values**: For each gene g:
   - Clip expression values to clip_vals[g] before computing variance
   - Prevents high-expression outlier genes from dominating selection

2. **Compute clipped moments**: For each gene in parallel:
   - Compute mean and variance after clipping
   - Uses SIMD-optimized accumulation

3. **Select top k**: Partial sort to select n_top genes with highest clipped variance

**Edge Cases**

- **Zero clip values**: If clip_val[g] = 0, all values clipped to 0, variance = 0
- **Very large clip values**: If clip_val >> max(expression), no clipping occurs
- **Empty matrix**: Returns empty selection

**Data Guarantees (Preconditions)**

- `clip_vals.len >= n_genes`
- `out_indices.len >= n_top`
- `out_mask.len >= n_genes`
- `out_variances.len >= n_genes`

**Complexity Analysis**

- **Time**: O(nnz + n_genes * log(n_top))
  - O(nnz) for clipped moment computation
  - O(n_genes + n_top * log(n_top)) for partial sort
- **Space**: O(n_genes) for intermediate buffers

**Example**

```cpp
// Compute clip values (e.g., from previous analysis)
Array<Real> clip_vals(n_genes);
// ... compute clip values per gene ...

Array<Index> selected_indices(n_top);
Array<uint8_t> mask(n_genes);
Array<Real> variances(n_genes);

scl::kernel::hvg::select_by_vst(
    expression,
    clip_vals,
    n_top,
    selected_indices,
    mask,
    variances
);
```

---

## Utility Functions

### detail::dispersion_simd

Compute dispersion = var / mean with SIMD optimization.

::: source_code file="scl/kernel/hvg.hpp" symbol="detail::dispersion_simd" collapsed
:::

**Complexity**

- Time: O(n)
- Space: O(1)

---

### detail::normalize_dispersion_simd

Z-score normalize dispersions within a mean range.

::: source_code file="scl/kernel/hvg.hpp" symbol="detail::normalize_dispersion_simd" collapsed
:::

**Complexity**

- Time: O(n)
- Space: O(1)

---

### detail::select_top_k_partial

Select top k elements using partial sort.

::: source_code file="scl/kernel/hvg.hpp" symbol="detail::select_top_k_partial" collapsed
:::

**Complexity**

- Time: O(n + k log k)
- Space: O(k)

---

### detail::compute_moments

Compute mean and variance for each gene.

::: source_code file="scl/kernel/hvg.hpp" symbol="detail::compute_moments" collapsed
:::

**Complexity**

- Time: O(nnz)
- Space: O(n_genes)

---

### detail::compute_clipped_moments

Compute mean and variance with per-gene value clipping.

::: source_code file="scl/kernel/hvg.hpp" symbol="detail::compute_clipped_moments" collapsed
:::

**Complexity**

- Time: O(nnz)
- Space: O(n_genes)

---

## Notes

**Dispersion vs VST**:
- Dispersion method: Simple variance/mean ratio, fast and effective
- VST method: Clips high values before variance computation, more robust to outliers

**Performance**:
- SIMD-accelerated for mean/variance computation
- Partial sorting for efficient top-k selection
- Parallelized over genes

**Typical Usage**:
- Select 2000-3000 highly variable genes for downstream analysis
- Use dispersion for standard workflows
- Use VST when dealing with highly expressed outlier genes

## See Also

- [Feature Selection](/cpp/kernels/feature) - Additional feature selection methods
- [Statistics](/cpp/kernels/statistics) - Statistical operations
