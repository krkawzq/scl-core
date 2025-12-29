---
title: Highly Variable Genes
description: Selection of highly variable genes for downstream analysis
---

# Highly Variable Genes

The `hvg` kernel provides efficient selection of highly variable genes (HVGs) using dispersion-based methods, optimized with SIMD and parallelization.

## Overview

Highly variable gene selection is essential for:
- Reducing dimensionality before downstream analysis
- Focusing on informative genes
- Improving clustering and visualization quality

The implementation uses dispersion (variance/mean) as the variability metric.

## Functions

### `compute_dispersion`

Compute dispersion (variance/mean) for each gene.

```cpp
template <typename T, bool IsCSR>
void compute_dispersion(
    const Sparse<T, IsCSR>& matrix,
    Array<const Real> means,
    Array<const Real> vars,
    Array<Real> out_dispersion
);
```

**Parameters**:
- `matrix` [in]: Expression matrix (cells × genes)
- `means` [in]: Mean expression per gene
- `vars` [in]: Variance per gene
- `out_dispersion` [out]: Dispersion values (length = matrix.cols())

**Mathematical Operation**: `dispersion = variance / mean` (for mean > ε)

**Example**:
```cpp
#include "scl/kernel/hvg.hpp"

// Compute statistics
auto means = memory::aligned_alloc<Real>(matrix.cols());
auto vars = memory::aligned_alloc<Real>(matrix.cols());
Array<Real> means_view = {means.get(), static_cast<Size>(matrix.cols())};
Array<Real> vars_view = {vars.get(), static_cast<Size>(matrix.cols())};

compute_gene_statistics(matrix, means_view, vars_view);

// Compute dispersion
auto dispersions = memory::aligned_alloc<Real>(matrix.cols());
Array<Real> disp_view = {dispersions.get(), static_cast<Size>(matrix.cols())};
kernel::hvg::compute_dispersion(matrix, means_view, vars_view, disp_view);
```

### `normalize_dispersion`

Normalize dispersion values using z-score.

```cpp
void normalize_dispersion(
    Array<Real> dispersions,
    Real min_mean,
    Real max_mean,
    Array<const Real> means
);
```

**Parameters**:
- `dispersions` [in,out]: Dispersion values (modified in-place)
- `min_mean` [in]: Minimum mean expression threshold
- `max_mean` [in]: Maximum mean expression threshold
- `means` [in]: Mean expression values

**Operation**: Z-score normalization of dispersions for genes within mean range

**Example**:
```cpp
// Normalize dispersions
Real min_mean = 0.01;
Real max_mean = 3.0;
kernel::hvg::normalize_dispersion(disp_view, min_mean, max_mean, means_view);
```

### `select_hvg`

Select top N highly variable genes.

```cpp
template <typename T, bool IsCSR>
void select_hvg(
    const Sparse<T, IsCSR>& matrix,
    Array<const Real> dispersions,
    Index n_top,
    Array<Index> out_indices
);
```

**Parameters**:
- `matrix` [in]: Expression matrix
- `dispersions` [in]: Dispersion values
- `n_top` [in]: Number of top genes to select
- `out_indices` [out]: Indices of selected genes (length = n_top)

**Example**:
```cpp
// Select top 2000 HVGs
constexpr Index N_TOP = 2000;
auto hvg_indices = memory::aligned_alloc<Index>(N_TOP);
Array<Index> hvg_view = {hvg_indices.get(), N_TOP};

kernel::hvg::select_hvg(matrix, disp_view, N_TOP, hvg_view);

// Use selected genes
for (Index i = 0; i < N_TOP; ++i) {
    Index gene_idx = hvg_view[i];
    // Process gene gene_idx
}
```

## Common Patterns

### Complete HVG Selection Pipeline

```cpp
void select_hvg_pipeline(
    const CSR& matrix,
    Index n_top,
    Real min_mean,
    Real max_mean,
    Array<Index>& hvg_indices
) {
    Index n_genes = matrix.cols();
    
    // 1. Compute means
    auto means = memory::aligned_alloc<Real>(n_genes);
    Array<Real> means_view = {means.get(), static_cast<Size>(n_genes)};
    compute_gene_means(matrix, means_view);
    
    // 2. Compute variances
    auto vars = memory::aligned_alloc<Real>(n_genes);
    Array<Real> vars_view = {vars.get(), static_cast<Size>(n_genes)};
    compute_gene_variances(matrix, means_view, vars_view);
    
    // 3. Compute dispersion
    auto dispersions = memory::aligned_alloc<Real>(n_genes);
    Array<Real> disp_view = {dispersions.get(), static_cast<Size>(n_genes)};
    kernel::hvg::compute_dispersion(matrix, means_view, vars_view, disp_view);
    
    // 4. Normalize dispersion
    kernel::hvg::normalize_dispersion(disp_view, min_mean, max_mean, means_view);
    
    // 5. Select top genes
    hvg_indices = memory::aligned_alloc<Index>(n_top);
    Array<Index> hvg_view = {hvg_indices.get(), n_top};
    kernel::hvg::select_hvg(matrix, disp_view, n_top, hvg_view);
}
```

### Filtering by Mean Expression

```cpp
void filter_by_mean(
    Array<Real> dispersions,
    Array<const Real> means,
    Real min_mean,
    Real max_mean
) {
    for (Index i = 0; i < dispersions.size(); ++i) {
        Real mean = means[i];
        if (mean < min_mean || mean > max_mean) {
            dispersions[i] = -std::numeric_limits<Real>::infinity();
        }
    }
}
```

## Performance Considerations

### SIMD Optimization

Dispersion computation uses SIMD for vectorized operations:

```cpp
// SIMD-accelerated dispersion
namespace s = scl::simd;
auto v_mean = s::Load(d, means.ptr + k);
auto v_var = s::Load(d, vars.ptr + k);
auto mask = s::Gt(v_mean, v_eps);
auto v_div = s::Div(v_var, v_mean);
auto v_res = s::IfThenElse(mask, v_div, v_zero);
s::Store(v_res, d, out_dispersion.ptr + k);
```

### Parallelization

Statistics computation is parallelized:

```cpp
// Parallel gene statistics
threading::parallel_for(0, n_genes, [&](size_t g) {
    compute_gene_stats(matrix, g, means[g], vars[g]);
});
```

## Configuration

```cpp
namespace scl::kernel::hvg::config {
    constexpr Real EPSILON = 1e-12;
    constexpr Size PREFETCH_DISTANCE = 16;
}
```

## Related Documentation

- [Normalization](./normalize.md) - Normalization operations
- [Feature Selection](./feature.md) - General feature selection
- [Kernels Overview](./overview.md) - General kernel usage
