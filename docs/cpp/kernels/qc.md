---
title: Quality Control
description: QC metrics for single-cell data
---

# Quality Control

The `qc` kernel provides efficient quality control metrics computation for single-cell data.

## Overview

QC metrics are used for:
- Data quality assessment
- Cell filtering
- Gene filtering
- Batch effect detection

## Functions

### `compute_qc_metrics`

Compute QC metrics for cells.

```cpp
template <typename T, bool IsCSR>
void compute_qc_metrics(
    const Sparse<T, IsCSR>& matrix,
    Array<Real> total_counts,
    Array<Real> n_genes,
    Array<Real> pct_mito
);
```

**Parameters**:
- `matrix` [in]: Expression matrix
- `total_counts` [out]: Total counts per cell
- `n_genes` [out]: Number of expressed genes per cell
- `pct_mito` [out]: Percentage of mitochondrial genes

**Example**:
```cpp
#include "scl/kernel/qc.hpp"

auto total_counts = memory::aligned_alloc<Real>(matrix.rows());
auto n_genes = memory::aligned_alloc<Real>(matrix.rows());
auto pct_mito = memory::aligned_alloc<Real>(matrix.rows());

Array<Real> counts_view = {total_counts.get(), static_cast<Size>(matrix.rows())};
Array<Real> genes_view = {n_genes.get(), static_cast<Size>(matrix.rows())};
Array<Real> mito_view = {pct_mito.get(), static_cast<Size>(matrix.rows())};

kernel::qc::compute_qc_metrics(matrix, counts_view, genes_view, mito_view);
```

## Configuration

```cpp
namespace scl::kernel::qc::config {
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Real PCT_SCALE = 100.0;
}
```

## Related Documentation

- [Sparse Statistics](./sparse_stats.md) - Statistical operations
- [Kernels Overview](./overview.md) - General kernel usage
