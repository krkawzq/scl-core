---
title: Feature Statistics
description: Feature-level statistics computation
---

# Feature Statistics

The `feature` kernel provides efficient feature-level statistics computation.

## Overview

Feature statistics are used for:
- Gene-level summaries
- Feature selection
- Quality assessment
- Data exploration

## Functions

### `compute_feature_stats`

Compute statistics for each feature.

```cpp
template <typename T, bool IsCSR>
void compute_feature_stats(
    const Sparse<T, IsCSR>& matrix,
    Array<Real> means,
    Array<Real> variances,
    Array<Real> n_expressed
);
```

**Parameters**:
- `matrix` [in]: Expression matrix
- `means` [out]: Mean expression per feature
- `variances` [out]: Variance per feature
- `n_expressed` [out]: Number of cells expressing each feature

**Example**:
```cpp
#include "scl/kernel/feature.hpp"

Index n_genes = matrix.cols();
auto means = memory::aligned_alloc<Real>(n_genes);
auto vars = memory::aligned_alloc<Real>(n_genes);
auto n_expr = memory::aligned_alloc<Real>(n_genes);

Array<Real> means_view = {means.get(), static_cast<Size>(n_genes)};
Array<Real> vars_view = {vars.get(), static_cast<Size>(n_genes)};
Array<Real> n_expr_view = {n_expr.get(), static_cast<Size>(n_genes)};

kernel::feature::compute_feature_stats(matrix, means_view, vars_view, n_expr_view);
```

## Configuration

```cpp
namespace scl::kernel::feature::config {
    constexpr Size CHUNK_SIZE = 256;
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Real EPSILON = 1e-12;
}
```

## Related Documentation

- [HVG](./hvg.md) - Highly variable genes
- [Sparse Statistics](./sparse_stats.md) - Statistical operations
