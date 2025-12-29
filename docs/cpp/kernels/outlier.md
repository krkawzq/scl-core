---
title: Outlier Detection
description: Outlier and anomaly detection methods
---

# Outlier Detection

The `outlier` kernel provides efficient outlier detection methods for single-cell data.

## Overview

Outlier detection is used for:
- Quality control
- Ambient RNA detection
- Empty droplet detection
- Anomaly identification

## Functions

### `local_outlier_factor`

Compute Local Outlier Factor (LOF).

```cpp
template <typename T, bool IsCSR>
void local_outlier_factor(
    const Sparse<T, IsCSR>& distances,
    Array<Real> lof_scores,
    Index k = config::DEFAULT_K
);
```

**Parameters**:
- `distances` [in]: Distance matrix
- `lof_scores` [out]: LOF scores (higher = more outlier)
- `k` [in]: Number of neighbors

**Example**:
```cpp
#include "scl/kernel/outlier.hpp"

auto lof_scores = memory::aligned_alloc<Real>(n_cells);
Array<Real> lof_view = {lof_scores.get(), n_cells};

kernel::outlier::local_outlier_factor(distances, lof_view, k=20);
```

### `empty_drops`

Detect empty droplets (EmptyDrops algorithm).

```cpp
template <typename T, bool IsCSR>
void empty_drops(
    const Sparse<T, IsCSR>& matrix,
    Array<Real> p_values,
    Index min_umi = config::EMPTY_DROPS_MIN_UMI
);
```

## Configuration

```cpp
namespace scl::kernel::outlier::config {
    constexpr Size DEFAULT_K = 20;
    constexpr Real LOF_THRESHOLD = 1.5;
    constexpr Size EMPTY_DROPS_MIN_UMI = 100;
    constexpr Size PARALLEL_THRESHOLD = 256;
}
```

## Related Documentation

- [QC](./qc.md) - Quality control
- [Kernels Overview](./overview.md) - General kernel usage
