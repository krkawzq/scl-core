---
title: Missing Value Imputation
description: High-performance sparse-aware imputation methods
---

# Missing Value Imputation

The `impute` kernel provides efficient imputation methods for single-cell data, optimized with SIMD and parallel processing.

## Overview

Imputation is used for:
- Filling missing values in sparse matrices
- Smoothing noisy data
- Improving downstream analysis quality

## Imputation Modes

```cpp
enum class ImputeMode {
    KNN,           // K-nearest neighbors
    WeightedKNN,   // Weighted KNN
    Diffusion,     // Diffusion-based
    MAGIC          // MAGIC algorithm
};
```

## Functions

### `impute`

Impute missing values in sparse matrix.

```cpp
template <typename T, bool IsCSR>
void impute(
    Sparse<T, IsCSR>& matrix,
    ImputeMode mode = ImputeMode::KNN,
    Index k = 15,
    Real bandwidth = config::DEFAULT_BANDWIDTH
);
```

**Parameters**:
- `matrix` [in,out]: Matrix to impute (modified in-place)
- `mode` [in]: Imputation method
- `k` [in]: Number of neighbors for KNN methods
- `bandwidth` [in]: Bandwidth for weighted methods

**Example**:
```cpp
#include "scl/kernel/impute.hpp"

// KNN imputation
kernel::impute::impute(
    matrix,
    ImputeMode::KNN,
    k=15
);

// Diffusion-based imputation
kernel::impute::impute(
    matrix,
    ImputeMode::Diffusion,
    k=30,
    bandwidth=1.0
);
```

## Configuration

```cpp
namespace scl::kernel::impute::config {
    constexpr Real DEFAULT_THRESHOLD = 0.0;
    constexpr Real DISTANCE_EPSILON = 1e-12;
    constexpr Real DEFAULT_BANDWIDTH = 1.0;
    constexpr Index DEFAULT_DIFFUSION_STEPS = 3;
    constexpr Real MIN_IMPUTED_VALUE = 1e-10;
    constexpr Size PARALLEL_THRESHOLD = 128;
}
```

## Related Documentation

- [Neighbors](./neighbors.md) - Neighbor search
- [Kernels Overview](./overview.md) - General kernel usage
