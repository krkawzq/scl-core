---
title: Sampling Strategies
description: Advanced sampling methods for large datasets
---

# Sampling Strategies

The `sampling` kernel provides efficient sampling methods for large single-cell datasets.

## Overview

Sampling strategies are used for:
- Geometric sketching for rare population preservation
- Density-preserving downsampling
- Landmark selection
- Representative cell selection

## Functions

### `geometric_sketching`

Geometric sketching for preserving rare populations.

```cpp
template <typename T, bool IsCSR>
void geometric_sketching(
    const Sparse<T, IsCSR>& data,
    Size target_size,
    Index* selected_indices,
    Size& n_selected,
    uint64_t seed = 0
);
```

**Parameters**:
- `data` [in]: Expression matrix
- `target_size` [in]: Target number of cells to select
- `selected_indices` [out]: Indices of selected cells
- `n_selected` [out]: Number of cells actually selected
- `seed` [in]: Random seed

**Example**:
```cpp
#include "scl/kernel/sampling.hpp"

constexpr Size TARGET_SIZE = 10000;
auto selected = memory::aligned_alloc<Index>(TARGET_SIZE);
Size n_selected = 0;

kernel::sampling::geometric_sketching(
    matrix, TARGET_SIZE,
    selected.get(), n_selected,
    seed=42
);
```

## Configuration

```cpp
namespace scl::kernel::sampling::config {
    constexpr Real EPSILON = 1e-10;
    constexpr Size DEFAULT_BINS = 64;
    constexpr Size MAX_ITERATIONS = 1000;
    constexpr Size PARALLEL_THRESHOLD = 256;
}
```

## Related Documentation

- [Kernels Overview](./overview.md) - General kernel usage
- [Neighbors](./neighbors.md) - Neighbor search
